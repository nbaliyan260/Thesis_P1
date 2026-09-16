"""Independent transactional-controller tests using tiny real HF decoders.

These are offline CPU control/correctness tests, not pretrained measurements.
Reference generation calls native HF forward directly rather than the recovery
routine. Two negative controls deliberately document the detector's blind spots.
The forced-divergence test is an explicitly synthetic control-flow diagnostic.
"""

from dataclasses import replace
from pathlib import Path
import sys

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from controller import TransactionalSession, WindowRejected
from recut_engine import HFIncrementalEngine, StepResult, clone_cache, compare_caches


PREFIX = [1, 3, 5]


def make_engine(family="llama", constant_logits=False):
    torch.manual_seed(112)
    common = dict(vocab_size=47, hidden_size=32, intermediate_size=56,
        num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, attention_dropout=0.0,
        tie_word_embeddings=False, pad_token_id=0, bos_token_id=1,
        eos_token_id=2)
    config = (LlamaConfig(**common) if family == "llama" else
              Qwen2Config(**common, use_sliding_window=False))
    config._attn_implementation = "eager"
    model = (LlamaForCausalLM(config) if family == "llama" else
             Qwen2ForCausalLM(config)).eval()
    if constant_logits:
        with torch.no_grad():
            model.lm_head.weight.zero_()
    return HFIncrementalEngine(model)


def make_session(engine=None, policy="sparse", cuts=(1, 2, 3)):
    engine = engine or make_engine()
    cache, logits = engine.prefill(PREFIX)
    initial = int(logits[0, -1].argmax())
    return TransactionalSession(engine, cache, initial, policy=policy, cuts=cuts)


@torch.no_grad()
def native_reference(model, steps):
    """Independent incremental execution, including initial pending-token offset."""
    cache = DynamicCache()
    for token in PREFIX:
        output = model(input_ids=torch.tensor([[token]]), past_key_values=cache,
                       use_cache=True, return_dict=True)
    token = int(output.logits[0, -1].argmax())
    tokens = []
    for _ in range(steps):
        output = model(input_ids=torch.tensor([[token]]), past_key_values=cache,
                       use_cache=True, return_dict=True)
        token = int(output.logits[0, -1].argmax())
        tokens.append(token)
    return tuple(tokens), cache, output.logits, token


def assert_clean(session, engine, steps):
    tokens, cache, logits, pending = native_reference(engine.model, steps)
    assert session.committed_tokens == tokens
    assert session.next_token == pending
    assert torch.equal(session.last_logits, logits)
    assert compare_caches(session.cache, cache)["equal"]
    # Continuation checks the pending-token convention and all restored state.
    actual, expected = clone_cache(session.cache), clone_cache(cache)
    with torch.no_grad():
        for _ in range(2):
            left = engine.model(input_ids=torch.tensor([[pending]]),
                past_key_values=actual, use_cache=True, return_dict=True)
            right = engine.model(input_ids=torch.tensor([[pending]]),
                past_key_values=expected, use_cache=True, return_dict=True)
            assert torch.equal(left.logits, right.logits)
            assert compare_caches(actual, expected)["equal"]
            pending = int(right.logits[0, -1].argmax())


def advance(tx, steps):
    for _ in range(steps):
        assert tx.step() is None


def corrupt_prefix(tx, layer, kind="value_cache", magnitude=3.0):
    collection = getattr(tx.working, kind)
    with torch.no_grad():
        collection[layer][0, 0, 0, :] += magnitude


def assert_rejected_without_commit(session, tx, before=(), windows=0):
    with pytest.raises(WindowRejected):
        tx.finish()
    assert session.poisoned
    assert session.committed_tokens == before
    assert session.windows_committed == windows


def test_speculation_is_private_until_successful_commit():
    engine = make_engine()
    session = make_session(engine)
    original, pending = clone_cache(session.cache), session.next_token
    tx = session.begin(4)
    for _ in range(4):
        assert tx.step() is None
        assert session.committed_tokens == ()
        assert session.windows_committed == 0
        assert session.next_token == pending
        assert compare_caches(session.cache, original)["equal"]
    result = tx.finish()
    assert result.tokens == session.committed_tokens
    assert session.windows_committed == 1
    assert not session.poisoned
    assert_clean(session, engine, 4)


@pytest.mark.parametrize("family", ["llama", "qwen2"])
def test_clean_window_matches_native_hf(family):
    engine = make_engine(family)
    session = make_session(engine)
    tx = session.begin(3)
    advance(tx, 3)
    result = tx.finish()
    assert result.detected_layers == ()
    assert not result.recovered
    assert result.first_divergence is None
    assert result.fallback_reason is None
    assert_clean(session, engine, 3)


@pytest.mark.parametrize("layers,expected_cut", [
    ((0,), 0), ((1,), 1), ((2,), 2), ((3,), 3), ((3, 1), 1), ((2, 0, 3), 0),
])
def test_detector_routes_by_earliest_changed_layer_not_injection_order(layers, expected_cut):
    engine = make_engine(constant_logits=True)
    session = make_session(engine)
    tx = session.begin(4)
    for layer in layers:
        corrupt_prefix(tx, layer)
    advance(tx, 4)
    result = tx.finish()
    assert result.detected_layers == tuple(sorted(layers))
    assert result.selected_cut == expected_cut
    assert result.replay_starts == (expected_cut,) * 4
    assert result.first_divergence is None
    assert result.recovered
    assert_clean(session, engine, 4)


def test_consecutive_windows_promote_only_corrected_state():
    engine = make_engine()
    session = make_session(engine)
    for window_index in range(3):
        tx = session.begin(3)
        if window_index == 1:
            corrupt_prefix(tx, 2)
        if window_index == 2:
            corrupt_prefix(tx, 0, "key_cache")
        advance(tx, 3)
        result = tx.finish()
        assert result.recovered == (window_index != 0)
        assert session.windows_committed == window_index + 1
        assert_clean(session, engine, 3 * (window_index + 1))


def test_staggered_prefix_faults_at_different_layers():
    engine = make_engine()
    session = make_session(engine)
    tx = session.begin(4)
    corrupt_prefix(tx, 3)
    advance(tx, 2)
    corrupt_prefix(tx, 1, "key_cache")
    advance(tx, 2)
    result = tx.finish()
    assert result.detected_layers == (1, 3)
    assert result.selected_cut == 1
    assert result.recovered
    assert_clean(session, engine, 4)


def test_full_policy_has_no_partial_replay():
    engine = make_engine()
    session = make_session(engine, policy="full", cuts=())
    tx = session.begin(4)
    corrupt_prefix(tx, 3)
    advance(tx, 4)
    result = tx.finish()
    assert result.detected_layers == (3,)
    assert result.selected_cut == 0
    assert result.replay_starts == (0, 0, 0, 0)
    assert result.recovered
    assert_clean(session, engine, 4)


def test_no_eligible_saved_cut_uses_full_replay():
    engine = make_engine()
    session = make_session(engine, cuts=(3,))
    tx = session.begin(3)
    corrupt_prefix(tx, 1)
    advance(tx, 3)
    result = tx.finish()
    assert result.selected_cut == 0
    assert result.replay_starts == (0, 0, 0)
    assert_clean(session, engine, 3)


@pytest.mark.parametrize("damage", ["hidden", "missing", "position", "engine_id"])
def test_untrusted_journal_forces_fresh_full_replay(damage):
    engine = make_engine()
    session = make_session(engine)
    tx = session.begin(4)
    corrupt_prefix(tx, 2)
    advance(tx, 4)
    entry = tx.journals[0][2]
    if damage == "hidden":
        with torch.no_grad():
            entry.hidden.add_(1.0)
    elif damage == "missing":
        del tx.journals[0][2]
    elif damage == "position":
        tx.journals[0][2] = replace(entry, position=entry.position + 1)
    else:
        tx.journals[0][2] = replace(entry, engine_id=-1)
    result = tx.finish()
    assert result.detected_layers == (2,)
    assert result.fallback_reason
    assert result.replay_starts == (0, 0, 0, 0)
    assert result.recovered
    assert_clean(session, engine, 4)


@pytest.mark.parametrize("damage", ["tensor", "seen_tokens", "layer_length"])
def test_checkpoint_corruption_fails_closed(damage):
    session = make_session()
    original = clone_cache(session.cache)
    tx = session.begin(3)
    advance(tx, 3)
    if damage == "tensor":
        with torch.no_grad():
            tx.checkpoint.value_cache[0][0, 0, 0, 0] += 1
    elif damage == "seen_tokens":
        tx.checkpoint._seen_tokens += 1
    else:
        tx.checkpoint.key_cache[1] = tx.checkpoint.key_cache[1][..., :-1, :]
    assert_rejected_without_commit(session, tx)
    assert compare_caches(session.cache, original)["equal"]


@pytest.mark.parametrize("damage", ["key_length", "value_length", "dtype"])
def test_malformed_working_layout_fails_closed(damage):
    session = make_session()
    tx = session.begin(3)
    advance(tx, 3)
    if damage == "key_length":
        tx.working.key_cache[1] = tx.working.key_cache[1][..., :-1, :]
    elif damage == "value_length":
        tx.working.value_cache[1] = tx.working.value_cache[1][..., :-1, :]
    else:
        tx.working.key_cache[1] = tx.working.key_cache[1].double()
    assert_rejected_without_commit(session, tx)


def test_replay_exception_cannot_commit_partial_results(monkeypatch):
    engine = make_engine()
    session = make_session(engine)
    tx = session.begin(3)
    corrupt_prefix(tx, 2)
    advance(tx, 3)

    def failure(*args, **kwargs):
        raise RuntimeError("deliberate replay failure")

    monkeypatch.setattr(engine, "step", failure)
    assert_rejected_without_commit(session, tx)


def test_later_rejection_preserves_previously_committed_window():
    engine = make_engine()
    session = make_session(engine)
    first = session.begin(3)
    advance(first, 3)
    first.finish()
    committed = session.committed_tokens
    known_good = clone_cache(session.cache)
    tx = session.begin(3)
    advance(tx, 3)
    with torch.no_grad():
        tx.checkpoint.value_cache[1][0, 0, 0, 0] += 1
    assert_rejected_without_commit(session, tx, committed, windows=1)
    assert compare_caches(session.cache, known_good)["equal"]
    assert_clean(session, engine, 3)


def test_checkpoint_working_and_committed_storage_are_independent():
    session = make_session()
    tx = session.begin(3)
    for name in ("key_cache", "value_cache"):
        for working, checkpoint, public in zip(getattr(tx.working, name),
                getattr(tx.checkpoint, name), getattr(session.cache, name)):
            assert len({working.data_ptr(), checkpoint.data_ptr(), public.data_ptr()}) == 3
    corrupt_prefix(tx, 2)
    assert compare_caches(session.cache, tx.checkpoint)["equal"]
    advance(tx, 3)
    tx.finish()


@pytest.mark.parametrize("divergence_index", [0, 1, 3])
def test_forced_divergence_gate_never_reuses_reconverged_history(monkeypatch, divergence_index):
    """White-box branch diagnostic, not a realistic KV fault experiment.

    Force one speculative output decision with an otherwise constant-logit
    model, so later token IDs reconverge while their preceding history differs.
    A separate persistent prefix change ensures detector-driven recovery runs.
    """
    engine = make_engine(constant_logits=True)
    session = make_session(engine, cuts=(2,))
    tx = session.begin(4)
    corrupt_prefix(tx, 2)
    original_step, calls = engine.step, 0

    def forced_decision(*args, **kwargs):
        nonlocal calls
        result = original_step(*args, **kwargs)
        if calls == divergence_index:
            logits = result.logits.clone()
            logits[0, -1, 1] = 1
            result = StepResult(logits, result.journals, result.layers_executed)
        calls += 1
        return result

    monkeypatch.setattr(engine, "step", forced_decision)
    advance(tx, 4)
    monkeypatch.setattr(engine, "step", original_step)
    result = tx.finish()
    assert result.first_divergence == divergence_index + 1
    assert result.replay_starts == ((2,) * (divergence_index + 1) +
                                    (0,) * (3 - divergence_index))
    assert result.tokens == (0, 0, 0, 0)
    assert_clean(session, engine, 4)


def test_suffix_only_corruption_is_explicit_detector_blind_spot():
    """Outside coverage: unchanged old prefix does not certify the suffix."""
    engine = make_engine()
    session = make_session(engine)
    tx = session.begin(3)
    advance(tx, 3)
    with torch.no_grad():
        tx.working.value_cache[2][0, 0, -1, :] += 3
    result = tx.finish()
    _, clean, _, _ = native_reference(engine.model, 3)
    assert result.detected_layers == ()
    assert not result.recovered
    assert not compare_caches(session.cache, clean)["equal"]


def test_reverted_prefix_fault_is_explicit_detector_blind_spot():
    """Outside coverage: a reverted direct error can leave a polluted suffix."""
    engine = make_engine()
    session = make_session(engine)
    tx = session.begin(3)
    corrupt_prefix(tx, 1)
    advance(tx, 3)
    with torch.no_grad():
        tx.working.value_cache[1][..., :len(PREFIX), :].copy_(tx.checkpoint.value_cache[1])
    result = tx.finish()
    _, clean, _, _ = native_reference(engine.model, 3)
    assert result.detected_layers == ()
    assert not result.recovered
    assert not compare_caches(session.cache, clean)["equal"]
