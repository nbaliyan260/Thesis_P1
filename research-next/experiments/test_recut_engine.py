"""Offline unit tests: tiny random HF models, no downloads or remote code.

Run from this directory with the pinned environment: python -m pytest -q.
The reference is native HF incremental forward, not another engine instance.
These tests check exact elementwise equality (not signed-zero byte identity).
"""

from dataclasses import replace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache

from recut_engine import (
    HFIncrementalEngine,
    cache_nbytes,
    clone_cache,
    compare_caches,
    journal_nbytes,
    restore_layers,
    truncate_cache,
)


@pytest.fixture(params=["llama", "qwen2"])
def model(request):
    torch.manual_seed(112)
    common = dict(
        vocab_size=47,
        hidden_size=32,
        intermediate_size=56,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    if request.param == "llama":
        config = LlamaConfig(**common)
        config._attn_implementation = "eager"
        return LlamaForCausalLM(config).eval()
    config = Qwen2Config(**common, use_sliding_window=False)
    config._attn_implementation = "eager"
    return Qwen2ForCausalLM(config).eval()


@torch.no_grad()
def native_step(model, token, cache):
    # Deliberately let native HF infer position/mask, independent of the engine.
    return model(
        input_ids=torch.tensor([[token]], dtype=torch.long),
        past_key_values=cache,
        use_cache=True,
        return_dict=True,
    )


def test_native_incremental_equality_every_token(model):
    engine = HFIncrementalEngine(model)
    actual_cache, reference_cache = engine.empty_cache(), DynamicCache()
    for token in [1, 4, 12, 2, 0, 17, 31]:
        actual = engine.step(token, actual_cache, capture_cuts=range(engine.num_layers))
        expected = native_step(model, token, reference_cache)
        assert torch.equal(actual.logits, expected.logits)
        assert compare_caches(actual_cache, reference_cache)["equal"]
        assert actual.layers_executed == engine.num_layers
        assert len(actual.journals) == engine.num_layers


def test_prefill_is_native_incremental_not_batched(model):
    engine = HFIncrementalEngine(model)
    cache, logits = engine.prefill(torch.tensor([[1, 7, 9, 4]]))
    native = DynamicCache()
    for token in [1, 7, 9, 4]:
        result = native_step(model, token, native)
    assert torch.equal(logits, result.logits)
    assert compare_caches(cache, native)["equal"]


@pytest.mark.parametrize("fault_layer,cut", [(0, 0), (1, 1), (2, 1), (3, 3)])
@pytest.mark.parametrize("window", [1, 4])
def test_partial_replay_matches_native_with_lower_future_retained(model, fault_layer, cut, window):
    engine = HFIncrementalEngine(model)
    prefix = [1, 6, 11]
    fixed_inputs = [4, 9, 13, 22][:window]
    checkpoint, _ = engine.prefill(prefix)
    faulty = clone_cache(checkpoint)
    faulty.value_cache[fault_layer][0, 0, 0, 0] += 3.0
    journals = []
    for token in fixed_inputs:
        journals.append(engine.step(token, faulty, capture_cuts=range(engine.num_layers)).journals)

    native = DynamicCache()
    for token in prefix + fixed_inputs:
        clean_result = native_step(model, token, native)

    repaired = clone_cache(faulty)
    lower_before = [(x.clone(), y.clone()) for x, y in zip(repaired.key_cache[:cut], repaired.value_cache[:cut])]
    restore_layers(repaired, checkpoint, start_layer=cut)
    for j, token in enumerate(fixed_inputs):
        result = engine.step(
            token, repaired, position=len(prefix) + j, start_layer=cut,
            journal=journals[j][cut] if cut else None,
            fault_layer=fault_layer if cut else None,
        )
        assert result.layers_executed == engine.num_layers - cut
        for i in range(cut):
            assert torch.equal(repaired.key_cache[i], lower_before[i][0])
            assert torch.equal(repaired.value_cache[i], lower_before[i][1])
    assert torch.equal(result.logits, clean_result.logits)
    assert compare_caches(repaired, native)["equal"]
    # Continuation exercises the entire repaired state after the window.
    continued = engine.step(27, repaired)
    native_continued = native_step(model, 27, native)
    assert torch.equal(continued.logits, native_continued.logits)
    assert compare_caches(repaired, native)["equal"]


def test_slot_restore_can_match_tokens_but_not_cache(model):
    # Constant logits deliberately hide an internal state error from token tests.
    with torch.no_grad():
        model.lm_head.weight.zero_()
    engine = HFIncrementalEngine(model)
    checkpoint, _ = engine.prefill([1, 3, 5])
    clean, faulty = clone_cache(checkpoint), clone_cache(checkpoint)
    faulty.value_cache[0][0, 0, 0, 0] += 3.0
    for token in [7, 9]:
        good = engine.step(token, clean)
        bad = engine.step(token, faulty)
    assert good.next_token == bad.next_token
    assert torch.equal(good.logits, bad.logits)
    faulty.value_cache[0][0, 0, 0, 0] = checkpoint.value_cache[0][0, 0, 0, 0]
    assert not compare_caches(faulty, clean)["equal"]


def test_invalid_journal_provenance_rejected_before_mutation(model):
    engine = HFIncrementalEngine(model)
    checkpoint, _ = engine.prefill([1, 4])
    capture_cache = clone_cache(checkpoint)
    entries = engine.step(7, capture_cache, capture_cuts=[1, 2]).journals
    cache = clone_cache(checkpoint)
    invalid = [
        (dict(start_layer=1, fault_layer=1, journal=None), "requires a journal"),
        (dict(start_layer=2, fault_layer=1, journal=entries[2]), "localized fault"),
        (dict(start_layer=1, fault_layer=1, journal=replace(entries[1], token=8)), "provenance"),
        (dict(start_layer=1, fault_layer=1, journal=replace(entries[1], position=3)), "provenance"),
        (dict(start_layer=1, fault_layer=1, journal=replace(entries[1], engine_id=-1)), "provenance"),
    ]
    for kwargs, message in invalid:
        with pytest.raises(ValueError, match=message):
            engine.step(7, cache, position=2, **kwargs)
        assert compare_caches(cache, checkpoint)["equal"]


def test_unsafe_cut_negative_control_is_really_contaminated(model):
    engine = HFIncrementalEngine(model)
    checkpoint, _ = engine.prefill([1, 4])
    clean, faulty = clone_cache(checkpoint), clone_cache(checkpoint)
    faulty.value_cache[0][0, 0, 0, 0] += 3.0
    clean_entry = engine.step(7, clean, capture_cuts=[1]).journals[1]
    faulty_entry = engine.step(7, faulty, capture_cuts=[1]).journals[1]
    assert not torch.equal(clean_entry.hidden, faulty_entry.hidden)
    with pytest.raises(ValueError, match="localized fault"):
        engine.step(7, clone_cache(checkpoint), start_layer=1, journal=faulty_entry, fault_layer=0)


def test_truncate_every_layer_before_changed_next_input(model):
    engine = HFIncrementalEngine(model)
    checkpoint, _ = engine.prefill([1, 4])
    faulty = clone_cache(checkpoint)
    captured = []
    for token in [7, 9, 11]:
        captured.append(engine.step(token, faulty, capture_cuts=[2]).journals[2])
    repaired = clone_cache(faulty)
    restore_layers(repaired, checkpoint, start_layer=2)
    engine.step(7, repaired, position=2, start_layer=2, journal=captured[0], fault_layer=2)
    # Emulate a changed decision after processing token at position 2. No stale
    # journal is used for the changed input at position 3 or later.
    truncate_cache(repaired, 3)
    for token in [15, 19]:
        actual = engine.step(token, repaired)
    native = DynamicCache()
    for token in [1, 4, 7, 15, 19]:
        expected = native_step(model, token, native)
    assert torch.equal(actual.logits, expected.logits)
    assert compare_caches(repaired, native)["equal"]


def test_clone_truncate_restore_and_byte_accounting(model):
    engine = HFIncrementalEngine(model)
    cache, _ = engine.prefill([1, 5, 9])
    saved = clone_cache(cache)
    old_size = cache_nbytes(cache)
    assert old_size > 0
    result = engine.step(7, cache, capture_cuts=[1, 3])
    expected_bytes = 2 * model.config.hidden_size * model.dtype.itemsize
    assert journal_nbytes(result.journals) == expected_bytes
    assert journal_nbytes([result.journals]) == expected_bytes
    assert cache.get_seq_length() == 4 and saved.get_seq_length() == 3
    truncate_cache(cache, 3)
    assert compare_caches(cache, saved)["equal"]
    assert cache_nbytes(cache) == old_size
    # Logical byte counts coincide with newly owned tensor storage after crop.
    for tensor in cache.key_cache + cache.value_cache:
        assert tensor.untyped_storage().nbytes() == tensor.numel() * tensor.element_size()
    cache.key_cache[0][0, 0, 0, 0] += 1.0
    assert not compare_caches(cache, saved)["equal"]
    restore_layers(cache, saved)
    assert compare_caches(cache, saved)["equal"]
    with pytest.raises(ValueError, match="Cannot extend"):
        truncate_cache(cache, 5)
    truncate_cache(cache, 0)
    actual = engine.step(1, cache)
    expected = native_step(model, 1, DynamicCache())
    assert torch.equal(actual.logits, expected.logits)


def test_nonfinite_comparison_fails_closed(model):
    engine = HFIncrementalEngine(model)
    cache, _ = engine.prefill([1])
    cache.value_cache[0][0, 0, 0, 0] = float("nan")
    report = compare_caches(cache, clone_cache(cache))
    assert not report["equal"] and not report["finite"]


def test_reject_unready_cache_and_unsupported_runtime(model):
    engine = HFIncrementalEngine(model)
    cache, _ = engine.prefill([1, 4])
    with pytest.raises(ValueError, match="append-ready"):
        engine.step(7, cache, position=1)
    with pytest.raises(ValueError, match="evaluation"):
        model.train()
        engine.step(7, cache)
    model.eval()
    model.config._attn_implementation = "sdpa"
    with pytest.raises(ValueError, match="eager"):
        HFIncrementalEngine(model)


def test_noop_fault_and_full_suffix_replay_identity(model):
    engine = HFIncrementalEngine(model)
    checkpoint, _ = engine.prefill([1, 4])
    first, second = clone_cache(checkpoint), clone_cache(checkpoint)
    first.value_cache[2][0, 0, 0, 0] += 0.0
    for token in [7, 9, 11]:
        a = engine.step(token, first)
        b = engine.step(token, second, capture_cuts=[1, 2, 3])
        assert torch.equal(a.logits, b.logits)
    assert compare_caches(first, second)["equal"]
