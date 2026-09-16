"""Independent CPU-only RECUT diagnostic tests.

The Fraction toy is an analytic witness, not a language-model performance result.
Native-model tests use random tiny weights, no downloads and no generated code.
Only this file is owned by the independent reviewer.
"""

from fractions import Fraction as F

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM

from recut_engine import HFIncrementalEngine, clone_cache, compare_caches, restore_layers, truncate_cache


def toy_transition(cache, embedding=F(1), cut=0, hidden=None):
    """Two scalar uniform-attention residual layers; values are layer inputs."""
    state = [list(layer) for layer in cache]
    h = F(embedding) if cut == 0 else F(hidden)
    for layer in range(cut, 2):
        state[layer].append(h)
        h += sum(state[layer], F(0)) / len(state[layer])
    return state, h


def test_analytic_slot_restore_and_unsafe_cut_guaranteed():
    checkpoint = [[F(1)], [F(2)]]
    clean, clean_logit = toy_transition(checkpoint)
    faulty, faulty_logit = toy_transition([[F(3)], [F(2)]])
    assert clean == [[F(1), F(1)], [F(2), F(2)]]
    assert clean_logit == 4
    assert faulty == [[F(3), F(1)], [F(2), F(3)]]
    assert faulty_logit == F(11, 2)

    faulty[0][0] = checkpoint[0][0]
    assert faulty[0] == clean[0]
    assert faulty[1] != clean[1]  # Stale descendant survives local repair.

    # A forced unsafe cut receives the recorded contaminated layer-1 input.
    # This bypass belongs only to the mathematical negative control.
    unsafe_state, unsafe_logit = toy_transition(
        [[F(1), F(1)], [F(2)]], cut=1, hidden=F(3)
    )
    assert unsafe_state[1] == [F(2), F(3)]
    assert unsafe_logit == F(11, 2)
    assert unsafe_state != clean


def test_analytic_same_tokens_do_not_imply_same_state():
    clean, clean_logit = toy_transition([[F(1)], [F(2)]])
    faulty, faulty_logit = toy_transition([[F(3)], [F(2)]])
    faulty[0][0] = F(1)
    # A fixed dominant alternative logit gives the same decision in both runs.
    decision = lambda h: int(h > 10)
    assert decision(clean_logit) == decision(faulty_logit)
    assert clean != faulty


def test_analytic_immediate_divergence_changes_next_input():
    checkpoint = [[F(1)], [F(2)]]
    repaired_first, good_logit = toy_transition(checkpoint)
    _, bad_logit = toy_transition([[F(3)], [F(2)]])
    decision = lambda h: int(h > 5)
    assert decision(good_logit) == 0
    assert decision(bad_logit) == 1
    embeddings = {0: F(2), 1: F(7)}
    correct, _ = toy_transition(repaired_first, embeddings[0])
    discarded, _ = toy_transition(repaired_first, embeddings[1])
    assert correct != discarded
    assert correct[0][-1] == 2
    assert discarded[0][-1] == 7


def make_engine():
    torch.manual_seed(876)
    config = LlamaConfig(
        vocab_size=29, hidden_size=16, intermediate_size=24,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=48, attention_dropout=0.0,
        tie_word_embeddings=False,
    )
    config._attn_implementation = "eager"
    return HFIncrementalEngine(LlamaForCausalLM(config).eval())


def test_last_layer_one_step_slot_restore_can_fix_cache_not_logits():
    engine = make_engine()
    checkpoint, _ = engine.prefill([1, 3])
    clean, faulty = clone_cache(checkpoint), clone_cache(checkpoint)
    last = engine.num_layers - 1
    faulty.value_cache[last][0, 0, 0, 0] += 3.0
    expected = engine.step(7, clean)
    stale = engine.step(7, faulty)
    assert not torch.equal(expected.logits, stale.logits)
    faulty.value_cache[last][0, 0, 0, 0] = checkpoint.value_cache[last][0, 0, 0, 0]
    # Newly cached last-layer K/V are computed before its faulty attention read.
    assert compare_caches(clean, faulty)["equal"]
    assert not torch.equal(expected.logits, stale.logits)


def test_matching_local_metadata_after_reconvergence_does_not_certify_history():
    engine = make_engine()
    prefix, _ = engine.prefill([1, 3])

    old = clone_cache(prefix)
    engine.step(5, old)
    stale_entry = engine.step(7, old, capture_cuts=[1]).journals[1]

    changed_prefix = clone_cache(prefix)
    engine.step(11, changed_prefix)  # Earlier input differs.
    correct = clone_cache(changed_prefix)
    expected = engine.step(7, correct)  # Current token and position reconverge.

    mixed = clone_cache(correct)  # Lower layer is clean for the new history.
    restore_layers(mixed, changed_prefix, start_layer=1)
    wrong = engine.step(
        7, mixed, position=3, start_layer=1, journal=stale_entry, fault_layer=1
    )
    # Engine intentionally checks local provenance, not whole-history validity.
    # A recovery runner must prevent this call after its first divergence.
    assert not compare_caches(mixed, correct)["equal"]
    assert not torch.equal(wrong.logits, expected.logits)


def test_future_lower_suffix_requires_all_layer_crop_before_full_step():
    engine = make_engine()
    prefix, _ = engine.prefill([1, 3])
    speculative = clone_cache(prefix)
    entry = engine.step(5, speculative, capture_cuts=[1]).journals[1]
    engine.step(7, speculative)
    engine.step(9, speculative)
    repaired = clone_cache(speculative)
    restore_layers(repaired, prefix, start_layer=1)
    engine.step(5, repaired, position=2, start_layer=1, journal=entry, fault_layer=1)
    before = clone_cache(repaired)
    with pytest.raises(ValueError, match="append-ready"):
        engine.step(11, repaired, position=3, start_layer=0)
    assert compare_caches(repaired, before)["equal"]  # Rejected before mutation.
    truncate_cache(repaired, 3, start_layer=0)
    actual = engine.step(11, repaired, position=3)
    reference, expected = engine.prefill([1, 3, 5, 11])
    assert compare_caches(repaired, reference)["equal"]
    assert torch.equal(actual.logits, expected)


def test_journal_and_checkpoint_own_storage():
    engine = make_engine()
    checkpoint, _ = engine.prefill([1, 3])
    old = clone_cache(checkpoint)
    captured = engine.step(7, old, capture_cuts=[1]).journals[1]
    hidden_copy = captured.hidden.clone()
    checkpoint_copy = clone_cache(checkpoint)
    engine.step(9, old)
    restore_layers(old, checkpoint)
    assert torch.equal(captured.hidden, hidden_copy)
    assert compare_caches(checkpoint, checkpoint_copy)["equal"]
    for original, copied in zip(checkpoint.key_cache, old.key_cache):
        assert original.data_ptr() != copied.data_ptr()


def test_exact_elementwise_is_not_bytewise_signed_zero():
    left = torch.tensor([0.0], dtype=torch.float32)
    right = torch.tensor([-0.0], dtype=torch.float32)
    assert torch.equal(left, right)
    assert not torch.equal(left.view(torch.uint8), right.view(torch.uint8))


@pytest.mark.parametrize("cut", [1, 2])
@pytest.mark.parametrize("divergence_index", [0, 1, 3])
def test_runner_irreversible_gate_pending_output_and_exact_work(cut, divergence_index):
    """Force a coherent output-history branch without searching model behavior.

    This is a white-box runner control-flow test, not a prefix-KV-fault episode.
    Clean model logits are constant; one speculative decision is overridden.
    Subsequent steps consume that changed token and then reconverge in tokens.
    """
    from run_recut import Trajectory, decode, repair, verify

    engine = make_engine()
    with torch.no_grad():
        engine.model.lm_head.weight.zero_()
    checkpoint, prefix_logits = engine.prefill([1, 3, 5])
    initial = int(prefix_logits[0, -1].argmax())
    assert initial == 0
    window = 4
    reference = decode(engine, clone_cache(checkpoint), initial, window)
    reference_cont = decode(engine, clone_cache(reference.cache), reference.tokens[-1], 2)
    working = clone_cache(checkpoint)
    token, outputs, journals = initial, [], []
    for index in range(window):
        step = engine.step(token, working, capture_cuts=[1, 2])
        logits = step.logits.clone()
        if index == divergence_index:
            logits[0, -1, 1] = 1.0
        token = int(logits[0, -1].argmax())
        outputs.append(token)
        journals.append(step.journals)
    faulty = Trajectory(working, outputs, logits, journals, [0] * window)
    assert faulty.tokens[divergence_index] == 1
    assert all(t == 0 for i, t in enumerate(faulty.tokens) if i != divergence_index)
    # Future journals are unusable after divergence even when tokens reconverge.
    for index in range(divergence_index + 1, window):
        faulty.journals[index] = {}
    repaired = repair(engine, clone_cache(faulty.cache), checkpoint, initial,
                      faulty, fault_layer=2, cut=cut)
    assert repaired.first_divergence == divergence_index + 1
    assert repaired.starts == [cut] * (divergence_index + 1) + [0] * (window - divergence_index - 1)
    assert sum(engine.num_layers - start for start in repaired.starts) == (
        engine.num_layers * window - cut * (divergence_index + 1)
    )
    assert repaired.tokens[-1] == reference.tokens[-1] == 0
    assert repaired.cache.get_seq_length() == 3 + window
    assert verify(engine, repaired, reference, reference_cont)["all_equal"]


def test_runner_decode_verify_continuation_has_correct_offset():
    from run_recut import decode, verify

    engine = make_engine()
    checkpoint, logits = engine.prefill([1, 3, 5])
    initial = int(logits[0, -1].argmax())
    reference = decode(engine, clone_cache(checkpoint), initial, 3)
    assert reference.cache.get_seq_length() == 6
    # Native-forward reference uses initial + all outputs except the pending last.
    native = engine.empty_cache()
    for token in [1, 3, 5, initial] + reference.tokens[:-1]:
        expected = engine.model(input_ids=torch.tensor([[token]]),
                                past_key_values=native, use_cache=True)
    assert compare_caches(reference.cache, native)["equal"]
    assert torch.equal(reference.logits, expected.logits)
    continuation = decode(engine, clone_cache(reference.cache), reference.tokens[-1], 2)
    assert continuation.cache.get_seq_length() == 8
    assert verify(engine, reference, reference, continuation)["all_equal"]


@pytest.mark.parametrize("layer_fraction,kind,fault", [
    (0.0, "K", "scalar"), (0.0, "V", "vector"),
    (0.5, "K", "vector"), (0.5, "V", "scalar"),
    (1.0, "V", "vector"), (1.0, "K", "noop"),
])
def test_runner_run_case_cpu_fairness_and_branch_isolation(layer_fraction, kind, fault):
    from run_recut import native_parity, run_case

    class StubTokenizer:
        def encode(self, text, add_special_tokens=True):
            return [1, 3, 5, 7]

    engine = make_engine()
    assert native_parity(engine, [1, 3, 5, 7] * 2, decode_steps=6)["passed"]
    case = dict(id=f"independent-{layer_fraction}-{kind}-{fault}", prompt="unit only",
                prefix_tokens=8, window=4, layer_fraction=layer_fraction,
                kind=kind, fault=fault, seed=1234)
    config = dict(repetitions=2, overhead_repetitions=2)
    row, audit = run_case(engine, StubTokenizer(), case, config, capture_audit=True)
    assert row["valid_repairs_equal"]
    assert len(row["timings"]) == 6
    assert len(row["overhead_trials"]) == 6
    assert {item["method"] for item in row["timings"]} == {"full", "sparse", "all"}
    assert all(item["checks"]["all_equal"] for item in row["timings"])
    assert all(item["verified_equal"] for item in row["overhead_trials"])
    assert all(item["seconds"] >= 0 for item in row["timings"])
    assert all(item["peak_allocated_bytes"] == 0 for item in row["timings"])
    # Raw audit snapshots still agree after all subsequent recovery branches.
    for name in ("full", "sparse", "all"):
        assert audit[name]["tokens"] == audit["reference"]["tokens"]
        for kind_name in ("keys", "values"):
            for actual, expected in zip(audit[name][kind_name], audit["reference"][kind_name]):
                assert torch.equal(actual, expected)
    if fault == "noop":
        assert row["methods"]["none"]["all_equal"]
        assert row["methods"]["slotonly"]["all_equal"]
    assert row["timings"][0]["order"] != row["timings"][3]["order"]


def test_runner_partial_missing_journal_fails_closed():
    from run_recut import decode, repair

    engine = make_engine()
    checkpoint, logits = engine.prefill([1, 3, 5])
    initial = int(logits[0, -1].argmax())
    faulty = decode(engine, clone_cache(checkpoint), initial, 2, capture_cuts=[])
    with pytest.raises(KeyError):
        repair(engine, clone_cache(faulty.cache), checkpoint, initial,
               faulty, fault_layer=2, cut=1)


def supplemental_engine(family):
    if family == "llama":
        return make_engine()
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(321)
    config = Qwen2Config(vocab_size=29, hidden_size=16, intermediate_size=24,
        num_hidden_layers=3, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=48, attention_dropout=0.0, use_sliding_window=False)
    config._attn_implementation = "eager"
    return HFIncrementalEngine(Qwen2ForCausalLM(config).eval())


def test_supplemental_prefix_views_restore_whole_layer_without_coordinate():
    from run_recut import decode
    from run_optimized_replay import prefix_views, unchanged
    engine = make_engine()
    checkpoint, logits = engine.prefill([1, 3, 5])
    checkpoint_guard = clone_cache(checkpoint)
    faulty = clone_cache(checkpoint)
    # Two faults in different coordinates/kinds: helper receives only layer 1.
    faulty.key_cache[1][0, 0, 0, 0] += 4.0
    faulty.value_cache[1][0, 1, 1, 2] -= 3.0
    decode(engine, faulty, int(logits[0, -1].argmax()), 3)
    source_guard = clone_cache(faulty)
    working = clone_cache(faulty)
    pointers = [(x.untyped_storage().data_ptr(), x.untyped_storage().nbytes())
                for x in working.key_cache]
    owners = prefix_views(working, checkpoint, 1)
    assert working._seen_tokens == 3
    for layer in range(3):
        assert working.get_seq_length(layer) == 3
        assert torch.equal(working.key_cache[layer], checkpoint.key_cache[layer])
        assert torch.equal(working.value_cache[layer], checkpoint.value_cache[layer])
        if layer != 1:
            assert working.key_cache[layer].untyped_storage().data_ptr() == pointers[layer][0]
            assert working.key_cache[layer].untyped_storage().nbytes() == pointers[layer][1]
            assert owners[0][layer].shape[-2] == 6  # Backing suffix not released.
        else:
            assert working.key_cache[layer].data_ptr() != checkpoint.key_cache[layer].data_ptr()
    assert unchanged(checkpoint, checkpoint_guard)
    assert unchanged(faulty, source_guard)


@pytest.mark.parametrize("family", ["llama", "qwen2"])
@pytest.mark.parametrize("layer", [0, 1, 2])
@pytest.mark.parametrize("window", [1, 4])
def test_supplemental_full_baselines_exact_and_immutable(family, layer, window):
    from run_recut import decode, verify
    from run_optimized_replay import alias_checkpoint, checkpoint_alias_full, layer_view_full, unchanged
    engine = supplemental_engine(family)
    checkpoint, logits = engine.prefill([1, 3, 5])
    checkpoint_guard = clone_cache(checkpoint)
    initial = int(logits[0, -1].argmax())
    reference = decode(engine, clone_cache(checkpoint), initial, window)
    continuation = decode(engine, clone_cache(reference.cache), reference.tokens[-1], 2)
    faulty_cache = clone_cache(checkpoint)
    faulty_cache.value_cache[layer][0, 0, 0, 0] += 3.0
    faulty = decode(engine, faulty_cache, initial, window, range(1, 3))
    faulty_guard = clone_cache(faulty.cache)

    alias = alias_checkpoint(checkpoint)
    assert alias.key_cache is not checkpoint.key_cache
    assert alias.value_cache is not checkpoint.value_cache
    assert alias._seen_tokens == checkpoint._seen_tokens
    for x, y in zip(alias.key_cache, checkpoint.key_cache):
        assert x.data_ptr() == y.data_ptr()
    engine.step(initial, alias)
    assert unchanged(checkpoint, checkpoint_guard)
    for x, y in zip(alias.key_cache, checkpoint.key_cache):
        assert x.data_ptr() != y.data_ptr()  # Out-of-place cat replaced each alias.

    candidates = [
        layer_view_full(engine, clone_cache(faulty.cache), checkpoint, initial, faulty, layer),
        checkpoint_alias_full(engine, checkpoint, initial, faulty),
    ]
    for recovered in candidates:
        assert recovered.starts == [0] * window
        assert verify(engine, recovered, reference, continuation)["all_equal"]
        assert unchanged(checkpoint, checkpoint_guard)
        assert unchanged(faulty.cache, faulty_guard)


@pytest.mark.parametrize("family", ["llama", "qwen2"])
def test_supplemental_matrix_reconstruction_counterbalance_and_no_retuning(family):
    from run_recut import run_case
    from run_optimized_replay import METHODS, run_sensitivity_case
    class StubTokenizer:
        def encode(self, text, add_special_tokens=True):
            return [1, 3, 5, 7]
    engine = supplemental_engine(family)
    case = dict(id="supplement-unit", prompt="unit", prefix_tokens=8,
        window=4, layer_fraction=0.5, kind="V", fault="vector", seed=119)
    primary, _ = run_case(engine, StubTokenizer(), case,
                         dict(repetitions=1, overhead_repetitions=1))
    row = run_sensitivity_case(engine, StubTokenizer(), case, primary, repetitions=5)
    assert row["valid"] and row["reconstruction_matches_primary"]
    assert len(row["timings"]) == 25
    for method in METHODS:
        timings = [x for x in row["timings"] if x["method"] == method]
        assert len(timings) == 5
        assert {x["order"].index(method) for x in timings} == set(range(5))
        for item in timings:
            assert item["checks"]["all_equal"]
            assert item["checks"]["checkpoint_unchanged"]
            assert item["checks"]["initial_faulty_unchanged"]
    inconsistent = dict(primary, initial_token=(primary["initial_token"] + 1) % 29)
    with pytest.raises(RuntimeError, match="differs from primary"):
        run_sensitivity_case(engine, StubTokenizer(), case, inconsistent, repetitions=1)


def test_supplemental_immutable_guard_distinguishes_signed_zero_and_nan_bits():
    from run_optimized_replay import unchanged
    engine = make_engine()
    original, _ = engine.prefill([1])
    original.value_cache[0][0, 0, 0, 0] = float("nan")
    assert unchanged(original, clone_cache(original))
    original.value_cache[0][0, 0, 0, 0] = 0.0
    changed = clone_cache(original)
    changed.value_cache[0][0, 0, 0, 0] = -0.0
    assert not unchanged(original, changed)


def test_prefix_detector_localizes_and_exposes_suffix_and_transient_blindspots():
    from run_recut import decode
    from prefix_detector import detect_prefix_layers, measure_prefix_detection
    engine = make_engine()
    checkpoint, logits = engine.prefill([1, 3, 5])
    initial = int(logits[0, -1].argmax())
    clean = decode(engine, clone_cache(checkpoint), initial, 3)
    assert detect_prefix_layers(clean.cache, checkpoint) == []
    suffix_fault = clone_cache(clean.cache)
    suffix_fault.value_cache[1][0, 0, 3, 0] += 2.0
    assert detect_prefix_layers(suffix_fault, checkpoint) == []
    assert not compare_caches(suffix_fault, clean.cache)["equal"]

    faulty_cache = clone_cache(checkpoint)
    faulty_cache.value_cache[0][0, 0, 0, 0] += 3.0
    faulty = decode(engine, faulty_cache, initial, 3)
    assert detect_prefix_layers(faulty.cache, checkpoint) == [0]
    # Fault returns to its original bytes, but its derived suffix remains stale.
    faulty.cache.value_cache[0][0, 0, 0, 0] = checkpoint.value_cache[0][0, 0, 0, 0]
    assert detect_prefix_layers(faulty.cache, checkpoint) == []
    assert not compare_caches(faulty.cache, clean.cache)["equal"]

    multiple = clone_cache(clean.cache)
    multiple.key_cache[0][0, 0, 0, 0] += 1.0
    multiple.value_cache[2][0, 0, 0, 0] += 1.0
    measured = measure_prefix_detection(engine, multiple, checkpoint)
    assert measured["detected_layers"] == [0, 2]
    assert measured["route"] == "reject_multiple"
    assert len(measured["timings"]) == 5
    assert measured["logical_bytes_read"] == 2 * measured["checkpoint_payload_bytes"]


def test_supplemental_no_alarm_fallback_noops_and_multiple_fail_closed(monkeypatch):
    from run_recut import run_case
    import run_optimized_replay as supplemental
    class StubTokenizer:
        def encode(self, text, add_special_tokens=True):
            return [1, 3, 5, 7]
    engine = make_engine()
    with torch.no_grad():
        engine.decoder.layers[1].self_attn.k_proj.weight.zero_()
    case = dict(id="zero-change-unit", prompt="unit", prefix_tokens=8,
        window=2, layer_fraction=0.5, kind="K", fault="scalar", seed=119)
    prior, _ = run_case(engine, StubTokenizer(), case, dict(repetitions=1, overhead_repetitions=1))
    assert prior["fault"]["changed_elements"] == 0
    result = supplemental.run_sensitivity_case(engine, StubTokenizer(), case, prior, repetitions=1)
    assert result["valid"]
    assert result["detected_layer"] is None
    assert result["detection"]["route"] == "no_alarm_full"
    assert all(item["checks"]["start_layers"] == [0, 0] for item in result["timings"])

    noop = dict(case, id="noop-detector-unit", fault="noop")
    noop_prior, _ = run_case(engine, StubTokenizer(), noop, dict(repetitions=1, overhead_repetitions=1))
    detection = supplemental.run_sensitivity_case(engine, StubTokenizer(), noop, noop_prior, detection_only=True)
    assert detection["valid"] and detection["detection_only"]
    assert detection["timings"] == []
    assert detection["detection"]["detected_layers"] == []
    assert len(detection["detection"]["timings"]) == 5

    monkeypatch.setattr(supplemental, "measure_prefix_detection",
                        lambda *args, **kwargs: {"detected_layers": [0, 2]})
    with pytest.raises(RuntimeError, match="Multiple changed layers"):
        supplemental.run_sensitivity_case(engine, StubTokenizer(), case, prior, repetitions=1)
