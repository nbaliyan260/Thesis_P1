"""Offline stage-3 regression and seal-equivalence tests.

Frozen stage-2 tests are imported under unique names, then their two controller
bindings are explicitly monkeypatched *only in tests*. The stage-2 implementation
and helpers are not edited. These tiny random-HF-model tests are not pretrained
measurements, field fault trials, or evidence of a general fault detector.
"""
from dataclasses import replace
import importlib.util
import inspect
import itertools
from pathlib import Path
import sys
from unittest.mock import patch

import pytest
import torch
from transformers.cache_utils import DynamicCache

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "experiments"))


def _unique_import(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # Required by dataclass annotation resolution.
    spec.loader.exec_module(module)
    return module


_stage2 = _unique_import("_recut_frozen_stage2_controller_test_reference", ROOT / "stage2/controller.py")
with patch.dict(sys.modules, {"controller": _stage2}):
    _frozen = _unique_import("_recut_frozen_stage2_tests_for_stage3", ROOT / "stage2/test_transactional.py")
_stage3 = _unique_import("_recut_stage3_controller_under_test", ROOT / "stage3/recut_controller.py")
from recut_engine import HFIncrementalEngine, JournalEntry, clone_cache, compare_caches
from prefix_detector import detect_prefix_layers

MODES = ("legacy", "batched")
DEVICES = ("cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is unavailable; no GPU result claimed")))


def _frozen_cases():
    """Expand only existing stage-2 parametrizations; no implementation patch."""
    cases = []
    for name, function in sorted(vars(_frozen).items()):
        if not name.startswith("test_") or not inspect.isfunction(function):
            continue
        variants = [{}]
        for mark in getattr(function, "pytestmark", []):
            if mark.name != "parametrize":
                raise RuntimeError("Review newly added frozen mark: " + mark.name)
            names, values = mark.args[:2]
            names = [n.strip() for n in names.split(",")] if isinstance(names, str) else list(names)
            expanded = []
            for value in values:
                value = (value,) if len(names) == 1 else value
                expanded.extend(dict(previous, **dict(zip(names, value))) for previous in variants)
            variants = expanded
        for index, kwargs in enumerate(variants):
            cases.append(pytest.param(name, kwargs, id=name.removeprefix("test_") + "-" + str(index)))
    return cases


@pytest.mark.parametrize("seal_mode", MODES)
@pytest.mark.parametrize("frozen_name,frozen_kwargs", _frozen_cases())
def test_frozen_stage2_behavior_on_new_controller(monkeypatch, seal_mode, frozen_name, frozen_kwargs):
    """Includes forced divergence/reconvergence and explicit detector blind spots."""
    def construct(*args, **kwargs):
        return _stage3.TransactionalSession(*args, **kwargs, seal_mode=seal_mode, profile=False)
    monkeypatch.setattr(_frozen, "TransactionalSession", construct)
    monkeypatch.setattr(_frozen, "WindowRejected", _stage3.WindowRejected)
    function = getattr(_frozen, frozen_name)
    kwargs = dict(frozen_kwargs)
    if "monkeypatch" in inspect.signature(function).parameters:
        kwargs["monkeypatch"] = monkeypatch
    function(**kwargs)


def _tensor_bytes(tensor):
    return tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()


def _seal_cache(device):
    cache = DynamicCache()
    cache.key_cache = [
        torch.arange(24, dtype=torch.float32, device=device).reshape(2, 3, 4).transpose(0, 1),
        torch.tensor([0.0, -0.0, 1.5], dtype=torch.bfloat16, device=device),
        torch.empty((0, 2), dtype=torch.float16, device=device),
    ]
    cache.value_cache = [
        torch.tensor([True, False], device=device),
        torch.arange(5, dtype=torch.int32, device=device),
        torch.tensor([float("inf"), float("-inf"), float("nan")], dtype=torch.float64, device=device),
    ]
    cache._seen_tokens = 7
    return cache


def _seal_journals(device):
    return {
        3: JournalEntry(11, 17, 3, torch.arange(12, device=device, dtype=torch.float32).reshape(3, 4).t(), 91),
        1: JournalEntry(11, 17, 1, torch.tensor([0.0, -0.0, 2.0], device=device, dtype=torch.bfloat16), 91),
    }


@pytest.mark.parametrize("seal_mode", MODES)
@pytest.mark.parametrize("device", DEVICES)
def test_cache_seal_exact_frozen_digest_for_layout_dtype_device_bytes(seal_mode, device):
    cache = _seal_cache(device)
    original = [_tensor_bytes(t) for t in cache.key_cache + cache.value_cache]
    counters = {}
    assert _stage3._cache_seal(cache, seal_mode=seal_mode, counters=counters) == _stage2._cache_seal(cache)
    assert [_tensor_bytes(t) for t in cache.key_cache + cache.value_cache] == original
    assert counters["seal_calls"] == 1
    assert counters["seal_logical_bytes"] == sum(t.numel() * t.element_size()
                                                  for t in cache.key_cache + cache.value_cache)
    if device == "cpu":
        assert counters.get("seal_transfer_calls", 0) == counters.get("seal_transfer_bytes", 0) == 0


@pytest.mark.parametrize("seal_mode", MODES)
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("empty", [False, True])
def test_journal_seal_exact_frozen_digest_and_sorted_metadata(seal_mode, device, empty):
    row = {} if empty else _seal_journals(device)
    owner = ("owner-alpha", 2)
    expected = _stage2._journal_seal(row, owner, 4)
    counters = {}
    assert _stage3._journal_seal(row, owner, 4, seal_mode=seal_mode, counters=counters) == expected
    assert _stage3._journal_seal(dict(reversed(list(row.items()))), owner, 4,
                                  seal_mode=seal_mode) == expected
    assert counters["seal_calls"] == 1
    assert counters.get("seal_logical_bytes", 0) == sum(e.hidden.numel() * e.hidden.element_size() for e in row.values())
    if device == "cpu":
        assert counters.get("seal_transfer_calls", 0) == counters.get("seal_transfer_bytes", 0) == 0


@pytest.mark.parametrize("seal_mode", MODES)
@pytest.mark.parametrize("device", DEVICES)
def test_payloads_match_individual_canonical_bytes_in_order(seal_mode, device):
    cache = _seal_cache(device)
    tensors = cache.key_cache + cache.value_cache
    payloads = _stage3._payloads(tensors, seal_mode, {})
    assert [bytes(payload) for payload in payloads] == [_tensor_bytes(t) for t in tensors]


@pytest.mark.parametrize("seal_mode", MODES)
@pytest.mark.parametrize("mutation", ["shape", "dtype", "seen", "layer_count", "kv_order", "signed_zero"])
def test_cache_seal_binds_metadata_and_raw_bytes(seal_mode, mutation):
    cache = DynamicCache()
    cache.key_cache = [torch.tensor([0.0, 1.0, 2.0, 3.0])]
    cache.value_cache = [torch.tensor([4.0, 5.0, 6.0, 7.0])]
    cache._seen_tokens = 4
    before = _stage2._cache_seal(cache)
    if mutation == "shape":
        cache.key_cache[0] = cache.key_cache[0].reshape(2, 2)
    elif mutation == "dtype":
        cache.key_cache[0] = cache.key_cache[0].view(torch.int32)
    elif mutation == "seen":
        cache._seen_tokens += 1
    elif mutation == "layer_count":
        cache.value_cache.append(torch.empty(0))
    elif mutation == "kv_order":
        cache.key_cache, cache.value_cache = cache.value_cache, cache.key_cache
    else:
        cache.key_cache[0][0] = -0.0
    expected = _stage2._cache_seal(cache)
    assert expected != before
    assert _stage3._cache_seal(cache, seal_mode=seal_mode) == expected


@pytest.mark.parametrize("seal_mode", MODES)
@pytest.mark.parametrize("mutation", ["owner", "index", "cut_key", "position", "token", "cut", "engine_id", "hidden"])
def test_journal_seal_binds_every_frozen_metadata_field(seal_mode, mutation):
    row, owner, index = _seal_journals("cpu"), ("owner-alpha", 2), 4
    before = _stage2._journal_seal(row, owner, index)
    if mutation == "owner":
        owner = ("owner-beta", 2)
    elif mutation == "index":
        index += 1
    elif mutation == "cut_key":
        row[2] = row.pop(1)
    elif mutation == "hidden":
        row[1].hidden[0] = -0.0
    else:
        row[1] = replace(row[1], **{mutation: getattr(row[1], mutation) + 1})
    expected = _stage2._journal_seal(row, owner, index)
    assert expected != before
    assert _stage3._journal_seal(row, owner, index, seal_mode=seal_mode) == expected


@pytest.mark.parametrize("seal_mode", MODES)
@pytest.mark.parametrize("row", [{1: object()}, {"1": JournalEntry(1, 1, 1, torch.zeros(1), 1)}])
def test_malformed_journal_seals_rejected_like_frozen_reference(seal_mode, row):
    with pytest.raises((ValueError, TypeError)):
        _stage2._journal_seal(row, ("owner", 0), 0)
    with pytest.raises((ValueError, TypeError)):
        _stage3._journal_seal(row, ("owner", 0), 0, seal_mode=seal_mode)


def _engine(family, device="cpu"):
    engine = _frozen.make_engine(family)
    if device != "cpu":
        engine = HFIncrementalEngine(engine.model.to(device))
    return engine


def _session(engine, mode, policy="sparse", profile=False):
    cache, logits = engine.prefill(_frozen.PREFIX)
    return _stage3.TransactionalSession(engine, cache, int(logits[0, -1].argmax()),
                                        policy=policy, cuts=(1, 2, 3), seal_mode=mode, profile=profile)


@torch.no_grad()
def _native_state(engine, steps):
    cache = DynamicCache()
    for token in _frozen.PREFIX:
        output = engine.model(input_ids=torch.tensor([[token]], device=engine.device),
                              past_key_values=cache, use_cache=True, return_dict=True)
    token = int(output.logits[0, -1].argmax())
    tokens = []
    for _ in range(steps):
        output = engine.model(input_ids=torch.tensor([[token]], device=engine.device),
                              past_key_values=cache, use_cache=True, return_dict=True)
        token = int(output.logits[0, -1].argmax())
        tokens.append(token)
    return tuple(tokens), cache, output.logits, token


@torch.no_grad()
def _assert_native_and_continuation(session, steps):
    engine = session.engine
    tokens, cache, logits, pending = _native_state(engine, steps)
    assert session.committed_tokens == tokens and session.next_token == pending
    assert torch.equal(session.last_logits, logits)
    assert compare_caches(session.cache, cache)["equal"]
    actual = clone_cache(session.cache)
    for _ in range(2):
        left = engine.model(input_ids=torch.tensor([[pending]], device=engine.device),
                            past_key_values=actual, use_cache=True, return_dict=True)
        right = engine.model(input_ids=torch.tensor([[pending]], device=engine.device),
                             past_key_values=cache, use_cache=True, return_dict=True)
        assert torch.equal(left.logits, right.logits)
        assert compare_caches(actual, cache)["equal"]
        pending = int(right.logits[0, -1].argmax())


@pytest.mark.parametrize("seal_mode,policy,family", itertools.product(MODES, ("full", "sparse"), ("llama", "qwen2")))
@pytest.mark.parametrize("device", DEVICES)
def test_clean_then_repeated_staggered_kv_fault_windows_match_native(seal_mode, policy, family, device):
    session = _session(_engine(family, device), seal_mode, policy)
    for window in range(3):
        tx = session.begin(4)
        if window:
            _frozen.corrupt_prefix(tx, 3, "key_cache", 2.0)
        _frozen.advance(tx, 2)
        if window:
            _frozen.corrupt_prefix(tx, 1, "value_cache", -3.0)
            _frozen.corrupt_prefix(tx, 3, "key_cache", 1.0)  # Repeated persistent mutation.
        _frozen.advance(tx, 2)
        result = tx.finish()
        assert result.detected_layers == ((1, 3) if window else ())
        assert result.selected_cut == (1 if policy == "sparse" and window else 0)
        assert result.recovered is bool(window)
        _assert_native_and_continuation(session, 4 * (window + 1))


def _public_snapshot(session):
    return (_stage2._cache_seal(session.cache), session.next_token,
            session.committed_tokens, session.windows_committed,
            None if session.last_logits is None else _tensor_bytes(session.last_logits))


@pytest.mark.parametrize("seal_mode", MODES)
@pytest.mark.parametrize("damage", ["checkpoint_nan", "checkpoint_signed_zero", "suffix_nan", "logits_inf", "working_dtype"])
def test_fail_closed_preserves_all_previously_committed_public_fields(seal_mode, damage):
    session = _session(_engine("llama"), seal_mode)
    first = session.begin(2)
    _frozen.advance(first, 2)
    first.finish()
    # Trusted setup for a byte-only mutation guard; no semantic-reference claim.
    if damage == "checkpoint_signed_zero":
        session.cache.key_cache[0][0, 0, 0, 0] = 0.0
    before = _public_snapshot(session)
    tx = session.begin(3)
    _frozen.advance(tx, 3)
    if damage == "checkpoint_nan":
        tx.checkpoint.value_cache[1][0, 0, 0, 0] = float("nan")
    elif damage == "checkpoint_signed_zero":
        tx.checkpoint.key_cache[0][0, 0, 0, 0] = -0.0
    elif damage == "suffix_nan":
        tx.working.value_cache[1][0, 0, -1, 0] = float("nan")
    elif damage == "logits_inf":
        tx._last_logits[0, 0, 0] = float("inf")
    else:
        tx.working.key_cache[1] = tx.working.key_cache[1].double()
    with pytest.raises(_stage3.WindowRejected):
        tx.finish()
    assert _public_snapshot(session) == before
    assert session.poisoned and session._active is None
    with pytest.raises(_stage3.WindowRejected):
        session.begin(1)


@pytest.mark.parametrize("seal_mode", MODES)
def test_signed_zero_seal_is_stricter_than_numeric_prefix_detector(seal_mode):
    cache = DynamicCache()
    cache.key_cache = [torch.zeros((1, 1, 2, 1))]
    cache.value_cache = [torch.ones((1, 1, 2, 1))]
    cache._seen_tokens = 2
    working = clone_cache(cache)
    working.key_cache[0][0, 0, 0, 0] = -0.0
    assert torch.equal(cache.key_cache[0], working.key_cache[0])
    assert detect_prefix_layers(working, cache) == []  # Numeric contract, not byte equality.
    assert _stage3._cache_seal(cache, seal_mode=seal_mode) != _stage3._cache_seal(working, seal_mode=seal_mode)


@pytest.mark.parametrize("seal_mode", MODES)
@pytest.mark.parametrize("damage", ["token", "cut", "dtype", "owner", "row_swap"])
def test_additional_journal_misbinding_forces_full_replay(seal_mode, damage):
    session = _session(_engine("qwen2"), seal_mode)
    tx = session.begin(3)
    _frozen.corrupt_prefix(tx, 2)
    _frozen.advance(tx, 3)
    entry = tx.journals[0][2]
    if damage == "token":
        tx.journals[0][2] = replace(entry, token=(entry.token + 1) % 47)
    elif damage == "cut":
        tx.journals[0][2] = replace(entry, cut=1)
    elif damage == "dtype":
        tx.journals[0][2] = replace(entry, hidden=entry.hidden.double())
    elif damage == "owner":
        tx._owner = ("misbound-owner", 7)
    else:
        tx.journals[0], tx.journals[1] = tx.journals[1], tx.journals[0]
    result = tx.finish()
    assert result.fallback_reason == "journal_integrity_failure"
    assert result.selected_cut == 0 and result.replay_starts == (0, 0, 0)
    _assert_native_and_continuation(session, 3)


@pytest.mark.parametrize("seal_mode", MODES)
@pytest.mark.parametrize("invalid", [0, -1, True, 1.5, 100])
def test_invalid_window_rejected_before_public_state_change(seal_mode, invalid):
    session = _session(_engine("llama"), seal_mode)
    before = _public_snapshot(session)
    with pytest.raises(ValueError):
        session.begin(invalid)
    assert _public_snapshot(session) == before
    assert not session.poisoned and session._active is None


@pytest.mark.parametrize("seal_mode", MODES)
@pytest.mark.parametrize("damage", ["nan", "inf", "token", "cut", "policy", "seal_mode"])
def test_initial_session_guard_rejects_invalid_inputs(seal_mode, damage):
    engine = _engine("llama")
    cache, logits = engine.prefill(_frozen.PREFIX)
    kwargs = dict(policy="sparse", cuts=(1,), seal_mode=seal_mode)
    token = int(logits[0, -1].argmax())
    if damage in ("nan", "inf"):
        cache.value_cache[0][0, 0, 0, 0] = float(damage)
    elif damage == "token":
        token = -1
    elif damage == "cut":
        kwargs["cuts"] = (True,)
    elif damage == "policy":
        kwargs["policy"] = "invalid"
    else:
        kwargs["seal_mode"] = "invalid"
    with pytest.raises(ValueError):
        _stage3.TransactionalSession(engine, cache, token, **kwargs)


@pytest.mark.parametrize("seal_mode,profile,policy", itertools.product(MODES, (False, True), ("full", "sparse")))
def test_profiling_and_counters_do_not_change_semantics(seal_mode, profile, policy):
    session = _session(_engine("llama"), seal_mode, policy, profile)
    tx = session.begin(3)
    _frozen.corrupt_prefix(tx, 2)
    _frozen.advance(tx, 3)
    result = tx.finish()
    assert result.metadata["seal_mode"] == seal_mode
    assert result.metadata["profile_enabled"] is profile
    assert result.metadata["seal_calls"] >= 3
    assert result.metadata["seal_transfer_calls"] == result.metadata["seal_transfer_bytes"] == 0
    if profile:
        assert result.metadata["profile_seconds"]
        assert all(value >= 0 for value in result.metadata["profile_seconds"].values())
    else:
        assert not result.metadata.get("profile_seconds")
    _assert_native_and_continuation(session, 3)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable; transfer batching not exercised")
def test_cuda_batched_transfer_accounting_and_device_binding():
    cache = _seal_cache("cuda")
    legacy, batched = {}, {}
    left = _stage3._cache_seal(cache, seal_mode="legacy", counters=legacy)
    right = _stage3._cache_seal(cache, seal_mode="batched", counters=batched)
    assert left == right == _stage2._cache_seal(cache)
    assert 0 < batched["seal_transfer_calls"] < legacy["seal_transfer_calls"]
    assert batched["seal_transfer_bytes"] == legacy["seal_transfer_bytes"]
    assert batched["seal_peak_batch_bytes"] == batched["seal_transfer_bytes"]
    assert right != _stage3._cache_seal(_seal_cache("cpu"), seal_mode="batched")
