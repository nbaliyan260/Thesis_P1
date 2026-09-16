"""Transactional RECUT with batched, canonical cryptographic seal transfers.

This preserves the stage-2 bounded persistent-prefix fault model. No fault
location is accepted by the public API. The trusted initial state, model,
metadata/seals, and guard/recovery/commit execution remain assumptions. Suffix-
only, reverted, pre-checkpoint, and computation-origin faults are excluded.
Seals detect post-capture mutation, not semantic correctness at capture.

The batched seal has exactly the stage-2 SHA256 byte stream; only its device-to-
host transfer schedule changes. It requires a transient contiguous device batch
and host copy. Transfer counters describe seal payloads, not every transfer in
the controller, and do not measure allocator peak memory. Optional component
profiling adds explicit synchronizations and is for diagnostics only.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
import uuid

import torch
from transformers.cache_utils import DynamicCache

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from recut_engine import HFIncrementalEngine, JournalEntry, cache_nbytes, clone_cache, journal_nbytes, truncate_cache
from prefix_detector import detect_prefix_layers


class WindowRejected(RuntimeError):
    def __init__(self, reason, metadata=None):
        super().__init__(reason)
        self.reason = reason
        self.metadata = {} if metadata is None else metadata


@dataclass(frozen=True)
class WindowResult:
    tokens: tuple[int, ...]
    detected_layers: tuple[int, ...]
    selected_cut: int
    replay_starts: tuple[int, ...]
    first_divergence: int | None
    fallback_reason: str | None
    recovered: bool
    metadata: dict


def _sync(engine):
    if engine.device.type == "cuda":
        torch.cuda.synchronize(engine.device)


def _increment(counters, key, amount):
    if counters is not None:
        counters[key] = counters.get(key, 0) + amount


def _layout_bytes(tensor):
    return json.dumps([list(tensor.shape), str(tensor.dtype), str(tensor.device)],
                      separators=(",", ":")).encode()


def _tensor_update(digest, tensor):
    """Original stage-2 canonical update, retained for explicit compatibility."""
    digest.update(_layout_bytes(tensor))
    digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())


def _payloads(tensors, seal_mode, counters):
    """Return CPU byte views in input order, grouping copies by source device.

    Each memoryview retains its NumPy exporter, which retains its CPU tensor.
    Hashing slices therefore requires neither individual .cpu() calls nor
    per-tensor Python byte-string materialization in batched mode.
    """
    if seal_mode not in ("batched", "legacy"):
        raise ValueError("seal_mode must be batched or legacy")
    sizes = [tensor.numel() * tensor.element_size() for tensor in tensors]
    _increment(counters, "seal_logical_bytes", sum(sizes))
    if seal_mode == "legacy":
        result = []
        for tensor, size in zip(tensors, sizes):
            if tensor.device.type != "cpu":
                _increment(counters, "seal_transfer_calls", 1)
                _increment(counters, "seal_transfer_bytes", size)
                if counters is not None:
                    counters["seal_peak_batch_bytes"] = max(counters.get("seal_peak_batch_bytes", 0), size)
            result.append(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())
        return result
    result = [None] * len(tensors)
    groups = {}
    for index, tensor in enumerate(tensors):
        if tensor.device.type == "cpu":
            result[index] = memoryview(tensor.detach().contiguous().view(torch.uint8).reshape(-1).numpy())
        else:
            groups.setdefault(tensor.device, []).append(index)
    for indices in groups.values():
        chunks = [tensors[index].detach().contiguous().view(torch.uint8).reshape(-1) for index in indices]
        packed = torch.cat(chunks) if len(chunks) > 1 else chunks[0]
        host = memoryview(packed.cpu().numpy())
        size = sum(sizes[index] for index in indices)
        _increment(counters, "seal_transfer_calls", 1)
        _increment(counters, "seal_transfer_bytes", size)
        if counters is not None:
            counters["seal_peak_batch_bytes"] = max(counters.get("seal_peak_batch_bytes", 0), size)
        offset = 0
        for index in indices:
            result[index] = host[offset:offset + sizes[index]]
            offset += sizes[index]
    return result


def _cache_seal(cache, seal_mode="batched", counters=None):
    if seal_mode not in ("batched", "legacy"):
        raise ValueError("seal_mode must be batched or legacy")
    tensors = list(cache.key_cache) + list(cache.value_cache)
    payloads = _payloads(tensors, seal_mode, counters) if seal_mode == "batched" else None
    _increment(counters, "seal_calls", 1)
    digest = hashlib.sha256()
    digest.update(str((len(cache.key_cache), len(cache.value_cache), int(cache._seen_tokens))).encode())
    offset = 0
    for name in ("key_cache", "value_cache"):
        digest.update(name.encode())
        for tensor in getattr(cache, name):
            digest.update(_layout_bytes(tensor))
            # Legacy copies, hashes, and releases one tensor at a time just as
            # stage 2 did; do not accidentally retain all host byte strings in
            # the baseline while evaluating the batching optimization.
            digest.update(payloads[offset] if payloads is not None else _payloads([tensor], "legacy", counters)[0])
            offset += 1
    return digest.hexdigest()


def _journal_seal(row, owner, index, seal_mode="batched", counters=None):
    if seal_mode not in ("batched", "legacy"):
        raise ValueError("seal_mode must be batched or legacy")
    entries = sorted(row.items())
    for cut, entry in entries:
        if type(entry) is not JournalEntry or type(cut) is not int:
            raise ValueError("Malformed journal entry")
    payloads = _payloads([entry.hidden for _, entry in entries], seal_mode, counters) if seal_mode == "batched" else None
    _increment(counters, "seal_calls", 1)
    digest = hashlib.sha256()
    digest.update(json.dumps([owner, index], separators=(",", ":")).encode())
    for ordinal, (cut, entry) in enumerate(entries):
        digest.update(json.dumps([cut, entry.position, entry.token, entry.cut, entry.engine_id]).encode())
        digest.update(_layout_bytes(entry.hidden))
        digest.update(payloads[ordinal] if payloads is not None else _payloads([entry.hidden], "legacy", counters)[0])
    return digest.hexdigest()


def _validate_cache(engine, cache, length):
    if type(cache) is not DynamicCache or len(cache.key_cache) != engine.num_layers or len(cache.value_cache) != engine.num_layers:
        raise ValueError("Cache layer/type mismatch")
    if int(cache._seen_tokens) != length:
        raise ValueError("Cache length counter mismatch")
    config = engine.model.config
    shape = (1, config.num_key_value_heads, length, config.hidden_size // config.num_attention_heads)
    for tensor in cache.key_cache + cache.value_cache:
        if tuple(tensor.shape) != shape or tensor.dtype != engine.dtype or tensor.device != engine.device:
            raise ValueError("Cache layout mismatch")


def _finite(cache, logits=None):
    values = cache.key_cache + cache.value_cache + ([] if logits is None else [logits])
    return bool(torch.stack([torch.isfinite(t).all() for t in values]).all().item())


def _alias(checkpoint):
    # The trusted recovery implementation appends using out-of-place torch.cat.
    result = DynamicCache()
    result.key_cache = list(checkpoint.key_cache)
    result.value_cache = list(checkpoint.value_cache)
    result._seen_tokens = int(checkpoint._seen_tokens)
    return result


class TransactionalSession:
    """Single-owner in-process windows, not concurrent or crash-durable commits.

    Only `committed_tokens` is the public transcript. Transaction internals are
    exposed for controlled fault tests. The initial state/pending token must be
    trusted, as must periods between transactions and guard/recovery execution.
    `legacy` is a per-tensor-transfer ablation of the same seal semantics.
    `profile=True` is a separately synchronized diagnostic, not primary timing.
    """
    def __init__(self, engine: HFIncrementalEngine, cache, initial_token, *, policy="sparse", cuts=(),
                 seal_mode="batched", profile=False):
        if policy not in ("sparse", "full"):
            raise ValueError("policy must be sparse or full")
        if seal_mode not in ("batched", "legacy"):
            raise ValueError("seal_mode must be batched or legacy")
        if type(profile) is not bool:
            raise ValueError("profile must be boolean")
        if type(initial_token) is not int or not 0 <= initial_token < engine.model.config.vocab_size:
            raise ValueError("Invalid initial pending token")
        cuts = tuple(sorted(set(cuts)))
        if any(type(c) is not int or not 0 < c < engine.num_layers for c in cuts):
            raise ValueError("Cuts must be nonzero valid decoder layers")
        length = int(cache.get_seq_length())
        if length <= 0:
            raise ValueError("A trusted nonempty prefix is required")
        _validate_cache(engine, cache, length)
        if not _finite(cache):
            raise ValueError("Initial cache must be finite")
        self.engine, self.cache, self.next_token = engine, cache, initial_token
        self.policy, self.cuts = policy, cuts if policy == "sparse" else ()
        self.seal_mode, self.profile = seal_mode, profile
        self.committed_tokens = ()
        self.windows_committed = 0
        self.poisoned = False
        self.last_logits = None
        self._active = None
        self._owner = uuid.uuid4().hex

    def begin(self, window):
        if self.poisoned:
            raise WindowRejected("Session is poisoned")
        if self._active is not None:
            raise RuntimeError("A window is already active")
        if type(window) is not int or window <= 0:
            raise ValueError("Window length must be positive")
        if int(self.cache.get_seq_length()) + window > self.engine.model.config.max_position_embeddings:
            raise ValueError("Window exceeds model context")
        try:
            transaction = WindowTransaction(self, window)
        except Exception as exc:
            self.poisoned = True
            raise WindowRejected("Window initialization failed: " + str(exc)) from exc
        self._active = transaction
        return transaction


class WindowTransaction:
    def __init__(self, session, window):
        self.session, self.engine, self.window = session, session.engine, window
        self.prefix = int(session.cache.get_seq_length())
        self._metadata = {"window_index": session.windows_committed, "prefix_tokens": self.prefix,
                          "window_tokens": window, "checkpoint_bytes": cache_nbytes(session.cache),
                          "preserved_committed_cache_bytes": cache_nbytes(session.cache),
                          "journal_bytes": 0, "replay_layer_steps": 0, "replay_seconds": 0.0,
                          "journal_integrity_checked": False, "route": "pending",
                          "seal_mode": session.seal_mode, "profile_enabled": session.profile,
                          "seal_calls": 0, "seal_logical_bytes": 0, "seal_transfer_calls": 0,
                          "seal_transfer_bytes": 0, "seal_peak_batch_bytes": 0}
        if session.profile:
            self._metadata["profile_seconds"] = {}
        _validate_cache(self.engine, session.cache, self.prefix)
        # Two independent copies preserve public state even when a test damages
        # the separately guarded checkpoint. Both copies remain in total timing.
        self.working = self._measure("working_clone", clone_cache, session.cache)
        self.checkpoint = self._measure("checkpoint_clone", clone_cache, session.cache)
        self.initial_token = session.next_token
        self._pending = self.initial_token
        self._checkpoint_seal = self._measure("checkpoint_initial_seal", self._seal_cache)
        self._owner = (session._owner, session.windows_committed)
        self.journals, self._journal_seals, self._outputs = [], [], []
        self._last_logits = None
        self._state = "active"

    def _measure(self, name, function, *args, **kwargs):
        if not self.session.profile:
            return function(*args, **kwargs)
        _sync(self.engine)
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            _sync(self.engine)
            timings = self._metadata["profile_seconds"]
            timings[name] = timings.get(name, 0.0) + time.perf_counter() - started

    def _seal_cache(self):
        return _cache_seal(self.checkpoint, self.session.seal_mode, self._metadata)

    def _seal_journal(self, row, index):
        return _journal_seal(row, self._owner, index, self.session.seal_mode, self._metadata)

    def _require_active(self):
        if self._state != "active" or self.session._active is not self:
            raise RuntimeError("Transaction is no longer active")

    def _reject(self, reason):
        self._state = "rejected"
        self.session.poisoned = True
        self.session._active = None
        self._metadata["route"] = "rejected"
        raise WindowRejected(reason, dict(self._metadata))

    @torch.no_grad()
    def step(self):
        self._require_active()
        if len(self._outputs) >= self.window:
            raise RuntimeError("Window is already fully generated")
        try:
            index = len(self._outputs)
            result = self._measure("speculation", self.engine.step, self._pending, self.working,
                                   position=self.prefix + index, capture_cuts=self.session.cuts)
            self._pending = result.next_token
            self._outputs.append(self._pending)
            self._last_logits = result.logits
            self.journals.append(result.journals)
            # Full replay can never consume a saved hidden activation. Retain
            # empty rows for harness/API compatibility, but do not hash them.
            seal = None if self.session.policy == "full" else self._measure(
                "journal_capture_seal", self._seal_journal, result.journals, index)
            self._journal_seals.append(seal)
            self._metadata["journal_bytes"] += journal_nbytes(result.journals)
        except Exception as exc:
            self._reject("Speculative execution failed: " + str(exc))
        return None

    def _journals_valid(self):
        self._metadata["journal_integrity_checked"] = True
        if len(self.journals) != self.window or len(self._journal_seals) != self.window:
            return False
        try:
            return all(set(row) == set(self.session.cuts)
                       and self._seal_journal(row, i) == self._journal_seals[i]
                       for i, row in enumerate(self.journals))
        except Exception:
            return False

    def _replay(self, cut, earliest):
        if cut:
            working = self.working
            working.key_cache[cut:] = list(self.checkpoint.key_cache[cut:])
            working.value_cache[cut:] = list(self.checkpoint.value_cache[cut:])
        else:
            working = _alias(self.checkpoint)
        token, outputs, starts, divergence = self.initial_token, [], [], None
        result = None
        for index, old in enumerate(self._outputs):
            start = cut if divergence is None else 0
            result = self.engine.step(token, working, position=self.prefix + index,
                start_layer=start, journal=self.journals[index][cut] if start else None,
                fault_layer=earliest if start else None)
            token = result.next_token
            outputs.append(token)
            starts.append(start)
            if divergence is None and token != old:
                divergence = index + 1
                if cut:
                    truncate_cache(working, self.prefix + index + 1)
        return working, outputs, result.logits, starts, divergence

    @torch.no_grad()
    def finish(self):
        self._require_active()
        if len(self._outputs) != self.window:
            raise RuntimeError("Window must be fully generated before finishing")
        try:
            _validate_cache(self.engine, self.checkpoint, self.prefix)
            if self._measure("checkpoint_validation_seal", self._seal_cache) != self._checkpoint_seal:
                self._reject("Checkpoint integrity failure")
            _validate_cache(self.engine, self.working, self.prefix + self.window)
            detected = tuple(self._measure("prefix_detection", detect_prefix_layers, self.working, self.checkpoint))
            cut, fallback, starts, divergence = 0, None, [], None
            working, outputs, logits = self.working, self._outputs, self._last_logits
            if detected:
                earliest = min(detected)
                cut = max([0] + [c for c in self.session.cuts if c <= earliest])
                if cut and not self._measure("journal_validation_seal", self._journals_valid):
                    cut, fallback = 0, "journal_integrity_failure"
                if not cut:
                    self.working = None
                    working = None
                # These two syncs preserve stage-2 replay-only timing. No
                # additional component syncs occur with profile=False.
                _sync(self.engine)
                replay_start = time.perf_counter()
                working, outputs, logits, starts, divergence = self._replay(cut, earliest)
                _sync(self.engine)
                self._metadata["replay_seconds"] = time.perf_counter() - replay_start
                if self.session.profile:
                    self._metadata["profile_seconds"]["replay"] = self._metadata["replay_seconds"]
                self._metadata["replay_layer_steps"] = sum(self.engine.num_layers - c for c in starts)
                self._metadata["route"] = "journal_fallback" if fallback else "sparse" if cut else "full"
            else:
                self._metadata["route"] = "clean"
            _validate_cache(self.engine, working, self.prefix + self.window)
            if not self._measure("precommit_finite", _finite, working, logits):
                self._reject("Non-finite state or logits before commitment")
            if self._measure("checkpoint_final_seal", self._seal_cache) != self._checkpoint_seal:
                self._reject("Checkpoint changed during recovery")
            committed = tuple(outputs)
            result = WindowResult(committed, detected, cut, tuple(starts), divergence,
                                  fallback, bool(detected), dict(self._metadata))
            next_transcript = self.session.committed_tokens + committed
            next_windows = self.session.windows_committed + 1
            self.session.cache = working
            self.working = working
            self.session.next_token = committed[-1]
            self.session.last_logits = logits
            self.session.committed_tokens = next_transcript
            self.session.windows_committed = next_windows
            self.session._active = None
            self._state = "committed"
            return result
        except WindowRejected:
            raise
        except Exception as exc:
            self._reject("Window validation/recovery failed: " + str(exc))
