"""Single-owner transactional windows for RECUT's bounded persistent-prefix model.

No fault location is accepted by this API. This is not a production server or a
general correctness detector. Trusted inputs: initial state, weights, execution,
host metadata/seals, and guard/recovery/commit periods. During speculation only,
direct cache faults must persist in the window's old prefix until detection.
Suffix-only, reverted, pre-checkpoint and computation-origin faults are excluded.
Journal/checkpoint seals detect post-capture mutation, not semantic correctness.
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


def _tensor_update(digest, tensor):
    layout = [list(tensor.shape), str(tensor.dtype), str(tensor.device)]
    digest.update(json.dumps(layout, separators=(",", ":")).encode())
    digest.update(tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes())


def _cache_seal(cache):
    digest = hashlib.sha256()
    digest.update(str((len(cache.key_cache), len(cache.value_cache), int(cache._seen_tokens))).encode())
    for name in ("key_cache", "value_cache"):
        digest.update(name.encode())
        for tensor in getattr(cache, name):
            _tensor_update(digest, tensor)
    return digest.hexdigest()


def _journal_seal(row, owner, index):
    digest = hashlib.sha256()
    digest.update(json.dumps([owner, index], separators=(",", ":")).encode())
    for cut, entry in sorted(row.items()):
        if type(entry) is not JournalEntry or type(cut) is not int:
            raise ValueError("Malformed journal entry")
        digest.update(json.dumps([cut, entry.position, entry.token, entry.cut, entry.engine_id]).encode())
        _tensor_update(digest, entry.hidden)
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
    # Ordinary DynamicCache appends with out-of-place torch.cat. No in-place
    # mutation or injected fault is permitted during recovery.
    result = DynamicCache()
    result.key_cache = list(checkpoint.key_cache)
    result.value_cache = list(checkpoint.value_cache)
    result._seen_tokens = int(checkpoint._seen_tokens)
    return result


class TransactionalSession:
    """Sequential in-process controller, not concurrent/network durability.

    `committed_tokens` is the public transcript. Transaction internals are only
    exposed for controlled test injection and must not be used as public output.
    The caller supplies a trusted already-prefilled state and pending input token.
    Faults between transactions or within guards/recovery are outside this model.
    """
    def __init__(self, engine: HFIncrementalEngine, cache, initial_token, *, policy="sparse", cuts=()):
        if policy not in ("sparse", "full"):
            raise ValueError("policy must be sparse or full")
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
        _validate_cache(self.engine, session.cache, self.prefix)
        # Speculative writes must not mutate the last committed public state.
        # The separately guarded recovery copy also cannot alias it: deliberate
        # checkpoint corruption is a tested refusal case. Both copies are timed.
        self.working = clone_cache(session.cache)
        self.checkpoint = clone_cache(session.cache)
        self.initial_token = session.next_token
        self._pending = self.initial_token
        self._checkpoint_seal = _cache_seal(self.checkpoint)
        self._owner = (session._owner, session.windows_committed)
        self.journals, self._journal_seals, self._outputs = [], [], []
        self._last_logits = None
        self._state = "active"
        self._metadata = {"window_index": session.windows_committed, "prefix_tokens": self.prefix,
                          "window_tokens": window, "checkpoint_bytes": cache_nbytes(self.checkpoint),
                          "preserved_committed_cache_bytes": cache_nbytes(session.cache),
                          "journal_bytes": 0, "replay_layer_steps": 0, "replay_seconds": 0.0,
                          "journal_integrity_checked": False, "route": "pending"}

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
            result = self.engine.step(self._pending, self.working,
                                      position=self.prefix + index, capture_cuts=self.session.cuts)
            self._pending = result.next_token
            self._outputs.append(self._pending)
            self._last_logits = result.logits
            self.journals.append(result.journals)
            self._journal_seals.append(_journal_seal(result.journals, self._owner, index))
            self._metadata["journal_bytes"] += journal_nbytes(result.journals)
        except Exception as exc:
            self._reject("Speculative execution failed: " + str(exc))
        # No logits or tokens returned before commitment.
        return None

    def _journals_valid(self):
        self._metadata["journal_integrity_checked"] = True
        if len(self.journals) != self.window or len(self._journal_seals) != self.window:
            return False
        try:
            return all(set(row) == set(self.session.cuts)
                       and _journal_seal(row, self._owner, i) == self._journal_seals[i]
                       for i, row in enumerate(self.journals))
        except Exception:
            return False

    def _replay(self, cut, earliest):
        if cut:
            working = self.working
            # Every layer at/above the cut is restored, including layers that
            # did not have a directly detected prefix fault.
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
            if _cache_seal(self.checkpoint) != self._checkpoint_seal:
                self._reject("Checkpoint integrity failure")
            _validate_cache(self.engine, self.working, self.prefix + self.window)
            detected = tuple(detect_prefix_layers(self.working, self.checkpoint))
            cut, fallback, starts, divergence = 0, None, [], None
            working, outputs, logits = self.working, self._outputs, self._last_logits
            if detected:
                earliest = min(detected)
                cut = max([0] + [c for c in self.session.cuts if c <= earliest])
                if cut and not self._journals_valid():
                    cut, fallback = 0, "journal_integrity_failure"
                if not cut:
                    # Full replay never reads speculative cache state. Release
                    # both references so its memory comparison is not charged
                    # for an unnecessary retained corrupted cache.
                    self.working = None
                    working = None
                _sync(self.engine)
                replay_start = time.perf_counter()
                working, outputs, logits, starts, divergence = self._replay(cut, earliest)
                _sync(self.engine)
                self._metadata["replay_seconds"] = time.perf_counter() - replay_start
                self._metadata["replay_layer_steps"] = sum(self.engine.num_layers - c for c in starts)
                self._metadata["route"] = "journal_fallback" if fallback else "sparse" if cut else "full"
            else:
                self._metadata["route"] = "clean"
            _validate_cache(self.engine, working, self.prefix + self.window)
            if not _finite(working, logits):
                self._reject("Non-finite state or logits before commitment")
            # Guard alias-based recovery's trusted snapshot before any output
            # becomes public. This guard is charged to total window latency.
            if _cache_seal(self.checkpoint) != self._checkpoint_seal:
                self._reject("Checkpoint changed during recovery")
            committed = tuple(outputs)
            result = WindowResult(committed, detected, cut, tuple(starts), divergence,
                                  fallback, bool(detected), dict(self._metadata))
            # Prepare allocating operations before changing any public field.
            next_transcript = self.session.committed_tokens + committed
            next_windows = self.session.windows_committed + 1
            # Atomic at the controller's single-owner method boundary; not a
            # crash-durable/network transaction or a thread-safety guarantee.
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
