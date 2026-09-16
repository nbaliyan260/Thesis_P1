"""Bounded exact-prefix comparison feasibility, not a general online detector.

Only persistent post-checkpoint changes to the old ordinary KV prefix are
observable. Suffix-only faults, reverted/transient errors, pre-checkpoint faults,
and a corrupt trusted checkpoint are outside this detector's guarantee.
"""
import time
import torch
from transformers.cache_utils import DynamicCache
from recut_engine import cache_nbytes
from run_recut import sync


@torch.no_grad()
def detect_prefix_layers(working, checkpoint):
    if type(working) is not DynamicCache or type(checkpoint) is not DynamicCache:
        raise TypeError("Only ordinary DynamicCache is supported")
    if not len(checkpoint) or len(working) != len(checkpoint):
        raise ValueError("Nonempty matching layers required")
    length = int(checkpoint.get_seq_length())
    flags = []
    for layer in range(len(checkpoint)):
        if checkpoint.get_seq_length(layer) != length or working.get_seq_length(layer) < length:
            raise ValueError("Inconsistent prefix lengths")
        old_k, old_v = checkpoint.key_cache[layer], checkpoint.value_cache[layer]
        new_k, new_v = working.key_cache[layer][..., :length, :], working.value_cache[layer][..., :length, :]
        if any(a.shape != b.shape or a.dtype != b.dtype or a.device != b.device
               for a, b in ((new_k, old_k), (new_v, old_v))):
            raise ValueError("Prefix layout mismatch")
        flags.append(torch.any(new_k != old_k) | torch.any(new_v != old_v))
    # One consolidated device-to-host transfer for layer-level decisions.
    differs = torch.stack(flags).cpu().tolist()
    return [layer for layer, changed in enumerate(differs) if changed]


@torch.no_grad()
def measure_prefix_detection(engine, working, checkpoint, repetitions=5):
    rows, observed = [], None
    for rep in range(-1, repetitions):
        sync(engine)
        start = time.perf_counter()
        layers = detect_prefix_layers(working, checkpoint)
        sync(engine)
        elapsed = time.perf_counter() - start
        if observed is not None and layers != observed:
            raise RuntimeError("Detection changed across repeated reads")
        observed = layers
        if rep >= 0:
            rows.append(dict(repetition=rep, seconds=elapsed, detected_layers=layers))
    return dict(detected_layers=observed,
        route="unique_layer" if len(observed) == 1 else "no_alarm_full" if not observed else "reject_multiple",
        timings=rows, checkpoint_payload_bytes=cache_nbytes(checkpoint),
        logical_bytes_read=2 * cache_nbytes(checkpoint),
        prefix_tokens=int(checkpoint.get_seq_length()),
        semantics="exact elementwise old-prefix comparison; suffix intentionally ignored")
