"""Bounded HF decoder stepping for the RECUT diagnostic pilot.

Supported contract: transformers 4.51.3, eager Llama/Qwen2 attention, one
unpadded sequence, one input token per call, eval mode, default RoPE, fixed
weights, and a trusted ordinary DynamicCache. This is not a fault detector.
All numerical model operations are the installed Hugging Face modules.

Partial replay may retain future entries in *unexecuted* lower layers. The
explicit attention-mask length is consequently derived from ``position``,
never layer zero's possibly longer cache. Executed layers must be append-ready.
Journal metadata checks do not prove that the entire token history is clean:
the caller must stop journal reuse irreversibly at the first output divergence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

import torch
import transformers
from transformers import LlamaForCausalLM, Qwen2ForCausalLM
from transformers.cache_utils import DynamicCache


PINNED_TRANSFORMERS = "4.51.3"


@dataclass(frozen=True)
class JournalEntry:
    position: int
    token: int
    cut: int
    hidden: torch.Tensor
    engine_id: int


@dataclass(frozen=True)
class StepResult:
    logits: torch.Tensor
    journals: dict[int, JournalEntry]
    layers_executed: int

    @property
    def next_token(self) -> int:
        """Greedy decision; reading the result synchronizes a CUDA tensor."""
        return int(self.logits[0, -1].argmax().item())


def _require_cache(cache: DynamicCache) -> None:
    # Quantized/offloaded/sliding caches need different copy/truncation rules.
    if type(cache) is not DynamicCache:
        raise TypeError("Only ordinary transformers.DynamicCache is supported")
    if len(cache.key_cache) != len(cache.value_cache):
        raise ValueError("Mismatched K/V layer counts")


def _sync_seen_tokens(cache: DynamicCache) -> None:
    # In 4.51.3 this compatibility counter is updated only by layer zero.
    cache._seen_tokens = int(cache.get_seq_length(0))


@torch.no_grad()
def clone_cache(cache: DynamicCache) -> DynamicCache:
    """Deep independent tensor copy, including temporarily unequal lengths."""
    _require_cache(cache)
    result = DynamicCache()
    result.key_cache = [t.detach().clone() for t in cache.key_cache]
    result.value_cache = [t.detach().clone() for t in cache.value_cache]
    _sync_seen_tokens(result)
    return result


@torch.no_grad()
def truncate_cache(cache: DynamicCache, length: int, start_layer: int = 0) -> None:
    """In-place truncate selected layers, never extend; copies release suffix storage.

    On a changed token history call with start_layer=0 before the next step.
    Keeping a sliced view would falsely undercount the retained allocation.
    """
    _require_cache(cache)
    if not isinstance(length, int) or length < 0:
        raise ValueError("length must be a nonnegative integer")
    if not isinstance(start_layer, int) or not 0 <= start_layer <= len(cache):
        raise ValueError("start_layer is outside the cache")
    for i in range(start_layer, len(cache)):
        if cache.get_seq_length(i) < length:
            raise ValueError(f"Cannot extend layer {i} to length {length}")
    for collection in (cache.key_cache, cache.value_cache):
        for i in range(start_layer, len(collection)):
            t = collection[i]
            if t.ndim == 4:
                collection[i] = t[..., :length, :].clone()
            elif t.numel():
                raise ValueError("Malformed cache tensor")
    _sync_seen_tokens(cache)


@torch.no_grad()
def restore_layers(
    cache: DynamicCache, checkpoint: DynamicCache, start_layer: int = 0
) -> None:
    """Replace selected complete layer histories with independent checkpoint copies.

    The checkpoint is assumed trusted and is not mutated. Restoring upper
    layers does not validate lower layers or journal provenance.
    """
    _require_cache(cache)
    _require_cache(checkpoint)
    if len(cache) != len(checkpoint):
        raise ValueError("Cache and checkpoint must have identical layer counts")
    if not isinstance(start_layer, int) or not 0 <= start_layer <= len(cache):
        raise ValueError("start_layer is outside the cache")
    for destination, source in (
        (cache.key_cache, checkpoint.key_cache),
        (cache.value_cache, checkpoint.value_cache),
    ):
        for i in range(start_layer, len(source)):
            destination[i] = source[i].detach().clone()
    _sync_seen_tokens(cache)


def cache_nbytes(cache: DynamicCache) -> int:
    """Logical K/V payload bytes; allocator and Python metadata are excluded."""
    _require_cache(cache)
    return sum(t.numel() * t.element_size() for t in cache.key_cache + cache.value_cache)


def journal_nbytes(entries: Mapping | Iterable) -> int:
    """Payload bytes for an entry iterable/dict, or nested per-step dictionaries.

    Each journal has its own cloned tensor. Python metadata/allocator overhead
    is excluded and must not be presented as included by an experiment report.
    """
    values = entries.values() if isinstance(entries, Mapping) else entries
    total = 0
    for entry in values:
        if isinstance(entry, JournalEntry):
            total += entry.hidden.numel() * entry.hidden.element_size()
        elif isinstance(entry, Mapping):
            total += journal_nbytes(entry)
        else:
            raise TypeError("Expected JournalEntry objects or nested dictionaries")
    return total


@torch.no_grad()
def compare_caches(a: DynamicCache, b: DynamicCache) -> dict:
    """Exact tensor/state comparison; non-finite content is always a failure.

    Diagnostics synchronize devices and belong outside timed recovery regions.
    """
    _require_cache(a)
    _require_cache(b)
    mismatches = []
    max_error = 0.0
    finite = True
    if len(a) != len(b):
        mismatches.append("layer_count")
    for kind in ("key_cache", "value_cache"):
        left, right = getattr(a, kind), getattr(b, kind)
        for i, (x, y) in enumerate(zip(left, right)):
            label = f"{kind}:{i}"
            if x.shape != y.shape or x.dtype != y.dtype or x.device != y.device:
                mismatches.append(label + ":layout")
                max_error = float("inf")
                continue
            if not (bool(torch.isfinite(x).all()) and bool(torch.isfinite(y).all())):
                finite = False
                mismatches.append(label + ":nonfinite")
                max_error = float("inf")
            elif not torch.equal(x, y):
                mismatches.append(label)
                if x.numel():
                    error = float((x.double() - y.double()).abs().max().item())
                    max_error = max(max_error, error)
    if int(a._seen_tokens) != int(b._seen_tokens):
        mismatches.append("seen_tokens")
    equal = not mismatches and finite
    return {"equal": equal, "eq": equal, "max_abs_error": max_error,
            "finite": finite, "mismatches": mismatches}


class HFIncrementalEngine:
    """Expose HF decoder-layer boundaries without replacing model arithmetic."""

    def __init__(self, model: LlamaForCausalLM | Qwen2ForCausalLM):
        if transformers.__version__ != PINNED_TRANSFORMERS:
            raise RuntimeError(f"Require transformers=={PINNED_TRANSFORMERS}")
        if type(model) not in (LlamaForCausalLM, Qwen2ForCausalLM):
            raise TypeError("Only native LlamaForCausalLM and Qwen2ForCausalLM are supported")
        if model.training:
            raise ValueError("Call model.eval() before constructing the engine")
        config = model.config
        if config._attn_implementation != "eager":
            raise ValueError("Load the model with attn_implementation='eager'")
        if getattr(config, "use_sliding_window", False):
            raise ValueError("Sliding-window attention is outside the pilot contract")
        if getattr(config, "rope_scaling", None):
            raise ValueError("Only default RoPE is supported by this pilot")
        if getattr(model, "is_quantized", False):
            raise ValueError("Quantized model wrappers are unsupported")
        devices = {p.device for p in model.parameters()}
        dtypes = {p.dtype for p in model.parameters()}
        if len(devices) != 1 or next(iter(devices)).type == "meta":
            raise ValueError("Model must reside on one real device")
        if len(dtypes) != 1 or next(iter(dtypes)) not in (
            torch.float32, torch.float16, torch.bfloat16
        ):
            raise ValueError("Model requires one supported floating-point parameter dtype")
        self.model = model
        self.decoder = model.model
        self.num_layers = len(self.decoder.layers)
        self.device = next(iter(devices))
        self.dtype = next(iter(dtypes))
        self._engine_id = id(self)

    @staticmethod
    def empty_cache() -> DynamicCache:
        return DynamicCache()

    @torch.no_grad()
    def step(
        self,
        token: int,
        cache: DynamicCache,
        *,
        position: int | None = None,
        start_layer: int = 0,
        journal: JournalEntry | None = None,
        fault_layer: int | None = None,
        capture_cuts: Iterable[int] = (),
    ) -> StepResult:
        """Append one token to every executed layer; mutate ``cache`` in place.

        For partial replay, the caller attests the entire input history still
        agrees with the speculative history and localization is trustworthy.
        A mismatched/missing entry raises; caller may retry a full-layer replay
        only after preparing all layer lengths appropriately. No implicit
        truncation, token substitution or repair is performed here.
        """
        _require_cache(cache)
        if self.model.training:
            raise ValueError("The model must remain in evaluation mode")
        if not isinstance(token, int) or not 0 <= token < self.model.config.vocab_size:
            raise ValueError("token must be an in-vocabulary Python integer")
        if not isinstance(start_layer, int) or not 0 <= start_layer < self.num_layers:
            raise ValueError("Invalid start_layer")
        if len(cache) > self.num_layers:
            raise ValueError("Cache has too many layers")
        if position is None:
            position = int(cache.get_seq_length(start_layer))
        if not isinstance(position, int) or position < 0:
            raise ValueError("position must be a nonnegative integer")
        if position >= self.model.config.max_position_embeddings:
            raise ValueError("Position exceeds the supported model context")
        cuts = set(capture_cuts)
        if any(not isinstance(c, int) or not start_layer <= c < self.num_layers for c in cuts):
            raise ValueError("Capture cuts must belong to executed layers")
        for i in range(start_layer, self.num_layers):
            if int(cache.get_seq_length(i)) != position:
                raise ValueError(f"Layer {i} is not append-ready at position {position}")
            if i < len(cache) and cache.key_cache[i].numel():
                for t in (cache.key_cache[i], cache.value_cache[i]):
                    if t.ndim != 4 or t.shape[0] != 1 or t.device != self.device or t.dtype != self.dtype:
                        raise ValueError("Cache shape/device/dtype violates engine contract")
        if start_layer:
            if fault_layer is None or not start_layer <= fault_layer < self.num_layers:
                raise ValueError("Partial replay requires a cut at or below the localized fault")
            if journal is None:
                raise ValueError("Partial replay requires a journal entry")
            if (journal.engine_id, journal.cut, journal.position, journal.token) != (
                self._engine_id, start_layer, position, token
            ):
                raise ValueError("Journal provenance does not match this step")
            hidden = journal.hidden
            if hidden.shape != (1, 1, self.model.config.hidden_size) or hidden.device != self.device or hidden.dtype != self.dtype:
                raise ValueError("Journal tensor layout does not match the model")
        else:
            if journal is not None:
                raise ValueError("Layer-zero replay uses token embeddings, not a journal")
            ids = torch.tensor([[token]], dtype=torch.long, device=self.device)
            hidden = self.decoder.embed_tokens(ids)

        cache_position = torch.tensor([position], dtype=torch.long, device=self.device)
        position_ids = cache_position.unsqueeze(0)
        # Crucial: lower layers may still contain future speculative entries.
        # An explicit mask length keeps HF's mask factory independent of those.
        attention_mask = torch.ones((1, position + 1), dtype=torch.long, device=self.device)
        causal_mask = self.decoder._update_causal_mask(
            attention_mask, hidden, cache_position, cache, False
        )
        position_embeddings = self.decoder.rotary_emb(hidden, position_ids)
        captured = {}
        for i in range(start_layer, self.num_layers):
            if i in cuts:
                captured[i] = JournalEntry(
                    position, token, i, hidden.detach().clone(), self._engine_id
                )
            hidden = self.decoder.layers[i](
                hidden,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=cache,
                output_attentions=False,
                use_cache=True,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )[0]
        logits = self.model.lm_head(self.decoder.norm(hidden))
        _sync_seen_tokens(cache)
        return StepResult(logits, captured, self.num_layers - start_layer)

    @torch.no_grad()
    def prefill(self, tokens: Sequence[int] | torch.Tensor) -> tuple[DynamicCache, torch.Tensor]:
        """Build a prefix using the same single-token shapes as every replay."""
        if isinstance(tokens, torch.Tensor):
            if tokens.ndim == 2 and tokens.shape[0] == 1:
                tokens = tokens[0]
            if tokens.ndim != 1:
                raise ValueError("Prefill accepts one unpadded token sequence")
            tokens = tokens.tolist()
        if len(tokens) == 0:
            raise ValueError("The verified prefix cannot be empty")
        cache = self.empty_cache()
        result = None
        for token in tokens:
            result = self.step(token, cache)
        return cache, result.logits
