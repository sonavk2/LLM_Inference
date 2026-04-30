"""vLLM backend for Phase 4 (serving-system comparison).

Same small contract as ``HFBackend``:
  - ``.load()`` constructs the vLLM engine and pulls weights / tokenizer
  - ``.generate(input_ids, max_new_tokens) -> GenerationResult``
  - ``.tokenizer`` and ``.model_config`` exposed for the harness

vLLM differences worth knowing:
  - Continuous batching: ``llm.generate(prompts)`` takes a *list* and schedules
    them under PagedAttention. There's no "static batch slab" concept.
  - KV cache is pre-allocated as a pool sized by ``gpu_memory_utilization``
    (default 0.9). So ``torch.cuda.max_memory_allocated()`` reports the pool
    size, not per-cell usage. Don't read it as an HF-equivalent peak.
  - TTFT is read from ``RequestOutput.metrics.first_token_time`` rather than a
    streamer hook — vLLM tracks per-request timing natively.
  - Greedy decode uses ``temperature=0`` and ``ignore_eos=True`` so we always
    produce ``max_new_tokens``, matching HF's ``do_sample=False`` behavior for
    a fair latency/throughput comparison.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch


def _normalize_rope(rope_in):
    """vLLM accepts the new ``rope_type`` schema; collapse legacy ``type`` key."""
    rope = dict(rope_in)
    if "type" in rope and "rope_type" not in rope:
        rope["rope_type"] = rope["type"]
    return rope


@dataclass
class GenerationResult:
    output_token_count: int               # total new tokens, summed across batch
    ttft_seconds: Optional[float]         # earliest first_token_time across the batch
    total_latency_seconds: float


class VLLMBackend:
    name = "vllm"

    def __init__(
        self, model_id, dtype, device, trust_remote_code=False,
        rope_scaling: Optional[dict] = None,
        max_model_len: Optional[int] = None,
        gpu_memory_utilization: float = 0.9,
    ):
        self.model_id = model_id
        # vLLM accepts the dtype as a string identifier.
        self.dtype_str = dtype
        self.dtype = dtype
        self.device = device  # vLLM is CUDA-only; field kept for harness compat.
        self.trust_remote_code = trust_remote_code
        self.rope_scaling = rope_scaling
        self.max_model_len = max_model_len
        self.gpu_memory_utilization = gpu_memory_utilization
        self.tokenizer = None
        self.llm = None
        self.model_config = None

    def load(self):
        from vllm import LLM
        kwargs = {
            "model": self.model_id,
            "dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            # Pin off so a future vLLM default change doesn't silently flip the
            # experiment — prefix caching would skew throughput comparisons.
            "enable_prefix_caching": False,
            # Quieter logs in notebook output.
            "disable_log_stats": False,  # we NEED this off to populate RequestOutput.metrics
        }
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        if self.rope_scaling:
            kwargs["rope_scaling"] = _normalize_rope(self.rope_scaling)

        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        # The HF config of the underlying model — lets the KV-cache estimator
        # read num_layers / num_kv_heads / hidden_size the same way as HFBackend.
        self.model_config = self.llm.llm_engine.model_config.hf_config

    def generate(self, input_ids, max_new_tokens):
        """Run greedy generation. ``input_ids`` shape: (batch_size, seq_len)."""
        from vllm import SamplingParams
        assert self.llm is not None, "Call load() first"

        # vLLM accepts pre-tokenized prompts; using token IDs avoids re-tokenizing
        # and keeps context_length exact.
        prompts = [{"prompt_token_ids": ids.tolist()} for ids in input_ids]
        sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=max_new_tokens,
            ignore_eos=True,  # always produce max_new_tokens, matches HF do_sample=False
        )

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = self.llm.generate(prompts, sampling_params, use_tqdm=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_latency = time.perf_counter() - start

        # TTFT: earliest first-token time across the batch (analogous to HF's
        # streamer, which fires on the first decode step regardless of which
        # sequence "owns" it).
        ttft_seconds = None
        per_request_ttfts = []
        for out in outputs:
            m = out.metrics
            if m is not None and m.first_token_time is not None and m.arrival_time is not None:
                per_request_ttfts.append(m.first_token_time - m.arrival_time)
        if per_request_ttfts:
            ttft_seconds = min(per_request_ttfts)

        total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
        return GenerationResult(
            output_token_count=total_tokens,
            ttft_seconds=ttft_seconds,
            total_latency_seconds=total_latency,
        )
