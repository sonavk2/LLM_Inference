"""vLLM backend for Phase 4 comparison against HF.

Design choices:
- Keep decode greedy (`temperature=0`) for apples-to-apples with HF.
- Measure TTFT with a dedicated 1-token pass.
- Expose HF config from vLLM so KV-cache math stays shared.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch


def _normalize_rope(rope_in):
    """Normalize RoPE keys to the schema vLLM expects."""
    rope = dict(rope_in)
    if "type" in rope and "rope_type" not in rope:
        rope["rope_type"] = rope["type"]
    return rope


@dataclass
class GenerationResult:
    output_token_count: int               # new tokens across batch
    ttft_seconds: Optional[float]         # wall-clock TTFT
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
        # vLLM takes dtype as a string name.
        self.dtype_str = dtype
        self.dtype = dtype
        self.device = device  # kept for harness compatibility
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
            # Keep off so benchmark behavior does not change with vLLM defaults.
            "enable_prefix_caching": False,
            # Keep metrics enabled; some analysis depends on request stats.
            "disable_log_stats": False,
        }
        if self.max_model_len is not None:
            kwargs["max_model_len"] = self.max_model_len
        if self.rope_scaling:
            kwargs["rope_scaling"] = _normalize_rope(self.rope_scaling)

        self.llm = LLM(**kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        # Reuse shared KV estimator by exposing the model's HF config.
        self.model_config = self.llm.llm_engine.model_config.hf_config

    def generate(self, input_ids, max_new_tokens):
        """Run greedy generation on pre-tokenized prompts."""
        from vllm import SamplingParams
        assert self.llm is not None, "Call load() first"

        # Pass token IDs directly so context length is exact.
        prompts = [{"prompt_token_ids": ids.tolist()} for ids in input_ids]

        # Stage 1: 1-token run as TTFT proxy (prefill + first decode).
        sp_ttft = SamplingParams(temperature=0.0, max_tokens=1, ignore_eos=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft_start = time.perf_counter()
        _ = self.llm.generate(prompts, sp_ttft, use_tqdm=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        ttft_seconds = time.perf_counter() - ttft_start

        # Stage 2: full run for throughput + total latency.
        sp_full = SamplingParams(
            temperature=0.0, max_tokens=max_new_tokens, ignore_eos=True,
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        start = time.perf_counter()
        outputs = self.llm.generate(prompts, sp_full, use_tqdm=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        total_latency = time.perf_counter() - start

        total_tokens = sum(len(out.outputs[0].token_ids) for out in outputs)
        return GenerationResult(
            output_token_count=total_tokens,
            ttft_seconds=ttft_seconds,
            total_latency_seconds=total_latency,
        )
