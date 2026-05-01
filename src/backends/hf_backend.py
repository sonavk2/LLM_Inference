"""HF backend used by the benchmark harness.

Design choices:
- Use a streamer hook to measure TTFT directly.
- Apply RoPE edits on config before model weights load.
- Let bitsandbytes place weights when quantization is enabled.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.generation.streamers import BaseStreamer


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def _normalize_rope(rope_in, existing):
    """Merge RoPE keys so old/new transformers configs both work."""
    rope = dict(rope_in)
    if "type" in rope and "rope_type" not in rope:
        rope["rope_type"] = rope["type"]
    if "rope_type" in rope and "type" not in rope:
        rope["type"] = rope["rope_type"]
    base = dict(existing) if existing else {}
    base.update(rope)
    return base


@dataclass
class GenerationResult:
    output_token_count: int               # new tokens across batch
    ttft_seconds: Optional[float]         # wall-clock TTFT
    total_latency_seconds: float


class _FirstTokenTimer(BaseStreamer):
    """Record time of first generated token (skip prompt callback)."""

    def __init__(self):
        self.start: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self._saw_prompt = False

    def put(self, value):
        del value  # signature compatibility with BaseStreamer; we only need the call
        if not self._saw_prompt:
            self._saw_prompt = True
            return
        if self.first_token_time is None and self.start is not None:
            self.first_token_time = time.perf_counter() - self.start

    def end(self):
        pass


class HFBackend:
    name = "huggingface"

    def __init__(
        self, model_id, dtype, device, trust_remote_code=False,
        rope_scaling: Optional[dict] = None,
        quantization: Optional[dict] = None,
    ):
        self.model_id = model_id
        self.dtype_str = dtype
        self.dtype = _DTYPE_MAP[dtype]
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.rope_scaling = rope_scaling
        self.quantization = quantization
        self.tokenizer = None
        self.model = None
        self.model_config = None

    def _build_quantization_config(self):
        # Keep this import path so pyright resolves the symbol.
        from transformers.utils.quantization_config import BitsAndBytesConfig
        q = dict(self.quantization or {})
        # YAML stores dtype as string; bitsandbytes expects torch.dtype.
        compute = q.get("bnb_4bit_compute_dtype")
        if isinstance(compute, str):
            q["bnb_4bit_compute_dtype"] = _DTYPE_MAP[compute]
        return BitsAndBytesConfig(**q)

    def load(self):
        """Load tokenizer and model onto the target device."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        # Most causal LMs do not define pad_token; EOS is safe for greedy decode.
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Edit RoPE on config first so model init picks it up.
        config = AutoConfig.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        if self.rope_scaling:
            existing = (
                getattr(config, "rope_parameters", None)
                or getattr(config, "rope_scaling", None)
            )
            merged = _normalize_rope(self.rope_scaling, existing)
            config.rope_scaling = merged
            config.rope_parameters = merged

        load_kwargs = {
            "config": config,
            "torch_dtype": self.dtype,
            "trust_remote_code": self.trust_remote_code,
        }
        if self.quantization:
            load_kwargs["quantization_config"] = self._build_quantization_config()
            # bitsandbytes handles placement; explicit .to(device) can conflict.
            load_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **load_kwargs)
        if not self.quantization:
            self.model.to(self.device)
        self.model.eval()
        self.model_config = self.model.config

    @torch.inference_mode()
    def generate(self, input_ids, max_new_tokens):
        """Run greedy generation. input_ids shape: (batch_size, seq_len)."""
        assert self.model is not None and self.tokenizer is not None, "Call load() first"
        input_ids = input_ids.to(self.device)
        # Prompts in each batch are same length, so mask is all ones.
        # Passing it explicitly avoids pad/eos ambiguity warnings in HF.
        attention_mask = torch.ones_like(input_ids)
        timer = _FirstTokenTimer()

        # Sync before/after timing so latency includes GPU work, not queued kernels.
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        timer.start = time.perf_counter()

        output = self.model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id,
            streamer=timer,
        )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        total_latency = time.perf_counter() - timer.start

        new_tokens_per_seq = output.shape[1] - input_ids.shape[1]
        output_token_count = int(new_tokens_per_seq * input_ids.shape[0])
        return GenerationResult(
            output_token_count=output_token_count,
            ttft_seconds=timer.first_token_time,
            total_latency_seconds=total_latency,
        )
