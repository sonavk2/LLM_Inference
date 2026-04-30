"""Vision-language model backend (image + text inputs).

Same small contract as HFBackend but the input is a multimodal batch from a
processor, not a plain input_ids tensor. Used by scripts/run_vlm_context_sweep.py.

Phase 1: greedy generation, single request. TTFT comes from a streamer hook
just like the text backend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

import torch
from transformers import AutoConfig, AutoProcessor
from transformers.generation.streamers import BaseStreamer

# AutoModelForImageTextToText is the umbrella class for newer VLMs (Qwen2-VL,
# Llava-Next, etc.). On older transformers it may not exist; fall back to
# AutoModelForVision2Seq, which covers the same models under the previous
# umbrella name.
try:
    from transformers import AutoModelForImageTextToText as _AutoVLM
except ImportError:  # transformers < 4.45
    from transformers import AutoModelForVision2Seq as _AutoVLM


_DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


@dataclass
class GenerationResult:
    output_token_count: int
    ttft_seconds: Optional[float]
    total_latency_seconds: float


class _FirstTokenTimer(BaseStreamer):
    """Records wall-clock time of the first generated token (skips prompt put)."""

    def __init__(self):
        self.start: Optional[float] = None
        self.first_token_time: Optional[float] = None
        self._saw_prompt = False

    def put(self, value):
        del value
        if not self._saw_prompt:
            self._saw_prompt = True
            return
        if self.first_token_time is None and self.start is not None:
            self.first_token_time = time.perf_counter() - self.start

    def end(self):
        pass


class VLMBackend:
    name = "huggingface_vlm"

    def __init__(
        self, model_id, dtype, device, trust_remote_code=False,
        rope_scaling: Optional[dict] = None,
    ):
        self.model_id = model_id
        self.dtype_str = dtype
        self.dtype = _DTYPE_MAP[dtype]
        self.device = device
        self.trust_remote_code = trust_remote_code
        self.rope_scaling = rope_scaling
        self.processor = None
        self.tokenizer = None  # alias for runner compatibility
        self.model = None
        self.model_config = None

    def load(self):
        self.processor = AutoProcessor.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        self.tokenizer = self.processor.tokenizer

        config = AutoConfig.from_pretrained(
            self.model_id, trust_remote_code=self.trust_remote_code
        )
        if self.rope_scaling:
            # Qwen2-VL nests its text-stack config under .text_config; apply
            # rope_scaling there if it exists, else on the top-level config.
            target = getattr(config, "text_config", None) or config
            target.rope_scaling = dict(self.rope_scaling)

        self.model = _AutoVLM.from_pretrained(
            self.model_id,
            config=config,
            torch_dtype=self.dtype,
            trust_remote_code=self.trust_remote_code,
        )
        self.model.to(self.device)
        self.model.eval()

        # KV-cache estimator reads num_hidden_layers / num_key_value_heads /
        # hidden_size / num_attention_heads from this. Qwen2-VL puts those on
        # text_config; expose that so estimate_kv_cache_gb sees the right
        # numbers without special-casing the VLM in the harness.
        self.model_config = getattr(self.model.config, "text_config", self.model.config)

    @torch.inference_mode()
    def generate(self, inputs, max_new_tokens):
        """Run greedy generation. `inputs` is a BatchFeature from the processor."""
        assert self.model is not None and self.processor is not None, "Call load() first"
        assert self.tokenizer is not None, "Tokenizer not initialized; call load() first"
        # Move every tensor in the batch onto the device.
        inputs = {
            k: (v.to(self.device) if isinstance(v, torch.Tensor) else v)
            for k, v in inputs.items()
        }
        timer = _FirstTokenTimer()

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        timer.start = time.perf_counter()

        output = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.pad_token_id or self.tokenizer.eos_token_id,
            streamer=timer,
        )

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        total_latency = time.perf_counter() - timer.start

        prompt_len = int(inputs["input_ids"].shape[1])
        new_tokens = int(output.shape[1] - prompt_len)
        return GenerationResult(
            output_token_count=new_tokens * int(inputs["input_ids"].shape[0]),
            ttft_seconds=timer.first_token_time,
            total_latency_seconds=total_latency,
        )
