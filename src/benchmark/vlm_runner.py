"""VLM-specific single-experiment runner.

Parallels src/benchmark/runner.py but accepts an image-and-text prompt from
src/benchmark/vlm_prompts and a VLMBackend whose generate() takes a processor
batch dict, not a plain input_ids tensor.
"""

import torch

from src.benchmark.memory import reset_peak_memory, peak_memory_gb
from src.benchmark.metrics import estimate_kv_cache_gb, build_result_row
from src.benchmark.runner import _is_oom
from src.benchmark.vlm_prompts import build_vlm_prompt


def run_single_vlm_experiment(
    *, backend, context_length, batch_size, max_new_tokens, hardware_label,
    is_native_context=None,
):
    """Run one VLM (image + text) point and return a result-row dict."""

    kv_cache_gb = estimate_kv_cache_gb(
        backend.model_config, context_length, batch_size, backend.dtype
    )

    inputs, actual_total, image_tokens, text_tokens = build_vlm_prompt(
        backend.processor, context_length
    )
    if batch_size > 1:
        # Tile the per-tensor batch dim. Most processor outputs use dim 0 for
        # batch; left as a pass-through for now since Phase 1 is single request.
        raise NotImplementedError("VLM batch_size>1 not implemented yet.")

    reset_peak_memory(backend.device)

    success = True
    error = None
    ttft = None
    total_latency = None
    output_tokens = 0
    try:
        result = backend.generate(inputs, max_new_tokens)
        ttft = result.ttft_seconds
        total_latency = result.total_latency_seconds
        output_tokens = result.output_token_count
    except Exception as e:  # noqa: BLE001
        if _is_oom(e):
            success = False
            error = f"CUDA OOM: {e}"
        else:
            success = False
            error = repr(e)
        if backend.device.startswith("cuda"):
            torch.cuda.empty_cache()

    peak_gb = peak_memory_gb(backend.device)
    if success and total_latency is not None and output_tokens > 0:
        tpot = total_latency / output_tokens
        tps = output_tokens / total_latency
    else:
        tpot = None
        tps = None

    if success and backend.device.startswith("cuda"):
        torch.cuda.empty_cache()

    return build_result_row(
        model_id=backend.model_id,
        backend=backend.name,
        hardware=hardware_label,
        context_length=actual_total,
        batch_size=batch_size,
        max_new_tokens=max_new_tokens,
        ttft=ttft,
        tpot=tpot,
        total_latency=total_latency,
        tokens_per_second=tps,
        peak_gpu_gb=peak_gb,
        kv_cache_gb=kv_cache_gb,
        success=success,
        error=error,
        prompt_format="vlm_image+text",
        is_native_context=is_native_context,
        image_token_count=image_tokens,
        text_token_count=text_tokens,
    )
