"""vLLM-specific single-experiment runner.

Same shape as ``src/benchmark/runner.py`` but uses ``VLLMBackend``. Two notes
about how the vLLM result rows differ semantically:

  - ``peak_gpu_memory_gb`` reports the size of vLLM's pre-allocated KV cache
    pool, not per-cell working memory. Don't compare it to HF's peak memory
    directly — see CLAUDE.md Phase 4 section.
  - vLLM rarely OOMs under loads that crash HF; instead it serializes requests
    when the KV pool fills, so ``success=True`` even when ``total_latency``
    is much larger than HF's failed-cell baseline. The analysis notebook
    flags this as "soft degradation."
"""

import torch

from src.benchmark.memory import reset_peak_memory, peak_memory_gb
from src.benchmark.metrics import estimate_kv_cache_gb, build_result_row
from src.benchmark.runner import _is_oom
from src.benchmark.prompts import build_synthetic_prompt


def run_single_vllm_experiment(
    *, backend, context_length, batch_size, max_new_tokens, hardware_label,
    is_native_context=None,
):
    """Run one (context, batch) point on the vLLM backend."""

    kv_cache_gb = estimate_kv_cache_gb(
        backend.model_config, context_length, batch_size, backend.dtype_str
    )

    input_ids, actual_len = build_synthetic_prompt(backend.tokenizer, context_length)
    if batch_size > 1:
        input_ids = input_ids.repeat(batch_size, 1)

    reset_peak_memory("cuda")

    success = True
    error = None
    ttft = None
    total_latency = None
    output_tokens = 0
    try:
        result = backend.generate(input_ids, max_new_tokens)
        ttft = result.ttft_seconds
        total_latency = result.total_latency_seconds
        output_tokens = result.output_token_count
    except Exception as e:  # noqa: BLE001 — we want to label OOM separately from other errors
        if _is_oom(e):
            success = False
            error = f"CUDA OOM: {e}"
        else:
            success = False
            error = repr(e)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    peak_gb = peak_memory_gb("cuda")
    if success and total_latency is not None and output_tokens > 0:
        tpot = total_latency / output_tokens
        tps = output_tokens / total_latency
    else:
        tpot = None
        tps = None

    return build_result_row(
        model_id=backend.model_id,
        backend=backend.name,
        hardware=hardware_label,
        context_length=actual_len,
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
        prompt_format="synthetic_repeat",
        is_native_context=is_native_context,
    )
