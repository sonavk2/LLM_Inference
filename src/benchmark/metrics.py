"""KV-cache estimation and result-row construction."""

import torch


_DTYPE_BYTES = {
    "float32": 4, "float16": 2, "bfloat16": 2,
    torch.float32: 4, torch.float16: 2, torch.bfloat16: 2,
}


def estimate_kv_cache_gb(model_config, context_length, batch_size, dtype):
    """
    Estimate KV-cache memory in GB.

    Formula: 2 * num_layers * batch * context * num_kv_heads * head_dim * bytes
    The factor of 2 accounts for keys and values.

    Falls back to num_attention_heads when num_key_value_heads is absent,
    so the estimate works for both standard MHA and GQA/MQA models.
    """
    n_layers = model_config.num_hidden_layers
    n_attn_heads = model_config.num_attention_heads
    n_kv_heads = getattr(model_config, "num_key_value_heads", n_attn_heads)
    head_dim = model_config.hidden_size // n_attn_heads
    bytes_per_elem = _DTYPE_BYTES.get(dtype, 4)
    total_bytes = (
        2 * n_layers * batch_size * context_length * n_kv_heads * head_dim * bytes_per_elem
    )
    return total_bytes / 1e9


def build_result_row(
    *,
    model_id, backend, hardware, context_length, batch_size, max_new_tokens,
    ttft, tpot, total_latency, tokens_per_second, peak_gpu_gb, kv_cache_gb,
    success, error,
    # Optional metadata. Older callers can omit; rows from new callers carry the
    # extra context that distinguishes YaRN-extrapolated cells and VLM rows.
    prompt_format=None,
    is_native_context=None,
    image_token_count=None,
    text_token_count=None,
    quantization=None,
):
    """Assemble a single result row matching the JSONL schema in CLAUDE.md."""
    return {
        "model_name": model_id,
        "backend": backend,
        "hardware": hardware,
        "context_length": context_length,
        "batch_size": batch_size,
        "max_new_tokens": max_new_tokens,
        "ttft_seconds": ttft,
        "tpot_seconds": tpot,
        "total_latency_seconds": total_latency,
        "tokens_per_second": tokens_per_second,
        "peak_gpu_memory_gb": peak_gpu_gb,
        "kv_cache_memory_gb": kv_cache_gb,
        "success": success,
        "error": error,
        "prompt_format": prompt_format,
        "is_native_context": is_native_context,
        "image_token_count": image_token_count,
        "text_token_count": text_token_count,
        "quantization": quantization,
    }
