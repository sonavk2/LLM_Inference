"""Run a VLM context sweep with exact total-token prompts."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backends.vlm_backend import VLMBackend
from src.benchmark.config import load_experiment_config, load_yaml
from src.benchmark.memory import detect_device, device_label
from src.benchmark.result_writer import append_jsonl
from src.benchmark.vlm_runner import run_single_vlm_experiment


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to experiment YAML")
    p.add_argument(
        "--model-config", default=None,
        help="Override the model_config pointer in the experiment YAML.",
    )
    p.add_argument("--context-lengths", nargs="+", type=int, required=True)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument(
        "--model-id", default=None,
        help="Override the model_id from the model config.",
    )
    p.add_argument("--dtype", default=None)
    p.add_argument(
        "--hardware", default=None,
        help="Hardware label to record (defaults to auto-detected device name).",
    )
    p.add_argument(
        "--results-path", default=None,
        help="Override the results_path from the config.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment_config(args.config)
    if args.model_config:
        cfg["model"] = load_yaml(args.model_config)

    model_cfg = cfg["model"]
    if not model_cfg.get("is_vlm"):
        raise SystemExit(
            "Model config does not have is_vlm: true. "
            "Use scripts/run_context_sweep.py for text-only models."
        )

    model_id = args.model_id or model_cfg["model_id"]
    dtype = args.dtype or model_cfg.get("dtype", "bfloat16")
    trust = model_cfg.get("trust_remote_code", False)
    rope_scaling = model_cfg.get("rope_scaling")
    native_context = model_cfg.get("native_context")

    device = detect_device()
    hardware = args.hardware or device_label(device)
    results_path = Path(args.results_path or cfg["output"]["results_path"])

    print(f"Loading VLM {model_id} (dtype={dtype}) on {device} ...", flush=True)
    backend = VLMBackend(
        model_id=model_id, dtype=dtype, device=device, trust_remote_code=trust,
        rope_scaling=rope_scaling,
    )
    backend.load()
    print("Model loaded.", flush=True)

    for ctx in args.context_lengths:
        is_native = None if native_context is None else (ctx <= native_context)
        print(
            f"\n--- context_length={ctx}, native={is_native} ---", flush=True
        )
        row = run_single_vlm_experiment(
            backend=backend,
            context_length=ctx,
            batch_size=1,
            max_new_tokens=args.max_new_tokens,
            hardware_label=hardware,
            is_native_context=is_native,
        )
        append_jsonl(results_path, row)
        print(
            f"  success={row['success']} "
            f"ttft={row['ttft_seconds']} "
            f"latency={row['total_latency_seconds']} "
            f"peak_gpu_gb={row['peak_gpu_memory_gb']} "
            f"image_tokens={row['image_token_count']} "
            f"text_tokens={row['text_token_count']} "
            f"kv_cache_gb={row['kv_cache_memory_gb']:.4f}",
            flush=True,
        )
        if not row["success"]:
            print(f"  error: {row['error']}", flush=True)

    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
