"""Run a Phase-4 vLLM batch sweep (context x batch)."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backends.vllm_backend import VLLMBackend
from src.benchmark.config import load_experiment_config, load_yaml
from src.benchmark.memory import detect_device, device_label
from src.benchmark.result_writer import append_jsonl
from src.benchmark.vllm_runner import run_single_vllm_experiment


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to experiment YAML")
    p.add_argument("--model-config", default=None)
    p.add_argument("--context-lengths", nargs="+", type=int, required=True)
    p.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--max-model-len", type=int, default=None)
    p.add_argument("--gpu-mem-util", type=float, default=0.9)
    p.add_argument("--hardware", default=None)
    p.add_argument("--results-path", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment_config(args.config)
    if args.model_config:
        cfg["model"] = load_yaml(args.model_config)
    model_cfg = cfg["model"]

    if model_cfg.get("is_vlm"):
        raise SystemExit("Phase 4 vLLM batch sweep does not support VLMs.")

    model_id = model_cfg["model_id"]
    dtype = model_cfg.get("dtype", "bfloat16")
    trust = model_cfg.get("trust_remote_code", False)
    rope_scaling = model_cfg.get("rope_scaling")
    native_context = model_cfg.get("native_context")
    max_model_len = args.max_model_len or (max(args.context_lengths) + args.max_new_tokens + 8)

    device = detect_device()
    hardware = args.hardware or device_label(device)
    results_path = Path(args.results_path or cfg["output"]["results_path"])

    print(
        f"Loading {model_id} on vLLM (dtype={dtype}, max_model_len={max_model_len}, "
        f"gpu_mem_util={args.gpu_mem_util}) ...",
        flush=True,
    )
    backend = VLLMBackend(
        model_id=model_id, dtype=dtype, device=device,
        trust_remote_code=trust, rope_scaling=rope_scaling,
        max_model_len=max_model_len,
        gpu_memory_utilization=args.gpu_mem_util,
    )
    backend.load()
    print("Model loaded.", flush=True)

    for ctx in args.context_lengths:
        is_native = None if native_context is None else (ctx <= native_context)
        for bsz in args.batch_sizes:
            print(f"\n--- ctx={ctx}, batch={bsz}, native={is_native} ---", flush=True)
            row = run_single_vllm_experiment(
                backend=backend,
                context_length=ctx,
                batch_size=bsz,
                max_new_tokens=args.max_new_tokens,
                hardware_label=hardware,
                is_native_context=is_native,
            )
            append_jsonl(results_path, row)
            if row["success"]:
                per_req_tps = row["tokens_per_second"] / bsz
                print(
                    f"  ok  ttft={row['ttft_seconds']:.3f}s "
                    f"total={row['total_latency_seconds']:.3f}s "
                    f"agg_tps={row['tokens_per_second']:.2f} "
                    f"per_req_tps={per_req_tps:.2f} "
                    f"peak={row['peak_gpu_memory_gb']:.2f}GB",
                    flush=True,
                )
            else:
                print(f"  FAIL: {(row['error'] or '')[:140]}", flush=True)

    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
