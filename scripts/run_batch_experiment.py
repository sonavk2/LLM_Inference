"""Sweep batch sizes (and optionally context lengths) for one model.

Phase 2 — batched inference. For each (context_length, batch_size) pair we run
greedy generation, replicate one prompt batch_size times, and record the same
metrics as the single-request sweep. tokens_per_second is the AGGREGATE
throughput across the batch; per-request throughput is `tokens_per_second
/ batch_size` and is computed in the analysis notebook.

OOM is recorded as success=False with the error string, same as Phase 1, so a
sweep keeps going past the (context, batch) memory wall.

Run from the repo root, e.g.:

    python scripts/run_batch_experiment.py \\
        --config configs/baseline_hf.yaml \\
        --model-config configs/llama3_1_8b_instruct.yaml \\
        --context-lengths 8192 32768 \\
        --batch-sizes 1 2 4 8 16 \\
        --max-new-tokens 64 \\
        --results-path results/phase2_llama31_a100.jsonl
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backends.hf_backend import HFBackend
from src.benchmark.config import load_experiment_config, load_yaml
from src.benchmark.memory import detect_device, device_label
from src.benchmark.result_writer import append_jsonl
from src.benchmark.runner import run_single_experiment


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="Path to experiment YAML")
    p.add_argument(
        "--model-config", default=None,
        help="Override the model_config pointer in the experiment YAML.",
    )
    p.add_argument("--context-lengths", nargs="+", type=int, required=True)
    p.add_argument("--batch-sizes", nargs="+", type=int, required=True)
    p.add_argument("--max-new-tokens", type=int, default=64)
    p.add_argument("--hardware", default=None,
                   help="Hardware label to record (defaults to auto-detected device name).")
    p.add_argument("--results-path", default=None,
                   help="Override the results_path from the config.")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_experiment_config(args.config)
    if args.model_config:
        cfg["model"] = load_yaml(args.model_config)
    model_cfg = cfg["model"]

    if model_cfg.get("is_vlm"):
        raise SystemExit(
            "Model config marks itself as is_vlm=true. "
            "VLM batched inference isn't implemented; pick a text model."
        )

    model_id = model_cfg["model_id"]
    dtype = model_cfg.get("dtype", "bfloat16")
    trust = model_cfg.get("trust_remote_code", False)
    rope_scaling = model_cfg.get("rope_scaling")
    quantization = model_cfg.get("quantization")
    native_context = model_cfg.get("native_context")
    quant_label = "4bit-nf4" if (quantization and quantization.get("load_in_4bit")) else None

    device = detect_device()
    hardware = args.hardware or device_label(device)
    results_path = Path(args.results_path or cfg["output"]["results_path"])

    print(f"Loading {model_id} (dtype={dtype}, quant={quant_label or 'none'}) on {device} ...", flush=True)
    backend = HFBackend(
        model_id=model_id, dtype=dtype, device=device, trust_remote_code=trust,
        rope_scaling=rope_scaling, quantization=quantization,
    )
    backend.load()
    print("Model loaded.", flush=True)

    for ctx in args.context_lengths:
        is_native = None if native_context is None else (ctx <= native_context)
        for bsz in args.batch_sizes:
            print(f"\n--- ctx={ctx}, batch={bsz}, native={is_native} ---", flush=True)
            row = run_single_experiment(
                backend=backend,
                context_length=ctx,
                batch_size=bsz,
                max_new_tokens=args.max_new_tokens,
                hardware_label=hardware,
                is_native_context=is_native,
                quantization_label=quant_label,
            )
            append_jsonl(results_path, row)
            if row["success"]:
                # tokens_per_second is aggregate (output_tokens summed across batch).
                per_req_tps = row["tokens_per_second"] / bsz
                print(
                    f"  ok  ttft={row['ttft_seconds']:.3f}s "
                    f"total={row['total_latency_seconds']:.3f}s "
                    f"agg_tps={row['tokens_per_second']:.2f} "
                    f"per_req_tps={per_req_tps:.2f} "
                    f"peak={row['peak_gpu_memory_gb']:.2f}GB "
                    f"kv_est={row['kv_cache_memory_gb']:.2f}GB",
                    flush=True,
                )
            else:
                print(f"  FAIL: {(row['error'] or '')[:140]}", flush=True)

    print(f"\nResults written to {results_path}")


if __name__ == "__main__":
    main()
