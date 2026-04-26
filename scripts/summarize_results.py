"""Summarize JSONL benchmark results.

Reads every *.jsonl file in --results-dir, prints:
  - overall counts (runs, successes, failures)
  - failure rows with their error strings
  - means of latency / throughput / memory grouped by (model, backend, hardware)
    and either context_length or batch_size
  - the (model, backend) winner per context_length by tokens/sec

Run from the repo root:
    python scripts/summarize_results.py --results-dir results/
"""

import argparse
import json
from pathlib import Path

import pandas as pd


METRIC_COLS = [
    "ttft_seconds",
    "tpot_seconds",
    "total_latency_seconds",
    "tokens_per_second",
    "peak_gpu_memory_gb",
    "kv_cache_memory_gb",
]


def load_results(results_dir):
    """Read every *.jsonl file under results_dir into a single DataFrame."""
    rows = []
    for path in sorted(Path(results_dir).glob("*.jsonl")):
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                row["_source_file"] = path.name
                rows.append(row)
    if not rows:
        raise SystemExit(f"No result rows found in {results_dir}/*.jsonl")
    return pd.DataFrame(rows)


def print_overview(df):
    print("\n=== Overview ===")
    print(f"Total runs:    {len(df)}")
    print(f"Successes:     {int(df['success'].sum())}")
    print(f"Failures:      {int((~df['success']).sum())}")
    print(f"Models:        {sorted(df['model_name'].unique().tolist())}")
    print(f"Backends:      {sorted(df['backend'].unique().tolist())}")
    print(f"Hardware:      {sorted(df['hardware'].unique().tolist())}")


def print_failures(df):
    fails = df[~df["success"]]
    if fails.empty:
        return
    print(f"\n=== Failures ({len(fails)}) ===")
    cols = ["model_name", "backend", "hardware", "context_length", "batch_size", "error"]
    cols = [c for c in cols if c in fails.columns]
    print(fails[cols].to_string(index=False))


def aggregate(df, group_by_extra):
    """Group successful runs and print the mean of each metric."""
    s = df[df["success"]]
    if s.empty:
        print(f"\n(No successful runs to aggregate by {group_by_extra}.)")
        return
    keys = ["model_name", "backend", "hardware", group_by_extra]
    available = [m for m in METRIC_COLS if m in s.columns]
    agg = s.groupby(keys, dropna=False)[available].mean(numeric_only=True).round(4)
    counts = s.groupby(keys, dropna=False).size().rename("n_runs")
    out = pd.concat([counts, agg], axis=1).reset_index()
    print(f"\n=== Means by {group_by_extra} ===")
    print(out.to_string(index=False))


def best_by_throughput(df):
    """For each context length, show which (model, backend) had the highest tokens/sec."""
    s = df[df["success"] & df["tokens_per_second"].notna()]
    if s.empty:
        return
    rows = []
    for ctx, group in s.groupby("context_length"):
        winner = group.loc[group["tokens_per_second"].idxmax()]
        rows.append({
            "context_length": int(ctx),
            "model": winner["model_name"],
            "backend": winner["backend"],
            "hardware": winner["hardware"],
            "tokens_per_second": round(float(winner["tokens_per_second"]), 2),
        })
    if rows:
        print("\n=== Best (model, backend) per context_length by tokens/sec ===")
        print(pd.DataFrame(rows).to_string(index=False))


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument(
        "--results-dir", default="results",
        help="Directory of *.jsonl files (default: results/)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    df = load_results(args.results_dir)
    print_overview(df)
    print_failures(df)
    aggregate(df, "context_length")
    if "batch_size" in df.columns and df["batch_size"].nunique() > 1:
        aggregate(df, "batch_size")
    best_by_throughput(df)


if __name__ == "__main__":
    main()
