# Long-Context LLM Inference Benchmark

An independent-study project to measure how long-context workloads affect LLM inference performance across models, hardware, and serving frameworks.

## Why this matters

As context windows grow from 8k to 128k+ tokens, attention cost and KV-cache size become the dominant performance bottlenecks. This project produces controlled, reproducible measurements so we can:

- See where memory pressure forces failures.
- Compare model architectures (full MHA vs GQA vs MQA) by their actual KV-cache footprint.
- Compare serving frameworks (Hugging Face vs vLLM vs TensorRT-LLM) on the same workloads.

## What gets measured

Per run, the benchmark records:

- **TTFT** — time to first token
- **TPOT** — time per output token
- **Total latency**
- **Tokens / second**
- **Peak GPU memory**
- **Estimated KV-cache memory**
- **Success or OOM failure**

Each row is also tagged with the model, backend, hardware, context length, batch size, and `max_new_tokens` so results from different machines and backends can be joined.

## Status

**Phase 1 (Hugging Face baseline) is being scaffolded — the code is not yet runnable.** This commit lays out the directory tree, configs, and dependencies. See `CLAUDE.md` for the full plan.

## Install

```bash
# 1. Create a virtualenv
python -m venv .venv && source .venv/bin/activate

# 2. Install PyTorch for your platform first.
#    See https://pytorch.org/get-started/locally/ for the right wheel.
#    Examples:
#      Mac:        pip install torch
#      CUDA 12.1:  pip install torch --index-url https://download.pytorch.org/whl/cu121

# 3. Install the rest
pip install -r requirements.txt
```

## Planned commands (Phase 1, not yet implemented)

Sweep one model across context lengths, batch size 1:

```bash
python scripts/run_context_sweep.py \
  --config configs/baseline_hf.yaml \
  --context-lengths 8192 16384 32768 65536 \
  --max-new-tokens 128
```

Sweep batch sizes at a fixed context length:

```bash
python scripts/run_batch_experiment.py \
  --config configs/baseline_hf.yaml \
  --context-length 8192 \
  --batch-sizes 1 2 4 8 \
  --max-new-tokens 128
```

Aggregate the JSONL results in `results/`:

```bash
python scripts/summarize_results.py --results-dir results/
```

## Result files

Results are written as JSONL (one experiment per line) into `results/`. Each row carries the full set of metrics plus the experiment metadata (model, backend, hardware, context length, batch size). Failed runs (e.g. OOM) are still recorded with `success: false` and the error string, so memory limits show up as data rather than gaps.

## Repository layout

```
configs/      Experiment, model, and hardware-profile YAML
scripts/      CLI entry points (run_*.py, summarize_results.py)
src/
  benchmark/  Backend-agnostic harness (config, prompts, metrics, memory, results)
  backends/   One module per serving framework (HF first, vLLM/TRT later)
  analysis/   Aggregation and plotting helpers
results/      JSONL output
notebooks/    Ad-hoc analysis
docs/         Background notes (attention, batching, scheduling)
```

The harness in `src/benchmark/` is backend-agnostic. Adding a new backend means writing one module in `src/backends/` that implements the same small interface; the rest of the framework should not need to change.

## Future work

- **Phase 2**: batched runs to study throughput vs latency tradeoffs.
- **Phase 3**: comparison against [vLLM](https://github.com/vllm-project/vllm) (continuous batching, paged KV cache) and [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) (compiled kernels). These have heavy, platform-specific installs and will live in separate `requirements-vllm.txt` / `requirements-tensorrt.txt` files added later.
- Add a streaming generation path so TTFT is measured directly rather than estimated.
- Add reasoning and vision-language models for capability-vs-cost tradeoff plots.
