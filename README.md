# Long-Context LLM Inference Benchmark

Independent-study benchmark measuring how prompt length, batch size, model architecture, hardware, and serving framework affect LLM inference performance.

## What gets measured

Per (model, context, batch, backend, hardware) cell:

- **TTFT** — time to first token (real, not estimated)
- **TPOT** — time per output token
- **Total latency** and **tokens/sec throughput** (aggregate across batch)
- **Peak GPU memory** (caveat: vLLM pre-allocates a KV-cache pool)
- **Estimated KV-cache memory** (formula-based, identical across backends)
- **Success or OOM** — failures recorded as `success: false` rows, not crashes

## Status

Three phases complete on NVIDIA A100 (40 GB). Llama-3.1-8B-Instruct, Qwen3-8B, and Qwen2-VL-7B-Instruct were swept across 8k → 64k context, batched runs swept 1 → 16 at 8k & 32k, and the same workloads were re-run on vLLM for backend comparison. T4 is documented as a hardware-tier OOM data point. TensorRT-LLM was deferred (engine compilation overhead per model).

## Reproducing the experiments

Each phase has one notebook under `notebooks/`. Open in Colab on an A100 runtime, run top-to-bottom.

| Notebook | What it does |
|---|---|
| `phase1_baseline.ipynb` | Single-request sweep: 3 models × 4 context lengths |
| `phase2_batching.ipynb` | Batched sweep: Llama-3.1, batch 1/2/4/8/16 at ctx 8k & 32k |
| `phase4_vllm.ipynb`     | vLLM vs HF, mirrors Phase 1 + Phase 2 dimensions on Llama-3.1 |

Each notebook installs deps, runs the sweep, and writes:
- A JSONL results file: `results/phase<N>_<model>_<hw>.jsonl`
- Plots: `results/plots/phase<N>_*.png`

## Headline findings

- **TTFT scales superlinearly with context length** on all three architectures (prefill is O(N²)).
- **Qwen2-VL has ~half the per-token KV-cache footprint** of Llama-3.1 / Qwen3 — visible in single-request memory plots.
- **Static batching frontier** on A100: 8k context fits batch 8, 32k fits only batch 2; KV cache scales linearly with `batch × context`.
- **vLLM eliminates HF's hard-OOM frontier.** Every (context, batch) cell that crashed HF in Phase 2 succeeds on vLLM via PagedAttention, trading hard failure for soft latency degradation under saturation.
- **vLLM doesn't help single-request TTFT** (prefill-bound). The win is decode + scheduling: ~2× single-request throughput, ~40% more peak batched throughput.
- **Hardware tier matters more than fine-grained platform choice.** 8B-class bf16 weights (~16 GB) cannot load on a 16 GB T4 at any context length.

## Install

vLLM is a separate, heavy install — base requirements are kept lean.

```bash
python -m venv .venv && source .venv/bin/activate

# Install PyTorch for your platform first (https://pytorch.org/get-started/locally/).
pip install torch                                    # Mac
# pip install torch --index-url https://download.pytorch.org/whl/cu121   # CUDA

pip install -r requirements.txt
pip install -r requirements-vllm.txt                 # only for Phase 4
```

`Meta-Llama-3.1-8B-Instruct` is gated on Hugging Face: `huggingface-cli login` and accept the license before the first run.

## CLI commands

The notebooks call these scripts; you can also run them directly.

```bash
# Phase 1 — single-request context sweep
python scripts/run_context_sweep.py \
  --config configs/baseline_hf.yaml \
  --model-config configs/llama3_1_8b_instruct.yaml \
  --context-lengths 8192 16384 32768 65536 \
  --max-new-tokens 128 \
  --results-path results/phase1_llama31_a100.jsonl

# Phase 1 — VLM variant (image + text)
python scripts/run_vlm_context_sweep.py \
  --config configs/baseline_hf.yaml \
  --model-config configs/qwen2_vl_7b.yaml \
  --context-lengths 8192 16384 32768 65536 \
  --max-new-tokens 128 \
  --results-path results/phase1_qwen2vl_a100.jsonl

# Phase 2 — batched HF
python scripts/run_batch_experiment.py \
  --config configs/baseline_hf.yaml \
  --model-config configs/llama3_1_8b_instruct.yaml \
  --context-lengths 8192 32768 \
  --batch-sizes 1 2 4 8 16 \
  --max-new-tokens 64 \
  --results-path results/phase2_llama31_a100.jsonl

# Phase 4 — vLLM (single-request and batched, same JSONL)
python scripts/run_vllm_context_sweep.py     --config configs/baseline_hf.yaml --model-config configs/llama3_1_8b_instruct.yaml --context-lengths 8192 16384 32768 65536 --max-new-tokens 128 --max-model-len 65664 --results-path results/phase4_llama31_a100.jsonl
python scripts/run_vllm_batch_experiment.py  --config configs/baseline_hf.yaml --model-config configs/llama3_1_8b_instruct.yaml --context-lengths 8192 32768 --batch-sizes 1 2 4 8 16 --max-new-tokens 64 --max-model-len 33024 --results-path results/phase4_llama31_a100.jsonl

# Aggregate
python scripts/summarize_results.py --results-dir results/
```

## Repository layout

```
configs/        Model + experiment YAMLs
notebooks/      One notebook per phase
scripts/        CLI entry points (HF + vLLM sweeps, summary)
src/
  benchmark/    Backend-agnostic harness (config, prompts, metrics, runners)
  backends/     hf_backend, vlm_backend, vllm_backend
results/        JSONL output + plots/
requirements.txt          Base (HF Transformers)
requirements-vllm.txt     Phase 4 only
```

The harness is backend-agnostic. Adding a new backend means writing one module in `src/backends/` that implements `load()` and `generate(input_ids, max_new_tokens) → GenerationResult`; the rest of the framework doesn't change.

## Result files

Each row in a `phase<N>_*.jsonl` carries: model, backend, hardware, context_length, batch_size, max_new_tokens, plus all measured metrics. OOMs land as `success: false` rows with the error string, so memory limits show up as data rather than gaps. See `CLAUDE.md` for the full schema and per-backend caveats.
