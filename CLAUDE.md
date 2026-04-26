# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Status

The repository is currently empty. This file describes the intended project so that the first scaffolding work has a clear target. When real code lands, update this file to match what actually exists — do not leave aspirational claims here once the code diverges.

## Project

Independent-study benchmark for **long-context LLM inference**. The deliverable is a small, modular Python codebase that runs controlled inference experiments and records how prompt length, batch size, model choice, hardware, and serving framework affect:

- TTFT (time to first token)
- TPOT (time per output token)
- Total latency and tokens/sec throughput
- Peak GPU memory and KV-cache growth
- Success / OOM failure

Phases, in order:

1. Single-request baseline on Hugging Face Transformers, sweeping context length (8k → 16k → 32k → 64k).
2. Batched runs (batch sizes 1, 2, 4, 8) on the same baseline to see throughput vs. latency tradeoffs.
3. Backend comparison: same workloads against vLLM and TensorRT-LLM.

Hardware targets: local Mac (CPU/MPS), NVIDIA T4, NVIDIA A100. The hardware string is part of every result row.

## Architecture intent

Two layers that should stay decoupled:

- **`src/benchmark/`** — backend-agnostic harness: config loading, prompt construction, metrics recording, memory probes, result writing. Knows nothing about HF/vLLM/TRT specifics.
- **`src/backends/`** — one module per serving system (`hf_backend.py`, `vllm_backend.py`, `tensorrt_backend.py`), each implementing the same small interface (`load(model_cfg)`, `generate(prompts, max_new_tokens) -> GenerationResult`). Adding a backend should not require touching the harness.

Scripts in `scripts/` are thin CLI entry points that wire a config + a backend + a sweep dimension (context length or batch size) and write JSONL/CSV into `results/`. Analysis (`src/analysis/`, `notebooks/`) reads from `results/` only — never re-runs experiments.

Start with one model end-to-end (e.g. `meta-llama/Llama-3-8B` or `Qwen/Qwen3-9B`) before generalizing.

## Non-obvious constraints

- **Verify prompt token length with the tokenizer** after generating synthetic prompts. Don't trust character/word counts — long-context experiments are meaningless if the actual token count drifts from the target.
- **Graceful degradation when CUDA is absent.** GPU-memory probes (`torch.cuda.reset_peak_memory_stats`, `max_memory_allocated`, `max_memory_reserved`) must be guarded; on Mac/CPU return `null` for those fields rather than crashing. The harness must run on a laptop even if no real numbers come back.
- **OOM is a valid result, not a crash.** Catch it, record `success: false` with the error string, and continue the sweep.
- **KV-cache memory is estimated, not measured**, using `2 × layers × batch × context × kv_heads × head_dim × bytes_per_element`. Pull the architectural constants from the model config; the factor of 2 is K + V. This estimate is what makes MHA/MQA/GQA differences visible in results.
- **TTFT requires a streaming/token-callback path.** A first pass with `model.generate()` can only report total latency and approximate TPOT; plan to add a streaming path before the numbers are used for serious comparisons.
- **Every result row records model, backend, hardware, context length, batch size, and max_new_tokens** alongside the metrics — results across machines/backends must be joinable without guessing.

## Planned commands

These are the intended CLI shapes; create the scripts to match.

```bash
# Context-length sweep, single request
python scripts/run_context_sweep.py \
  --config configs/baseline_hf.yaml \
  --context-lengths 8192 16384 32768 65536 \
  --batch-size 1 --max-new-tokens 128

# Batch-size sweep at a fixed context length
python scripts/run_batch_experiment.py \
  --config configs/baseline_hf.yaml \
  --context-length 8192 \
  --batch-sizes 1 2 4 8 --max-new-tokens 128

# Aggregate results from results/*.jsonl
python scripts/summarize_results.py --results-dir results/
```

## Style

Student/research-project Python: clear names, light comments, no premature abstraction. Add a backend or a metric only when an experiment needs it; resist building plugin systems before the second backend exists.
