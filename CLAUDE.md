# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

Independent-study benchmark for **long-context LLM inference**. Runs controlled inference experiments and records how prompt length, batch size, model choice, hardware, and serving framework affect:

- TTFT (time to first token)
- TPOT (time per output token)
- Total latency and tokens/sec throughput
- Peak GPU memory and KV-cache growth
- Success / OOM failure

Phases, in order — each gets one notebook in `notebooks/`:

1. **`phase1_baseline.ipynb`** — single-request sweep across instruction (Llama-3.1-8B-Instruct), reasoning architecture (Qwen3-8B), and VLM (Qwen2-VL-7B-Instruct), context length 8k → 16k → 32k → 64k. T4 included as a single load-OOM data point; A100 is the main platform.
2. **`phase2_batching.ipynb`** — Llama-3.1-8B-Instruct only, sweep batch size 1/2/4/8/16 at context 8k and 32k, single-prompt-tiled batches. Goal: latency vs throughput tradeoff and the (context × batch) memory frontier.
3. **`phase4_vllm.ipynb`** — Llama-3.1-8B-Instruct only, runs both Phase 1 and Phase 2 sweep dimensions on the **vLLM** backend so the two backends can be plotted side by side. Headline question: how much do PagedAttention and continuous batching change throughput, TTFT, and the feasibility frontier vs the HF baseline. **TensorRT-LLM is in scope for the original plan but deferred** — engine compilation per model is too high-overhead for this study; vLLM alone covers the central PagedAttention / KV-cache claim.

Hardware targets: NVIDIA A100 (primary). T4 documented in Phase 1 only (8B-class bf16 weights don't fit in 16 GB). Mac MPS supported by the harness but not used in the current sweeps.

### Phase 1 model lineup

| Slot | Model | Native ctx | Notes |
|---|---|---|---|
| Instruction | `meta-llama/Meta-Llama-3.1-8B-Instruct` | 131072 | Gated; `huggingface-cli login` first. |
| Reasoning   | `Qwen/Qwen3-8B` | 32768 | YaRN factor=4 enabled so 64k is in the trained range. |
| VLM         | `Qwen/Qwen2-VL-7B-Instruct` | 32768 | YaRN factor=4. Image-token count is subtracted from text target so total context = target. |
| Extension   | Qwen3-8B + 4-bit NF4 | 32768 | T4-only; needs bitsandbytes. |

## Architecture intent

Two layers that should stay decoupled:

- **`src/benchmark/`** — backend-agnostic harness: config loading, prompt construction (text + VLM), metrics, memory probes, result writing. Knows nothing about HF / vLLM / TRT specifics.
- **`src/backends/`** — one module per serving system (`hf_backend.py`, `vlm_backend.py`, `vllm_backend.py`), each implementing the same small interface (`load()`, `generate(prompt, max_new_tokens) → GenerationResult`). Adding a backend should not require touching the harness.

Scripts in `scripts/` are thin CLI entry points that wire a config + a backend + a sweep dimension. Analysis cells live inside the per-phase notebooks and read from `results/` only — never re-run experiments.

### Directory layout

```
notebooks/
  phase1_baseline.ipynb         — single-request, model × context sweep (Phase 1)
  phase2_batching.ipynb         — batched inference, batch × context sweep (Phase 2)
  phase4_vllm.ipynb             — vLLM vs HF backend comparison (Phase 4)
  archive/                      — early scaffold notebooks, kept for reference

scripts/
  run_context_sweep.py          — Phase 1 text models (HF)
  run_vlm_context_sweep.py      — Phase 1 VLM (HF)
  run_batch_experiment.py       — Phase 2 (HF)
  run_vllm_context_sweep.py     — Phase 4 single-request (vLLM)
  run_vllm_batch_experiment.py  — Phase 4 batched (vLLM)
  summarize_results.py          — text aggregator over JSONL files

results/
  phase1_<model>_<hw>.jsonl
  phase2_<model>_<hw>.jsonl
  phase4_<model>_<hw>.jsonl     — both Phase-4 sweeps in one file
  plots/                        — PNGs written by the notebooks
  archive/                      — pre-rewrite results files
```

## Result schema

Each JSONL row carries:

- `model_name`, `backend`, `hardware` (e.g. `cuda:NVIDIA A100`), `context_length`, `batch_size`, `max_new_tokens`
- `ttft_seconds`, `tpot_seconds`, `total_latency_seconds`, `tokens_per_second`
- `peak_gpu_memory_gb`, `kv_cache_memory_gb`
- `success`, `error`
- `prompt_format`: `"synthetic_repeat"` or `"vlm_image+text"`
- `is_native_context`: `true` if `context_length <= native_context`, else `false`. Lets you filter YaRN-extrapolated cells out of analysis.
- `image_token_count`, `text_token_count`: VLM rows only. Sum equals `context_length`. **`image_token_count` includes chat-template scaffolding** (role markers, vision-start/end, bos/eos, generation prompt) — not image-pixel tokens alone — because that scaffolding is a fixed overhead we subtract from the text budget. `text_token_count` is the user-controllable text portion.
- `quantization`: `"4bit-nf4"` for the extension experiment, else `null`.

## Non-obvious constraints

- **Verify prompt token length with the tokenizer / processor** after generating synthetic prompts. Don't trust character/word counts. For VLM, the image expands into N image tokens that count against the total — we subtract those from the text target so `context_length` is the true total.
- **Graceful degradation when CUDA is absent.** Memory probes return `null` on Mac / CPU rather than crashing.
- **OOM is a valid result, not a crash.** `runner.py` catches `torch.cuda.OutOfMemoryError` *and* generic `RuntimeError("...out of memory...")` flavors, records `success: false` with the error string, and continues the sweep. `torch.cuda.empty_cache()` is called between iterations on success too, so fragmentation doesn't push later cells into false OOMs.
- **KV-cache memory is estimated, not measured**, using `2 × layers × batch × context × kv_heads × head_dim × bytes_per_element`. Architectural constants come from the model config; for VLMs we read `config.text_config` so MHA/MQA/GQA differences in the text stack still show up.
- **TTFT is measured, not estimated**, via a `BaseStreamer` hook that timestamps the first generated token (skipping the prompt-tokens callback HF emits up front).
- **YaRN is applied via `rope_scaling` on the model config before weights load.** Cells past `native_context` are tagged `is_native_context: false` so we can filter "extrapolated" rows out of fairness comparisons.
- **Reasoning-mode caveat.** Phase 1 prompts are tiled "the quick brown fox" — no chat template, no `<think>` tags. The Qwen3-8B rows measure the *architecture* under generic tokens, not its reasoning-mode behavior. Do not write up "reasoning models are slower at decode" from this data alone.

## Commands

```bash
# Phase 1 — text models. Use --model-config to swap models without
# duplicating the experiment YAML.
python scripts/run_context_sweep.py \
  --config configs/baseline_hf.yaml \
  --model-config configs/llama3_1_8b_instruct.yaml \
  --context-lengths 8192 16384 32768 65536 \
  --max-new-tokens 128 \
  --results-path results/phase1_llama31_a100.jsonl

python scripts/run_context_sweep.py \
  --config configs/baseline_hf.yaml \
  --model-config configs/qwen3_8b.yaml \
  --context-lengths 8192 16384 32768 65536 \
  --max-new-tokens 128 \
  --results-path results/phase1_qwen3_a100.jsonl

# Phase 1 — VLM. Image is fixed; image_tokens are subtracted from the text target.
python scripts/run_vlm_context_sweep.py \
  --config configs/baseline_hf.yaml \
  --model-config configs/qwen2_vl_7b.yaml \
  --context-lengths 8192 16384 32768 65536 \
  --max-new-tokens 128 \
  --results-path results/phase1_qwen2vl_a100.jsonl

# Phase 2 — batched inference (Llama-3.1, batch 1/2/4/8/16 at ctx 8k & 32k).
python scripts/run_batch_experiment.py \
  --config configs/baseline_hf.yaml \
  --model-config configs/llama3_1_8b_instruct.yaml \
  --context-lengths 8192 32768 \
  --batch-sizes 1 2 4 8 16 \
  --max-new-tokens 64 \
  --results-path results/phase2_llama31_a100.jsonl

# Phase 4 — vLLM context sweep (single-request, mirrors Phase 1).
pip install -r requirements-vllm.txt
python scripts/run_vllm_context_sweep.py \
  --config configs/baseline_hf.yaml \
  --model-config configs/llama3_1_8b_instruct.yaml \
  --context-lengths 8192 16384 32768 65536 \
  --max-new-tokens 128 \
  --max-model-len 65664 \
  --results-path results/phase4_llama31_a100.jsonl

# Phase 4 — vLLM batch sweep (mirrors Phase 2; appends to the same file).
python scripts/run_vllm_batch_experiment.py \
  --config configs/baseline_hf.yaml \
  --model-config configs/llama3_1_8b_instruct.yaml \
  --context-lengths 8192 32768 \
  --batch-sizes 1 2 4 8 16 \
  --max-new-tokens 64 \
  --max-model-len 33024 \
  --results-path results/phase4_llama31_a100.jsonl

# Aggregate
python scripts/summarize_results.py --results-dir results/
```

In Phase 2's JSONL, `tokens_per_second` is **aggregate** across the batch (sum of output tokens / total latency). Per-request throughput is `tokens_per_second / batch_size` and is computed in the analysis cell.

## Phase 4 / vLLM caveats

- **Peak GPU memory is not comparable across backends.** vLLM pre-allocates ~90% of GPU memory upfront (controlled by `gpu_memory_utilization`) for the KV cache pool, so `peak_gpu_memory_gb` will look ~constant on every Phase-4 row regardless of context length. The `phase4_vllm.ipynb` plots intentionally drop peak memory from the comparison; the formula-based `kv_cache_memory_gb` (identical math for both backends) stays in.
- **vLLM doesn't OOM where HF did.** Under workloads that crashed HF in Phase 2, vLLM serializes requests when the KV pool fills — `success` stays `True` but `total_latency` balloons because requests queue serially. The frontier figure shows this as "HF hard-fails; vLLM soft-degrades." Read the Pareto plot together with the frontier matrix; total latency tells you when soft degradation kicked in.
- **TTFT comes from `RequestOutput.metrics`**, not a streamer hook. Requires `disable_log_stats=False` (default in our backend). The smoke-test cell in `phase4_vllm.ipynb` asserts that TTFT is non-null before the long sweep starts.

Result files are named `<phase>_<model>_<hardware>.jsonl` so cross-platform comparisons join cleanly.

## Workflow

Notebooks are run in Colab; outputs are committed alongside the .ipynb files. Cursor and Colab both edit the repo, so **always `git pull` before starting work in either place** to avoid clobbering the other side's changes.

## Style

Student/research-project Python: clear names, light comments, no premature abstraction. Add a backend or a metric only when an experiment needs it; resist building plugin systems before the second backend exists.
