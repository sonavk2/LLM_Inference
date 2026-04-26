"""Regenerate the Colab notebooks under notebooks/.

Run from repo root:  python scripts/build_notebooks.py
"""

import json
from pathlib import Path


def md(source):
    return {"cell_type": "markdown", "metadata": {}, "source": source}


def code(source):
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": source,
    }


NOTEBOOK_META = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
    "colab": {"provenance": []},
    "accelerator": "GPU",
}


def write_notebook(path, cells):
    nb = {
        "cells": cells,
        "metadata": NOTEBOOK_META,
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    print(f"wrote {path}")


# ---------------------------------------------------------------------------
# Notebook 1: end-to-end run on Colab GPU
# ---------------------------------------------------------------------------

run_cells = [
    md(
        "# LLM Inference Baseline — Colab GPU run\n"
        "\n"
        "End-to-end Phase 1: verify GPU, install deps, smoke-test the harness on a tiny "
        "model, then run the real long-context sweep, then summarize.\n"
        "\n"
        "**Before starting:** `Runtime → Change runtime type → T4 GPU` (free) or "
        "A100 (Colab Pro+).\n"
    ),
    md("## 1. Confirm GPU"),
    code("!nvidia-smi"),
    md(
        "## 2. Get the code\n"
        "\n"
        "Edit `REPO_URL` to point at your GitHub repo, then run the cell. "
        "If the notebook is already inside a clone, the clone is skipped."
    ),
    code(
        "import os\n"
        "\n"
        "REPO_URL = \"https://github.com/YOUR_USERNAME/LLM_Inference.git\"  # <-- EDIT THIS\n"
        "\n"
        "if not os.path.exists(\"scripts/run_context_sweep.py\"):\n"
        "    !git clone {REPO_URL}\n"
        "    repo_name = REPO_URL.rstrip(\"/\").split(\"/\")[-1].replace(\".git\", \"\")\n"
        "    %cd {repo_name}\n"
        "\n"
        "!ls"
    ),
    md(
        "## 3. Install Python deps\n"
        "\n"
        "Colab already ships PyTorch with CUDA, so we only install the rest."
    ),
    code("!pip install -q -r requirements.txt"),
    md(
        "## 4. (Optional) Hugging Face auth\n"
        "\n"
        "Only needed for gated models like Llama-3. Accept the license on the model page "
        "first, then create a token at https://huggingface.co/settings/tokens and paste it "
        "into the prompt."
    ),
    code(
        "from huggingface_hub import login\n"
        "login()"
    ),
    md(
        "## 5. Smoke test (~30s, tiny public model)\n"
        "\n"
        "Verifies the harness end-to-end before pulling real weights. Should produce two "
        "JSONL rows in `results/baseline_hf.jsonl` with non-null TTFT and latency."
    ),
    code(
        "!python scripts/run_context_sweep.py \\\n"
        "  --config configs/baseline_hf.yaml \\\n"
        "  --model-id sshleifer/tiny-gpt2 \\\n"
        "  --dtype float16 \\\n"
        "  --context-lengths 128 512 1024 \\\n"
        "  --max-new-tokens 16"
    ),
    code("!cat results/baseline_hf.jsonl"),
    md(
        "## 6. Real Phase 1 sweep\n"
        "\n"
        "Edit the cell for your runtime:\n"
        "\n"
        "- **T4 (16 GB):** use `--dtype float16`. Llama-3-8B weights alone are ~16 GB so "
        "loading is tight and OOM at long context is expected — those failure rows are "
        "the data point that shows the memory wall.\n"
        "- **A100 (40 GB):** `bfloat16` is fine for the full 8k→64k sweep.\n"
        "\n"
        "If you skipped the HF login above, override the model with `--model-id` to "
        "something non-gated (e.g. `Qwen/Qwen2.5-1.5B-Instruct`, `microsoft/phi-2`)."
    ),
    code(
        "# T4 example (Llama-3-8B in fp16)\n"
        "!python scripts/run_context_sweep.py \\\n"
        "  --config configs/baseline_hf.yaml \\\n"
        "  --dtype float16 \\\n"
        "  --context-lengths 2048 4096 8192 16384 \\\n"
        "  --max-new-tokens 64\n"
        "\n"
        "# A100 example — uncomment if applicable\n"
        "# !python scripts/run_context_sweep.py \\\n"
        "#   --config configs/baseline_hf.yaml \\\n"
        "#   --context-lengths 8192 16384 32768 65536 \\\n"
        "#   --max-new-tokens 128"
    ),
    md("## 7. Summarize"),
    code("!python scripts/summarize_results.py --results-dir results/"),
    md(
        "## 8. Save results before the runtime disconnects\n"
        "\n"
        "Free Colab resets `/content` on disconnect. Either download to your laptop or "
        "push back to GitHub."
    ),
    code(
        "# Option A: download to laptop\n"
        "from google.colab import files\n"
        "files.download(\"results/baseline_hf.jsonl\")\n"
        "\n"
        "# Option B: push to GitHub from inside Colab (uncomment, edit name/email)\n"
        "# !git config user.email \"you@example.com\"\n"
        "# !git config user.name \"Your Name\"\n"
        "# !git add results/ && git commit -m \"Colab T4 baseline run\" && git push"
    ),
]


# ---------------------------------------------------------------------------
# Notebook 2: result analysis + plots
# ---------------------------------------------------------------------------

analysis_cells = [
    md(
        "# Result analysis\n"
        "\n"
        "Loads JSONL rows from `results/` and plots:\n"
        "\n"
        "1. TTFT vs context length\n"
        "2. Decode throughput (tokens/sec) vs context length\n"
        "3. Peak GPU memory vs KV-cache estimate\n"
        "4. Latency vs throughput scatter\n"
        "\n"
        "Runs in Colab or locally — needs only pandas + matplotlib."
    ),
    md("## 1. Get the code (skip if running locally inside the repo)"),
    code(
        "import os\n"
        "\n"
        "REPO_URL = \"https://github.com/YOUR_USERNAME/LLM_Inference.git\"  # <-- EDIT IF CLONING\n"
        "\n"
        "if not os.path.exists(\"results\"):\n"
        "    !git clone {REPO_URL}\n"
        "    repo_name = REPO_URL.rstrip(\"/\").split(\"/\")[-1].replace(\".git\", \"\")\n"
        "    %cd {repo_name}\n"
        "\n"
        "!ls results/"
    ),
    md("## 2. Load every JSONL row into a DataFrame"),
    code(
        "import json\n"
        "from pathlib import Path\n"
        "import pandas as pd\n"
        "\n"
        "rows = []\n"
        "for path in sorted(Path(\"results\").glob(\"*.jsonl\")):\n"
        "    with open(path) as f:\n"
        "        for line in f:\n"
        "            line = line.strip()\n"
        "            if line:\n"
        "                rows.append(json.loads(line))\n"
        "\n"
        "df = pd.DataFrame(rows)\n"
        "print(f\"Loaded {len(df)} rows\")\n"
        "df.head()"
    ),
    md("## 3. Quick overview"),
    code(
        "print(f\"Successes: {df['success'].sum()} / {len(df)}\")\n"
        "print(f\"Models:    {df['model_name'].unique().tolist()}\")\n"
        "print(f\"Backends:  {df['backend'].unique().tolist()}\")\n"
        "print(f\"Hardware:  {df['hardware'].unique().tolist()}\")\n"
        "print(f\"Contexts:  {sorted(df['context_length'].unique().tolist())}\")"
    ),
    md(
        "## 4. TTFT vs context length\n"
        "\n"
        "Linear-ish growth is the expected baseline. Steeper growth suggests memory "
        "pressure or prefill recomputation."
    ),
    code(
        "import matplotlib.pyplot as plt\n"
        "\n"
        "ok = df[df[\"success\"]].sort_values(\"context_length\")\n"
        "fig, ax = plt.subplots(figsize=(8, 5))\n"
        "for (model, backend), g in ok.groupby([\"model_name\", \"backend\"]):\n"
        "    ax.plot(g[\"context_length\"], g[\"ttft_seconds\"], marker=\"o\", label=f\"{model} ({backend})\")\n"
        "ax.set_xscale(\"log\", base=2)\n"
        "ax.set_xlabel(\"Context length (tokens)\")\n"
        "ax.set_ylabel(\"TTFT (s)\")\n"
        "ax.set_title(\"Time to first token vs context length\")\n"
        "ax.grid(True, alpha=0.3)\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## 5. Decode throughput vs context length\n"
        "\n"
        "Tokens/sec at decode is typically flat or mildly degrading with context, because "
        "every decode step reads the whole KV cache."
    ),
    code(
        "fig, ax = plt.subplots(figsize=(8, 5))\n"
        "for (model, backend), g in ok.groupby([\"model_name\", \"backend\"]):\n"
        "    ax.plot(g[\"context_length\"], g[\"tokens_per_second\"], marker=\"o\", label=f\"{model} ({backend})\")\n"
        "ax.set_xscale(\"log\", base=2)\n"
        "ax.set_xlabel(\"Context length (tokens)\")\n"
        "ax.set_ylabel(\"Tokens / sec\")\n"
        "ax.set_title(\"Decode throughput vs context length\")\n"
        "ax.grid(True, alpha=0.3)\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## 6. Memory: measured peak vs KV-cache estimate\n"
        "\n"
        "The gap between the two is everything else (model weights, activations, framework "
        "overhead). When they grow together, KV cache is the dominant cost — which is "
        "the whole motivation for paged-KV designs (vLLM)."
    ),
    code(
        "fig, ax = plt.subplots(figsize=(8, 5))\n"
        "for (model, backend), g in ok.groupby([\"model_name\", \"backend\"]):\n"
        "    if g[\"peak_gpu_memory_gb\"].notna().any():\n"
        "        ax.plot(g[\"context_length\"], g[\"peak_gpu_memory_gb\"],\n"
        "                marker=\"o\", label=f\"{model} measured peak\")\n"
        "    ax.plot(g[\"context_length\"], g[\"kv_cache_memory_gb\"],\n"
        "            marker=\"s\", linestyle=\"--\", label=f\"{model} KV-cache estimate\")\n"
        "ax.set_xscale(\"log\", base=2)\n"
        "ax.set_xlabel(\"Context length (tokens)\")\n"
        "ax.set_ylabel(\"GB\")\n"
        "ax.set_title(\"Memory: peak GPU vs KV-cache estimate\")\n"
        "ax.grid(True, alpha=0.3)\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## 7. Latency vs throughput scatter\n"
        "\n"
        "Each point is one (model, context length) run. Up-and-to-the-left wins. Useful for "
        "spotting Pareto-frontier candidates as more backends/models get added."
    ),
    code(
        "fig, ax = plt.subplots(figsize=(8, 5))\n"
        "for (model, backend), g in ok.groupby([\"model_name\", \"backend\"]):\n"
        "    ax.scatter(g[\"total_latency_seconds\"], g[\"tokens_per_second\"],\n"
        "               label=f\"{model} ({backend})\")\n"
        "    for _, row in g.iterrows():\n"
        "        ax.annotate(f\"{int(row['context_length'])}\",\n"
        "                    (row[\"total_latency_seconds\"], row[\"tokens_per_second\"]),\n"
        "                    fontsize=8, alpha=0.7)\n"
        "ax.set_xlabel(\"Total latency (s)\")\n"
        "ax.set_ylabel(\"Tokens / sec\")\n"
        "ax.set_title(\"Latency vs throughput (labels = context length)\")\n"
        "ax.grid(True, alpha=0.3)\n"
        "ax.legend()\n"
        "plt.tight_layout()\n"
        "plt.show()"
    ),
    md(
        "## What to look at\n"
        "\n"
        "- **Where TTFT bends sharply upward** — that's the prefill/memory cliff for "
        "this model on this hardware.\n"
        "- **Where throughput drops** — decode is becoming KV-cache-bandwidth bound.\n"
        "- **Failure rows** (in `results/*.jsonl` with `success: false`) mark the OOM wall. "
        "Check `summarize_results.py` output for the list."
    ),
]


def main():
    repo_root = Path(__file__).resolve().parent.parent
    write_notebook(repo_root / "notebooks/01_run_baseline_colab.ipynb", run_cells)
    write_notebook(repo_root / "notebooks/02_analysis.ipynb", analysis_cells)


if __name__ == "__main__":
    main()
