"""Append-only JSONL writer for benchmark result rows."""

import json
from pathlib import Path


def append_jsonl(path, row):
    """Append one row to a JSONL file, creating dirs as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")
