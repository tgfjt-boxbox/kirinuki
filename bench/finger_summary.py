"""Summarize finger_core.py output into the metrics defined in docs/finger-metric.md.

usage: .venv/bin/python bench/finger_core.py OUT_DIR IMAGE... > fingers.jsonl
       .venv/bin/python bench/finger_summary.py fingers.jsonl
Per model: finger survival rate with its standard error, pooled finger-core recall, and survival per finger type.
"""
import json
import math
import sys
from collections import defaultdict


def summarize(rows):
    models = sorted(rows[0]["recall"]) if rows else []
    out = {}
    for m in models:
        kept = [r["kept"][m] for r in rows]
        rate = sum(kept) / len(kept)
        by_finger = defaultdict(list)
        for r in rows:
            by_finger[r["finger"]].append(r["kept"][m])
        out[m] = {
            "fingers": len(kept),
            "survival": rate,
            "survival_se": math.sqrt(rate * (1 - rate) / len(kept)),
            "core_recall": sum(r["recall"][m] * r["core_px"] for r in rows) / sum(r["core_px"] for r in rows),
            "survival_by_finger": {f: sum(v) / len(v) for f, v in sorted(by_finger.items())},
        }
    return out


if __name__ == "__main__":
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    print(json.dumps({"images": len({r["image"] for r in rows}), "models": summarize(rows)}, indent=1))
