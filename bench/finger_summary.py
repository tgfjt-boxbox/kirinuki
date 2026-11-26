"""Summarize finger_core.py output into the metrics defined in docs/finger-metric.md.

usage: .venv/bin/python bench/finger_core.py OUT_DIR IMAGE... > fingers.jsonl
       .venv/bin/python bench/finger_summary.py fingers.jsonl
Per model: finger survival rate with its standard error, pooled finger-core recall, and survival per finger type.
The standard error treats the image as the sampling unit (cluster-robust, ratio estimator): fingers of one image
tend to be lost together, so treating fingers as independent understates it.
"""
import json
import math
import sys
from collections import defaultdict


def clustered_se(rows, m, rate):
    per_image = defaultdict(lambda: [0, 0])
    for r in rows:
        per_image[r["image"]][0] += r["kept"][m]
        per_image[r["image"]][1] += 1
    g = len(per_image)
    if g < 2:
        return float("nan")
    resid = sum((k - rate * n) ** 2 for k, n in per_image.values())
    return math.sqrt(g / (g - 1) * resid) / len(rows)


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
            "survival_se": clustered_se(rows, m, rate),
            "core_recall": sum(r["recall"][m] * r["core_px"] for r in rows) / sum(r["core_px"] for r in rows),
            "survival_by_finger": {f: sum(v) / len(v) for f, v in sorted(by_finger.items())},
        }
    return out


if __name__ == "__main__":
    rows = [json.loads(line) for line in open(sys.argv[1]) if line.strip()]
    print(json.dumps({"images": len({r["image"] for r in rows}), "models": summarize(rows)}, indent=1))
