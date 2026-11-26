"""Measure where a model's person mask goes wrong on EasyPortrait images, to support labelling failure types.

usage: .venv/bin/python bench/failure_areas.py RESULT.json MODEL DATA_DIR IDS.txt > areas.csv
Per image: false-positive and false-negative pixels as a share of the ground-truth person area.
The larger of the two tells whether a failure is something kept that is not person, or person cut away.
"""
import csv
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from compare import Run, Segmenters

def main():
    result, model, data, ids = sys.argv[1], sys.argv[2], Path(sys.argv[3]), Path(sys.argv[4]).read_text().split()
    d = json.loads(Path(result).read_text())
    run = Run(result, model, d["threshold"], {})
    per = d["models"][model]["per_image"]
    segs = Segmenters()
    out = csv.writer(sys.stdout)
    out.writerow(["id", "iou", "fp_share", "fn_share"])
    for i in ids:
        rgb = np.array(Image.open(data / "images" / f"{i}.jpg").convert("RGB"))
        gt = np.array(Image.open(data / "test" / f"{i}.png")) > 0
        pred = segs.predict(run, rgb)
        n = gt.sum()
        out.writerow([i, f"{per[i]:.4f}", f"{(pred & ~gt).sum() / n:.3f}", f"{(gt & ~pred).sum() / n:.3f}"])
    segs.close()


if __name__ == "__main__":
    main()
