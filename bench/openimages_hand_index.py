"""Build the hand-region evaluation index from Open Images: hand boxes paired with the union of person masks.

usage: .venv/bin/python bench/openimages_hand_index.py OPENIMAGES_DIR SPLIT OUT_CSV
Reads <SPLIT>-annotations-object-segmentation.csv, <SPLIT>-annotations-bbox.csv and <SPLIT>-masks/<SPLIT>-masks-<h>.zip.
Writes one row per non-group hand box in an image with at least one person mask, with the fraction
of the box covered by the union of person masks.
"""
import csv
import io
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

PERSON = {"/m/01g317", "/m/04yx4", "/m/03bt1vf", "/m/01bl7v", "/m/05r655"}
HAND = "/m/0k65p"


def main():
    root, split, out = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
    masks = defaultdict(list)
    for r in csv.DictReader(open(root / f"{split}-annotations-object-segmentation.csv")):
        if r["LabelName"] in PERSON:
            masks[r["ImageID"]].append(r["MaskPath"])
    hands = defaultdict(list)
    for r in csv.DictReader(open(root / f"{split}-annotations-bbox.csv")):
        if r["LabelName"] == HAND and r["IsGroupOf"] == "0" and r["ImageID"] in masks:
            hands[r["ImageID"]].append(tuple(float(r[k]) for k in ("XMin", "XMax", "YMin", "YMax")))
    zips = {}
    rows = []
    for image_id, boxes in sorted(hands.items()):
        union = None
        for path in masks[image_id]:
            h = path[0]
            if h not in zips:
                zips[h] = zipfile.ZipFile(root / f"{split}-masks" / f"{split}-masks-{h}.zip")
            m = np.array(Image.open(io.BytesIO(zips[h].read(path))).convert("L")) > 0
            union = m if union is None else (union | m)
        H, W = union.shape
        for x0, x1, y0, y1 in boxes:
            c0, c1 = int(x0 * W), max(int(x0 * W) + 1, int(round(x1 * W)))
            r0, r1 = int(y0 * H), max(int(y0 * H) + 1, int(round(y1 * H)))
            coverage = union[r0:r1, c0:c1].mean()
            rows.append((image_id, x0, x1, y0, y1, f"{coverage:.4f}", ";".join(masks[image_id])))
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ImageID", "XMin", "XMax", "YMin", "YMax", "coverage", "PersonMasks"])
        w.writerows(rows)
    cov = np.array([float(r[5]) for r in rows])
    print(f"hand boxes {len(rows)} in {len(hands)} images")
    for t in (0.1, 0.2, 0.3, 0.4, 0.5):
        sel = cov >= t
        print(f"coverage >= {t:.1f}: {sel.sum()} boxes in {len({r[0] for r, s in zip(rows, sel) if s})} images")


if __name__ == "__main__":
    main()
