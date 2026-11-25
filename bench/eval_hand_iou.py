"""Hand-region IoU: person-mask IoU inside Open Images hand boxes, for MediaPipe segmenters.

usage: .venv/bin/python bench/eval_hand_iou.py IMAGE_ID_LIST OUT_JSON
Uses data/openimages/hand_eval_index_test.csv (bench/openimages_hand_index.py), boxes with coverage >= 0.3,
images in data/openimages/test-images/, person masks from data/openimages/test-masks/.
"""
import csv
import io
import json
import sys
import zipfile
from collections import defaultdict
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image

from eval_person_iou import MODELS, THRESHOLD, segmenter

ROOT = Path("data/openimages")
MIN_COVERAGE = 0.3


def gt_union(paths, size, zips):
    union = np.zeros(size[::-1], bool)
    for p in paths:
        z = zips.setdefault(p[0], zipfile.ZipFile(ROOT / "test-masks" / f"test-masks-{p[0]}.zip"))
        m = Image.open(io.BytesIO(z.read(p))).convert("L").resize(size, Image.NEAREST)
        union |= np.array(m) > 0
    return union


def main():
    wanted = set(Path(sys.argv[1]).read_text().split())
    out = Path(sys.argv[2])
    boxes = defaultdict(list)
    for r in csv.DictReader(open(ROOT / "hand_eval_index_test.csv")):
        if r["ImageID"] in wanted and float(r["coverage"]) >= MIN_COVERAGE:
            boxes[r["ImageID"]].append(r)
    zips = {}
    samples = []
    for image_id, rs in sorted(boxes.items()):
        img = Image.open(ROOT / "test-images" / f"{image_id}.jpg").convert("RGB")
        gt = gt_union(rs[0]["PersonMasks"].split(";"), img.size, zips)
        W, H = img.size
        rects = [(int(float(r["YMin"]) * H), max(int(float(r["YMin"]) * H) + 1, round(float(r["YMax"]) * H)),
                  int(float(r["XMin"]) * W), max(int(float(r["XMin"]) * W) + 1, round(float(r["XMax"]) * W)))
                 for r in rs]
        samples.append((image_id, np.array(img), gt, rects))
    box_meta = [{"image": image_id, "person_area": round(float(gt.mean()), 4)}
                for image_id, _, gt, rects in samples for _ in rects]
    results = {}
    for name, to_person in MODELS.items():
        ious, inter, union = [], 0, 0
        with segmenter(name) as seg:
            for image_id, rgb, gt, rects in samples:
                res = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
                pred = np.squeeze(to_person([m.numpy_view() for m in res.confidence_masks])) >= THRESHOLD
                for r0, r1, c0, c1 in rects:
                    p, g = pred[r0:r1, c0:c1], gt[r0:r1, c0:c1]
                    a, b = int((p & g).sum()), int((p | g).sum())
                    ious.append(a / b if b else 1.0)
                    inter, union = inter + a, union + b
        ious = np.array(ious)
        results[name] = {"mean_box_iou": float(ious.mean()), "pooled_iou": inter / union, "box_ious": ious.round(4).tolist()}
        print(f"{name}\tboxes {len(ious)}\tmean {ious.mean():.4f}\tpooled {inter / union:.4f}")
    out.write_text(json.dumps({"images": len(samples), "min_coverage": MIN_COVERAGE, "boxes": box_meta, "models": results}, indent=1))


if __name__ == "__main__":
    main()
