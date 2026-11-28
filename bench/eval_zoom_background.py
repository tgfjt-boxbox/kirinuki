"""Background errors of the two-pass zoom on the EasyPortrait images where the zoom fires.

usage: .venv/bin/python bench/eval_zoom_background.py ROOT ONE_PASS_DIR ZOOM_DIR ZOOM_RESULT_JSON OUT_JSON
ROOT holds images/<id>.jpg and test/<id>.png (every class > 0 is person). ONE_PASS_DIR and ZOOM_DIR hold person
probabilities <id>.png from bench/browser/litert_masks.html (#ep/LIST/whole/gpu and #ep/LIST/whole/zoom).
ZOOM_RESULT_JSON is the zoom run's result; its `boxes` names the images where the zoom fired.
False-positive rate = background pixels predicted as person / background pixels. "inside" and "outside" split the
pixels by the zoom box; outside the box the zoom keeps the first pass, so its difference there should be near 0.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

THRESHOLD = 0.5
PASSES = ("one_pass", "zoom")


def person(path, shape):
    return np.asarray(Image.open(path).convert("L").resize(shape[::-1], Image.BILINEAR)) / 255.0 >= THRESHOLD


def box_mask(box, shape):
    h, w = shape
    x, y, bw, bh = box
    inside = np.zeros(shape, bool)
    inside[max(0, round(y * h)):min(h, round((y + bh) * h)), max(0, round(x * w)):min(w, round((x + bw) * w))] = True
    return inside


def counts(pred, gt, region):
    bg = ~gt & region
    return {"fp": int((pred & bg).sum()), "bg": int(bg.sum()), "tp": int((pred & gt & region).sum()),
            "person": int((gt & region).sum()), "union": int(((pred | gt) & region).sum())}


def paired(a, b):
    d = np.asarray(b) - np.asarray(a)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("one_pass", type=Path)
    p.add_argument("zoom", type=Path)
    p.add_argument("zoom_result", type=Path)
    p.add_argument("out", type=Path)
    args = p.parse_args()
    boxes = json.loads(args.zoom_result.read_text())["boxes"]
    regions = ("all", "inside", "outside")
    rows = {}
    for i, box in sorted(boxes.items()):
        gt = np.array(Image.open(args.root / "test" / f"{i}.png")) > 0
        inside = box_mask(box, gt.shape)
        preds = {"one_pass": person(args.one_pass / f"{i}.png", gt.shape), "zoom": person(args.zoom / f"{i}.png", gt.shape)}
        rows[i] = {k: {r: counts(pred, gt, m) for r, m in zip(regions, (np.ones_like(gt), inside, ~inside))}
                   for k, pred in preds.items()}
        rows[i]["box_area"] = box[2] * box[3]
    summary = {}
    for r in regions:
        s = {}
        for k in PASSES:
            tot = {c: sum(rows[i][k][r][c] for i in rows) for c in ("fp", "bg", "tp", "person", "union")}
            s[k] = {"pooled_fp_rate": tot["fp"] / max(1, tot["bg"]), "pooled_recall": tot["tp"] / max(1, tot["person"]),
                    "pooled_iou": tot["tp"] / max(1, tot["union"]), **tot}
        fp = {k: [rows[i][k][r]["fp"] / max(1, rows[i][k][r]["bg"]) for i in rows] for k in PASSES}
        iou = {k: [rows[i][k][r]["tp"] / max(1, rows[i][k][r]["union"]) for i in rows] for k in PASSES}
        s["image_fp_rate_diff"], s["image_fp_rate_diff_se"] = paired(fp["one_pass"], fp["zoom"])
        s["image_iou_diff"], s["image_iou_diff_se"] = paired(iou["one_pass"], iou["zoom"])
        s["images_fp_up"] = sum(b > a for a, b in zip(fp["one_pass"], fp["zoom"]))
        s["images_fp_down"] = sum(b < a for a, b in zip(fp["one_pass"], fp["zoom"]))
        summary[r] = s
        print(f"{r}\tfp {s['one_pass']['pooled_fp_rate']:.4%} -> {s['zoom']['pooled_fp_rate']:.4%}"
              f"\timage diff {s['image_fp_rate_diff']:+.4%} ± {s['image_fp_rate_diff_se']:.4%}"
              f"\tup/down {s['images_fp_up']}/{s['images_fp_down']}"
              f"\tiou {s['one_pass']['pooled_iou']:.4f} -> {s['zoom']['pooled_iou']:.4f}")
    print(f"zoomed images {len(rows)}")
    args.out.write_text(json.dumps({"threshold": THRESHOLD, "images": len(rows), "summary": summary, "per_image": rows}, indent=1))


if __name__ == "__main__":
    main()
