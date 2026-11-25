"""Person-mask IoU of MediaPipe segmenters on EasyPortrait samples (accuracy only; latency is measured in the browser).

usage: .venv/bin/python bench/eval_person_iou.py SAMPLE_DIR OUT_JSON
SAMPLE_DIR holds ids.txt, images/<id>.jpg and test/<id>.png (EasyPortrait: every class > 0 is person).
Person probability: selfie_segmenter* emit it directly; multiclass emits 6 classes, so person = 1 - background.
The image is passed whole; MediaPipe resizes to the model input and returns masks at image resolution.
"""
import json
import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image

MODELS = {
    "selfie_multiclass_256x256": lambda masks: 1.0 - masks[0],
    "selfie_segmenter": lambda masks: masks[0],
    "selfie_segmenter_landscape": lambda masks: masks[0],
}
THRESHOLD = 0.5
BOUNDARY_TOLERANCE = 0.008


def dilate(mask, r):
    for axis in (0, 1):
        n = mask.shape[axis]
        padded = np.pad(mask, [(r, r) if a == axis else (0, 0) for a in (0, 1)])
        mask = np.logical_or.reduce([np.take(padded, range(k, k + n), axis=axis) for k in range(2 * r + 1)])
    return mask


def boundary(mask):
    inner = mask.copy()
    inner[1:, :] &= mask[:-1, :]
    inner[:-1, :] &= mask[1:, :]
    inner[:, 1:] &= mask[:, :-1]
    inner[:, :-1] &= mask[:, 1:]
    return mask & ~inner


def boundary_f(pred, gt):
    r = max(1, round(BOUNDARY_TOLERANCE * np.hypot(*gt.shape)))
    bp, bg = boundary(pred), boundary(gt)
    if not bp.any() or not bg.any():
        return float(bp.any() == bg.any())
    precision = (bp & dilate(bg, r)).sum() / bp.sum()
    recall = (bg & dilate(bp, r)).sum() / bg.sum()
    return 0.0 if precision + recall == 0 else float(2 * precision * recall / (precision + recall))


def segmenter(name):
    options = vision.ImageSegmenterOptions(
        base_options=BaseOptions(model_asset_path=f"models/{name}.tflite"),
        output_confidence_masks=True,
        output_category_mask=False,
    )
    return vision.ImageSegmenter.create_from_options(options)


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    ids = root.joinpath("ids.txt").read_text().split()
    segs = {name: segmenter(name) for name in MODELS}
    iou = {name: {} for name in MODELS}
    bf = {name: {} for name in MODELS}
    totals = {name: [0, 0] for name in MODELS}
    for i in ids:
        rgb = np.array(Image.open(root / "images" / f"{i}.jpg").convert("RGB"))
        gt = np.array(Image.open(root / "test" / f"{i}.png")) > 0
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        for name, to_person in MODELS.items():
            res = segs[name].segment(image)
            prob = np.squeeze(to_person([m.numpy_view() for m in res.confidence_masks]))
            assert prob.shape == gt.shape, f"{name} {i}: mask {prob.shape} vs gt {gt.shape}"
            pred = prob >= THRESHOLD
            a, b = int((pred & gt).sum()), int((pred | gt).sum())
            iou[name][i] = a / b
            bf[name][i] = boundary_f(pred, gt)
            totals[name][0] += a
            totals[name][1] += b
    for seg in segs.values():
        seg.close()
    results = {}
    for name in MODELS:
        ious, bfs = np.array(list(iou[name].values())), np.array(list(bf[name].values()))
        dataset_iou = totals[name][0] / totals[name][1]
        results[name] = {
            "mean_image_iou": float(ious.mean()),
            "dataset_iou": dataset_iou,
            "min_image_iou": float(ious.min()),
            "worst": min(iou[name], key=iou[name].get),
            "mean_boundary_f": float(bfs.mean()),
            "per_image": iou[name],
            "per_image_boundary_f": bf[name],
        }
        print(f"{name}\tmean {ious.mean():.4f}\tdataset {dataset_iou:.4f}\tmin {ious.min():.4f}\tboundaryF {bfs.mean():.4f}")
    out.write_text(json.dumps({"n": len(ids), "threshold": THRESHOLD, "boundary_tolerance": BOUNDARY_TOLERANCE,
                               "models": results}, indent=1))


if __name__ == "__main__":
    main()
