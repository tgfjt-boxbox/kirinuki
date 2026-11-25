"""How much the background shows through when the confidence is used directly as alpha.

usage: .venv/bin/python bench/eval_soft_alpha.py SAMPLE_DIR OUT_JSON
SAMPLE_DIR is laid out as for eval_person_iou.py. For each model and each alpha curve it reports
bleed = mean alpha on ground-truth background pixels (the old room showing through the new background) and
hole = mean (1 - alpha) on ground-truth person pixels (the person turning see-through), both pooled over all pixels.
A curve maps the confidence p to alpha; "raw" is alpha = p and smoothstep(lo, hi) clamps both ends.
"""
import json
import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image

from eval_person_iou import MODELS, segmenter


def smoothstep(lo, hi):
    def curve(p):
        t = np.clip((p - lo) / (hi - lo), 0, 1)
        return t * t * (3 - 2 * t)
    return curve


CURVES = {
    "raw": lambda p: p,
    "smoothstep(0.2,0.8)": smoothstep(0.2, 0.8),
    "smoothstep(0.3,0.7)": smoothstep(0.3, 0.7),
    "step(0.5)": lambda p: (p >= 0.5).astype(np.float32),
}
HIST_EDGES = [0, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0001]


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    ids = root.joinpath("ids.txt").read_text().split()
    segs = {name: segmenter(name) for name in MODELS}
    sums = {m: {c: [0.0, 0.0] for c in CURVES} for m in MODELS}
    hist = {m: np.zeros(len(HIST_EDGES) - 1) for m in MODELS}
    n_bg = n_fg = 0
    for i in ids:
        rgb = np.array(Image.open(root / "images" / f"{i}.jpg").convert("RGB"))
        gt = np.array(Image.open(root / "test" / f"{i}.png")) > 0
        n_bg += int((~gt).sum())
        n_fg += int(gt.sum())
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        for name, to_person in MODELS.items():
            p = np.squeeze(to_person([m.numpy_view() for m in segs[name].segment(image).confidence_masks]))
            hist[name] += np.histogram(p[~gt], bins=HIST_EDGES)[0]
            for c, curve in CURVES.items():
                a = curve(p)
                sums[name][c][0] += float(a[~gt].sum())
                sums[name][c][1] += float((1 - a[gt]).sum())
    results = {
        name: {
            "background_confidence_histogram": dict(zip(
                [f"[{lo},{hi:.2g})" for lo, hi in zip(HIST_EDGES, HIST_EDGES[1:])],
                (hist[name] / n_bg).tolist())),
            "curves": {c: {"bleed": b / n_bg, "hole": h / n_fg} for c, (b, h) in sums[name].items()},
        }
        for name in MODELS
    }
    for name, r in results.items():
        print(name)
        for c, v in r["curves"].items():
            print(f"  {c:22s} bleed {v['bleed']:.4f}  hole {v['hole']:.4f}")
        print("  bg hist", {k: round(v, 4) for k, v in r["background_confidence_histogram"].items()})
    out.write_text(json.dumps({"n": len(ids), "models": results}, indent=1))


if __name__ == "__main__":
    main()
