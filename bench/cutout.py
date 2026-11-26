"""Render a side-by-side cutout sheet for each input image, one row per baseline model.

usage: .venv/bin/python bench/cutout.py OUT_DIR IMAGE...
Each image is center-cropped to 4:3 and resized to 640x480, matching the production pipeline.
Columns: input, confidence mask, mask overlay, composite on a flat background.
The overlay and the composite use alpha = smoothstep(0.2, 0.8) of the confidence, because multiclass never drops
below about 0.08 on the background and raw alpha lets the old room show through (eval_soft_alpha.py).
"""
import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw

from eval_person_iou import MODELS, segmenter
from eval_soft_alpha import smoothstep

SIZE = (640, 480)
BACKGROUND = np.array([40, 110, 220], dtype=np.float32)
OVERLAY = np.array([255, 40, 40], dtype=np.float32)
ALPHA = smoothstep(0.2, 0.8)


def crop_box(w, h):
    target = SIZE[0] / SIZE[1]
    if w / h > target:
        cw = round(h * target)
        return (w - cw) // 2, 0, (w - cw) // 2 + cw, h
    ch = round(w / target)
    return 0, (h - ch) // 2, w, (h - ch) // 2 + ch


def to_640x480(img):
    return img.crop(crop_box(*img.size)).resize(SIZE, Image.LANCZOS)


def panels(rgb, prob):
    a = ALPHA(prob)[..., None]
    mask = np.repeat(prob[..., None] * 255, 3, axis=2)
    overlay = rgb * (1 - 0.45 * a) + OVERLAY * 0.45 * a
    composite = rgb * a + BACKGROUND * (1 - a)
    return [rgb, mask, overlay, composite]


def sheet(rgb, probs):
    w, h = SIZE
    label_h = 28
    out = Image.new("RGB", (w * 4, (h + label_h) * len(probs)), "white")
    draw = ImageDraw.Draw(out)
    for row, (name, prob) in enumerate(probs.items()):
        y = row * (h + label_h)
        draw.text((8, y + 8), name, fill="black")
        for col, panel in enumerate(panels(rgb.astype(np.float32), prob)):
            out.paste(Image.fromarray(np.clip(panel, 0, 255).astype(np.uint8)), (col * w, y + label_h))
    return out


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    segs = {name: segmenter(name) for name in MODELS}
    for path in map(Path, sys.argv[2:]):
        rgb = np.array(to_640x480(Image.open(path).convert("RGB")))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        probs = {
            name: np.squeeze(to_person([m.numpy_view() for m in segs[name].segment(image).confidence_masks]))
            for name, to_person in MODELS.items()
        }
        dest = out_dir / f"{path.stem}.png"
        sheet(rgb, probs).save(dest)
        print(dest)


if __name__ == "__main__":
    main()
