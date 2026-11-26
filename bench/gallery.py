"""Render a one-file HTML gallery of where each model loses fingers, for a visual check next to the numbers.

usage: .venv/bin/python bench/gallery.py OUT.html FINGERS.jsonl IMAGE_DIR [--worst N] [--extra IMAGE...]
FINGERS.jsonl is finger_core.py output. The gallery shows the N images where multiclass loses the most fingers,
then any --extra images (ties keep FINGERS.jsonl order).
Hands are re-detected with HandLandmarker, so rows from --keypoints (source: manual) may count differently. Each row is the input and one background-replaced composite per model, with every finger
core outlined: green where the model kept the finger, red where it lost it.
"""
import argparse
import base64
import html
import io
import json
import sys
from collections import Counter
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw

from cutout import ALPHA, BACKGROUND, to_640x480
from eval_person_iou import MODELS, segmenter
from finger_core import KEPT, THRESHOLD, cores, from_landmarks, landmarker

LEAD = "selfie_multiclass_256x256"
KEPT_COLOR, LOST_COLOR = (40, 220, 90), (255, 40, 40)


def worst_images(fingers_jsonl, n):
    lost = Counter()
    for line in filter(str.strip, fingers_jsonl.read_text().splitlines()):
        row = json.loads(line)
        lost[row["image"]] += not row["kept"][LEAD]
    return [name for name, count in lost.most_common(n) if count]


def render(path, detector, segs):
    rgb = np.array(to_640x480(Image.open(path).convert("RGB")))
    image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    probs = {name: np.squeeze(to_person([m.numpy_view() for m in segs[name].segment(image).confidence_masks]))
             for name, to_person in MODELS.items()}
    hands = [from_landmarks(lm) for lm in detector.detect(image).hand_landmarks]
    if not hands:
        print(f"{path.name}: no hand detected", file=sys.stderr)
    panels = [("input", Image.fromarray(rgb), "")]
    for name, prob in probs.items():
        a = ALPHA(prob)[..., None]
        panel = Image.fromarray(np.clip(rgb * a + BACKGROUND * (1 - a), 0, 255).astype(np.uint8))
        draw = ImageDraw.Draw(panel)
        kept_count = total = 0
        for hand in hands:
            for _, pts, r, core in cores(hand):
                if not core.any():
                    continue
                kept = (prob[core] >= THRESHOLD).mean() >= KEPT
                kept_count, total = kept_count + kept, total + 1
                color = KEPT_COLOR if kept else LOST_COLOR
                draw.line(pts, fill=color, width=2)
                for x, y in pts:
                    draw.ellipse((x - r, y - r, x + r, y + r), outline=color, width=2)
        panels.append((name, panel, f"{kept_count}/{total}"))
    return panels


def data_uri(img):
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out", type=Path)
    p.add_argument("fingers_jsonl", type=Path)
    p.add_argument("image_dir", type=Path)
    p.add_argument("--worst", type=int, default=12)
    p.add_argument("--extra", type=Path, nargs="*", default=[])
    args = p.parse_args()
    paths = [args.image_dir / name for name in worst_images(args.fingers_jsonl, args.worst)] + args.extra
    detector = landmarker()
    segs = {name: segmenter(name) for name in MODELS}
    rows = []
    for path in paths:
        cells = "".join(
            f'<figure><img src="{data_uri(img)}" alt=""><figcaption>{html.escape(label)}'
            f'{" — 指 " + score if score else ""}</figcaption></figure>'
            for label, img, score in render(path, detector, segs)
        )
        rows.append(f'<section><h2>{html.escape(path.name)}</h2><div class="row">{cells}</div></section>')
    args.out.write_text(
        "<!doctype html><meta charset=utf-8><title>Finger gallery</title>"
        "<style>body{font-family:system-ui,sans-serif;margin:16px;background:#fff;color:#111}"
        ".row{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px}"
        "figure{margin:0}img{width:100%;display:block}figcaption{font-size:12px}h2{font-size:14px}</style>"
        "<p>緑の枠: そのモデルが残した指。赤の枠: 消した指。背景は青に置き換えている。</p>" + "".join(rows)
    )
    print(args.out)


if __name__ == "__main__":
    main()
