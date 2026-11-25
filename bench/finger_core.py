"""Measure whether each baseline model keeps the fingers, using finger cores built from hand keypoints.

usage: .venv/bin/python bench/finger_core.py OUT_DIR IMAGE...
A finger core is a capsule along the last three MediaPipe hand landmarks of each finger (PIP -> DIP -> TIP;
MCP -> IP -> TIP for the thumb), with radius
CORE_RADIUS x (wrist-to-middle-MCP distance). It sits inside the finger, so it gives finger recall, not IoU.
Prints one JSON line per finger and writes an overlay of the cores for a visual check.
Fingers whose core lies wholly outside the frame are skipped. Images with no detected hand are reported on stderr.
"""
import json
import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image, ImageDraw

from cutout import SIZE, to_640x480
from eval_person_iou import MODELS, segmenter

FINGERS = {"thumb": (2, 3, 4), "index": (6, 7, 8), "middle": (10, 11, 12), "ring": (14, 15, 16), "pinky": (18, 19, 20)}
CORE_RADIUS = 0.06
THRESHOLD = 0.5
KEPT = 0.5


def landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"), num_hands=2
    )
    return vision.HandLandmarker.create_from_options(options)


def cores(landmarks):
    p = np.array([(q.x * SIZE[0], q.y * SIZE[1]) for q in landmarks])
    r = max(2, round(CORE_RADIUS * np.linalg.norm(p[0] - p[9])))
    for name, idx in FINGERS.items():
        pts = [tuple(p[i]) for i in idx]
        canvas = Image.new("L", SIZE, 0)
        draw = ImageDraw.Draw(canvas)
        draw.line(pts, fill=255, width=2 * r)
        for x, y in pts:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
        yield name, pts, r, np.array(canvas) > 0


def main():
    out_dir = Path(sys.argv[1])
    out_dir.mkdir(parents=True, exist_ok=True)
    hands = landmarker()
    segs = {name: segmenter(name) for name in MODELS}
    for path in map(Path, sys.argv[2:]):
        rgb = np.array(to_640x480(Image.open(path).convert("RGB")))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        probs = {
            name: np.squeeze(to_person([m.numpy_view() for m in segs[name].segment(image).confidence_masks]))
            for name, to_person in MODELS.items()
        }
        overlay = Image.fromarray(rgb)
        draw = ImageDraw.Draw(overlay)
        detected = hands.detect(image).hand_landmarks
        if not detected:
            print(f"{path.name}: no hand detected", file=sys.stderr)
        for hand, landmarks in enumerate(detected):
            for finger, pts, r, core in cores(landmarks):
                if not core.any():
                    continue
                recall = {name: float((prob[core] >= THRESHOLD).mean()) for name, prob in probs.items()}
                kept = {name: v >= KEPT for name, v in recall.items()}
                print(json.dumps({"image": path.name, "hand": hand, "finger": finger, "core_px": int(core.sum()),
                                  "recall": recall, "kept": kept}))
                draw.line(pts, fill=(255, 0, 0), width=1)
                for x, y in pts:
                    draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 0))
        overlay.save(out_dir / f"{path.stem}.png")


if __name__ == "__main__":
    main()
