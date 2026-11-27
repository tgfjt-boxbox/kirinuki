"""Measure whether each baseline model keeps the fingers, using finger cores built from hand keypoints.

usage: .venv/bin/python bench/finger_core.py OUT_DIR [--keypoints KEYPOINTS.json] [--masks DIR] IMAGE...
A finger core is a capsule along the last three MediaPipe hand landmarks of each finger (PIP -> DIP -> TIP;
MCP -> IP -> TIP for the thumb), with radius
CORE_RADIUS x (wrist-to-middle-MCP distance). It sits inside the finger, so it gives finger recall, not IoU.
Prints one JSON line per finger and writes an overlay of the cores for a visual check.
Fingers whose core lies wholly outside the frame are skipped. Images with no detected hand are reported on stderr.

--masks DIR adds one more model, MASKS_MODEL: person probabilities read from DIR/<image stem>.png (8-bit, any size,
resized bilinearly to 640x480). bench/browser/litert_masks.html writes such masks from LiteRT.js.

Hands come from HandLandmarker, or from KEYPOINTS.json for images where a person placed the points (hard cases the
detector misses). An image listed there uses only its listed hands and fingers. The format, in original image pixels
(points may lie outside the image, for a hand cut off at the frame edge):
  {"<image file name>": [{"wrist": [x, y], "middle_mcp": [x, y],
                          "fingers": {"index": [[x, y], [x, y], [x, y]], ...}}, ...]}
Each finger lists its three points in the FINGERS order (PIP, DIP, TIP; MCP, IP, TIP for the thumb).
"""
import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision
from PIL import Image, ImageDraw

from cutout import SIZE, crop_box, to_640x480
from eval_person_iou import MODELS, segmenter

FINGERS = {"thumb": (2, 3, 4), "index": (6, 7, 8), "middle": (10, 11, 12), "ring": (14, 15, 16), "pinky": (18, 19, 20)}
CORE_RADIUS = 0.06
THRESHOLD = 0.5
KEPT = 0.5
MASKS_MODEL = "litertjs_multiclass"


def landmarker():
    options = vision.HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"), num_hands=2
    )
    return vision.HandLandmarker.create_from_options(options)


@dataclass(frozen=True)
class Hand:
    scale_px: float
    fingers: dict[str, list[tuple[float, float]]]
    source: str


def from_landmarks(landmarks):
    p = [(q.x * SIZE[0], q.y * SIZE[1]) for q in landmarks]
    return Hand(float(np.linalg.norm(np.subtract(p[0], p[9]))),
                {name: [p[i] for i in idx] for name, idx in FINGERS.items()}, "auto")


def from_keypoints(entry, image_size):
    x0, y0, x1, _ = crop_box(*image_size)
    k = SIZE[0] / (x1 - x0)

    def to_frame(pt):
        return (pt[0] - x0) * k, (pt[1] - y0) * k

    for name, pts in entry["fingers"].items():
        if name not in FINGERS or len(pts) != 3:
            raise ValueError(f"finger {name!r} needs one of {list(FINGERS)} with 3 points, got {pts}")
    return Hand(float(np.linalg.norm(np.subtract(to_frame(entry["wrist"]), to_frame(entry["middle_mcp"])))),
                {name: [to_frame(pt) for pt in pts] for name, pts in entry["fingers"].items()}, "manual")


def cores(hand):
    r = max(2, round(CORE_RADIUS * hand.scale_px))
    for name, pts in hand.fingers.items():
        canvas = Image.new("L", SIZE, 0)
        draw = ImageDraw.Draw(canvas)
        draw.line(pts, fill=255, width=2 * r)
        for x, y in pts:
            draw.ellipse((x - r, y - r, x + r, y + r), fill=255)
        yield name, pts, r, np.array(canvas) > 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir", type=Path)
    p.add_argument("images", type=Path, nargs="+")
    p.add_argument("--keypoints", type=Path)
    p.add_argument("--masks", type=Path)
    args = p.parse_args()
    keypoints = json.loads(args.keypoints.read_text()) if args.keypoints else {}
    args.out_dir.mkdir(parents=True, exist_ok=True)
    detector = landmarker()
    segs = {name: segmenter(name) for name in MODELS}
    for path in args.images:
        src = Image.open(path).convert("RGB")
        rgb = np.array(to_640x480(src))
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        probs = {
            name: np.squeeze(to_person([m.numpy_view() for m in segs[name].segment(image).confidence_masks]))
            for name, to_person in MODELS.items()
        }
        if args.masks:
            probs[MASKS_MODEL] = np.asarray(Image.open(args.masks / f"{path.stem}.png").convert("L").resize(SIZE, Image.BILINEAR)) / 255.0
        overlay = Image.fromarray(rgb)
        draw = ImageDraw.Draw(overlay)
        if path.name in keypoints:
            hands = [from_keypoints(entry, src.size) for entry in keypoints[path.name]]
        else:
            hands = [from_landmarks(lm) for lm in detector.detect(image).hand_landmarks]
        if not hands:
            print(f"{path.name}: no hand detected", file=sys.stderr)
        for i, hand in enumerate(hands):
            for finger, pts, r, core in cores(hand):
                if not core.any():
                    continue
                recall = {name: float((prob[core] >= THRESHOLD).mean()) for name, prob in probs.items()}
                kept = {name: v >= KEPT for name, v in recall.items()}
                print(json.dumps({"image": path.name, "hand": i, "source": hand.source, "finger": finger,
                                  "core_px": int(core.sum()), "recall": recall, "kept": kept}))
                draw.line(pts, fill=(255, 0, 0), width=1)
                for x, y in pts:
                    draw.ellipse((x - r, y - r, x + r, y + r), outline=(255, 255, 0))
        overlay.save(args.out_dir / f"{path.stem}.png")


if __name__ == "__main__":
    main()
