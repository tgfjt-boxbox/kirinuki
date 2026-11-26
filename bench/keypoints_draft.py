"""Write a draft KEYPOINTS.json for finger_core.py from HandLandmarker, for a person to correct.

usage: .venv/bin/python bench/keypoints_draft.py IMAGE... > keypoints.json
Points are in original image pixels, in the format finger_core.py documents. Images with no detected hand get an
empty list, so the person sees which images still need every point placed.
"""
import json
import sys
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image

from cutout import SIZE, crop_box, to_640x480
from finger_core import FINGERS, landmarker


def to_original(q, box):
    x0, y0, x1, _ = box
    k = (x1 - x0) / SIZE[0]
    return [q.x * SIZE[0] * k + x0, q.y * SIZE[1] * k + y0]


def main():
    detector = landmarker()
    out = {}
    for path in map(Path, sys.argv[1:]):
        src = Image.open(path).convert("RGB")
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.array(to_640x480(src)))
        box = crop_box(*src.size)
        out[path.name] = [
            {"wrist": to_original(lm[0], box), "middle_mcp": to_original(lm[9], box),
             "fingers": {name: [to_original(lm[i], box) for i in idx] for name, idx in FINGERS.items()}}
            for lm in detector.detect(image).hand_landmarks
        ]
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
