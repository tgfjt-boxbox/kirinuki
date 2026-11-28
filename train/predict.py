"""Write a trained student's person masks in the format bench/finger_core.py and bench/eval_person_iou.py read with --masks.

usage: .venv/bin/python -I train/predict.py RUN_DIR OUT_DIR hagrid LIST crop43|whole
       .venv/bin/python -I train/predict.py RUN_DIR OUT_DIR ep IDS_FILE whole
RUN_DIR holds student.pt and history.json from train/run.py (the width and input size come from history.json).
Framing matches bench/browser/litert_masks.html: crop43 is the 640x480 centre crop finger_core.py evaluates, whole is
the image as is. The frame is squashed to the model input and the mask is written at that size as <stem>.png (8-bit);
the evaluators resize it back bilinearly. Runs on the CPU, so it needs no agent/train-mps.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

KIRINUKI = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(KIRINUKI / "train"), str(KIRINUKI / "bench")]
from cutout import to_640x480  # noqa: E402
from run import Student  # noqa: E402

SOURCES = {
    "hagrid": lambda line: KIRINUKI / "data/hagrid_v1" / line,
    "ep": lambda line: KIRINUKI / "data/easyportrait/images" / f"{line}.jpg",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=Path)
    ap.add_argument("out_dir", type=Path)
    ap.add_argument("source", choices=SOURCES)
    ap.add_argument("list", type=Path)
    ap.add_argument("framing", choices=["crop43", "whole"])
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    train_args = json.loads((args.run_dir / "history.json").read_text())["args"]
    size = train_args["size"]
    model = Student(train_args["width"])
    model.load_state_dict(torch.load(args.run_dir / "student.pt", map_location="cpu"))
    model.eval()
    lines = args.list.read_text().split()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i in range(0, len(lines), args.batch):
        chunk = lines[i:i + args.batch]
        frames = []
        for line in chunk:
            img = Image.open(SOURCES[args.source](line)).convert("RGB")
            if args.framing == "crop43":
                img = to_640x480(img)
            frames.append(np.asarray(img.resize((size, size), Image.BILINEAR)))
        x = torch.from_numpy(np.stack(frames)).permute(0, 3, 1, 2).float() / 127.5 - 1
        with torch.no_grad():
            prob = torch.sigmoid(model(x))[:, 0].numpy()
        for line, p in zip(chunk, prob):
            Image.fromarray(np.round(p * 255).astype(np.uint8)).save(args.out_dir / f"{Path(line).stem}.png")
    run = {"run_dir": str(args.run_dir), "train_args": train_args, "source": args.source, "list": str(args.list),
           "framing": args.framing, "images": len(lines), "seconds": round(time.time() - t0, 1)}
    (args.out_dir / "masks_run.json").write_text(json.dumps(run, indent=1))
    print(json.dumps(run))


if __name__ == "__main__":
    main()
