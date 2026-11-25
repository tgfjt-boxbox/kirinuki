"""Summarize EasyPortrait samples and render an overlay contact sheet for visual review.

usage: .venv/bin/python bench/inspect_easyportrait.py SAMPLE_DIR OUT_PNG
SAMPLE_DIR holds ids.txt, images/<id>.jpg and test/<id>.png.
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

CLASSES = ["bg", "person", "skin", "lbrow", "rbrow", "leye", "reye", "lips", "teeth"]
THUMB = 256


def main():
    root, out = Path(sys.argv[1]), Path(sys.argv[2])
    ids = root.joinpath("ids.txt").read_text().split()
    tiles = []
    print("id\tWxH\tperson%\ttouches(L,R,B,T)\tclasses")
    for i in ids:
        img = Image.open(root / "images" / f"{i}.jpg").convert("RGB")
        mask = np.array(Image.open(root / "test" / f"{i}.png"))
        assert mask.shape == (img.height, img.width), f"{i}: mask {mask.shape} vs image {img.size}"
        person = mask > 0
        edges = [person[:, 0].any(), person[:, -1].any(), person[-1, :].any(), person[0, :].any()]
        present = [CLASSES[c] for c in np.unique(mask)]
        print(f"{i[:8]}\t{img.width}x{img.height}\t{100 * person.mean():.1f}\t"
              f"{''.join('x' if e else '.' for e in edges)}\t{','.join(present)}")
        scale = THUMB / max(img.size)
        size = (round(img.width * scale), round(img.height * scale))
        rgb = np.array(img.resize(size))
        m = np.array(Image.fromarray(person.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 0
        overlay = rgb.copy()
        overlay[~m] = (overlay[~m] * 0.25).astype(np.uint8)
        tile = Image.new("RGB", (THUMB, THUMB))
        tile.paste(Image.fromarray(overlay), ((THUMB - size[0]) // 2, (THUMB - size[1]) // 2))
        ImageDraw.Draw(tile).text((4, 4), i[:8], fill=(255, 255, 0))
        tiles.append(tile)
    cols = 6
    rows = -(-len(tiles) // cols)
    sheet = Image.new("RGB", (cols * THUMB, rows * THUMB))
    for k, t in enumerate(tiles):
        sheet.paste(t, ((k % cols) * THUMB, (k // cols) * THUMB))
    sheet.save(out)


if __name__ == "__main__":
    main()
