"""Temporal flicker of the person mask on synthetic slow-motion clips, with and without the two-pass zoom.

Each clip pans a 640x480 window over a HaGRID still upscaled by SCALE, STEP pixels per frame on both axes. Consecutive
masks are compared in source coordinates on their overlap: flicker = |A xor B| / |A or B| of the masks thresholded at 0.5.
outside = share of the final person mask that lies outside the zoom box. That part keeps only the first pass. It is 0
without EMA and measures how much of a moving person a lagging smoothed box leaves out of the second pass.
The zoom path ports zoomBox and zoomedAt from browser/litert_gpu.js; the model runs on MediaPipe (CPU), not LiteRT.js.
"""
import argparse
import json
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image

from cutout import to_640x480
from eval_person_iou import MODELS, THRESHOLD, segmenter

MODEL = "selfie_multiclass_256x256"
SIZE = 256
W, H = 640, 480


def person_prob(seg, rgb):
    out = seg.segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))).confidence_masks
    return np.squeeze(MODELS[MODEL]([m.numpy_view() for m in out])).astype(np.float32)


def zoom_box(person, margin, max_area):
    ys, xs = np.nonzero(person >= THRESHOLD)
    if len(xs) == 0:
        return None
    l, r, t, b = xs.min() / SIZE, (xs.max() + 1) / SIZE, ys.min() / SIZE, (ys.max() + 1) / SIZE
    mx, my = margin * (r - l), margin * (b - t)
    l, r, t, b = max(0, l - mx), min(1, r + mx), max(0, t - my), min(1, b + my)
    side = min(1, max(r - l, b - t))
    if side * side > max_area:
        return None
    cx = min(max((l + r) / 2, side / 2), 1 - side / 2)
    cy = min(max((t + b) / 2, side / 2), 1 - side / 2)
    return np.array([cx - side / 2, cy - side / 2, side, side])


def bilinear(m, u, v):
    fx = np.clip(u * SIZE - 0.5, 0, SIZE - 1)
    fy = np.clip(v * SIZE - 0.5, 0, SIZE - 1)
    x0, y0 = np.floor(fx).astype(int), np.floor(fy).astype(int)
    x1, y1 = np.minimum(x0 + 1, SIZE - 1), np.minimum(y0 + 1, SIZE - 1)
    wx, wy = fx - x0, fy - y0
    return (m[y0, x0] * (1 - wx) + m[y0, x1] * wx) * (1 - wy) + (m[y1, x0] * (1 - wx) + m[y1, x1] * wx) * wy


U, V = np.meshgrid((np.arange(W) + 0.5) / W, (np.arange(H) + 0.5) / H)


def frame_mask(seg, frame, zoom, margin, max_area, prev_box=None, ema=1.0):
    small = np.array(Image.fromarray(frame).resize((SIZE, SIZE), Image.BILINEAR))
    first = person_prob(seg, small)
    p = bilinear(first, U, V)
    box = zoom_box(first, margin, max_area) if zoom else None
    if box is not None and prev_box is not None:
        box = ema * box + (1 - ema) * prev_box
    outside = 0.0
    if box is not None:
        x, y, w, h = box
        crop = Image.fromarray(frame).transform((SIZE, SIZE), Image.EXTENT, (x * W, y * H, (x + w) * W, (y + h) * H), Image.BILINEAR)
        second = person_prob(seg, np.array(crop))
        su, sv = (U - x) / w, (V - y) / h
        inside = (su >= 0) & (su <= 1) & (sv >= 0) & (sv <= 1)
        p = np.where(inside, np.maximum(p, bilinear(second, su, sv)), p)
        person = p >= THRESHOLD
        outside = float((person & ~inside).sum() / max(person.sum(), 1))
    return p >= THRESHOLD, box, outside


def clip_flicker(seg, src, offsets, zoom, margin, max_area, ema=1.0):
    masks, boxes, outside = [], [], []
    for ox, oy in offsets:
        m, box, out = frame_mask(seg, src[oy:oy + H, ox:ox + W], zoom, margin, max_area, boxes[-1] if boxes else None, ema)
        masks.append(m)
        boxes.append(box)
        outside.append(out)
    values = []
    for (ax, ay), (bx, by), a, b in zip(offsets, offsets[1:], masks, masks[1:]):
        dx, dy = bx - ax, by - ay
        a = a[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)]
        b = b[max(-dy, 0):H + min(-dy, 0), max(-dx, 0):W + min(-dx, 0)]
        union = (a | b).sum()
        values.append((a ^ b).sum() / union if union else 0.0)
    return float(np.mean(values)), float(np.max(values)), sum(b is not None for b in boxes), max(outside)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--list", default="data/hagrid_v1/peace_300.txt")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--frames", type=int, default=24)
    ap.add_argument("--step", type=int, default=1)
    ap.add_argument("--scale", type=float, default=1.1)
    ap.add_argument("--margin", type=float, default=0.1)
    ap.add_argument("--max-area", type=float, default=0.95)
    ap.add_argument("--box-ema", type=float, default=1.0, help="weight of the new box; 1 keeps no history")
    args = ap.parse_args()

    seg = segmenter(MODEL)
    pan = [(i * args.step, i * args.step) for i in range(args.frames)]
    static = [(0, 0)] * 3
    rows = []
    for line in Path(args.list).read_text().split()[:args.limit]:
        base = to_640x480(Image.open(Path(args.list).parent / line).convert("RGB"))
        src = np.array(base.resize((round(W * args.scale), round(H * args.scale)), Image.BILINEAR))
        assert (args.frames - 1) * args.step + H <= src.shape[0], "pan leaves the source"
        row = {"image": line}
        for name, zoom in (("whole", False), ("zoom", True)):
            row[name], row[f"{name}_max"], row[f"{name}_zoomed_frames"], row[f"{name}_outside_max"] = clip_flicker(
                seg, src, pan, zoom, args.margin, args.max_area, args.box_ema)
            row[f"{name}_static"], _, _, _ = clip_flicker(seg, src, static, zoom, args.margin, args.max_area, args.box_ema)
        rows.append(row)

    whole = np.array([r["whole"] for r in rows])
    zoom = np.array([r["zoom"] for r in rows])
    d = zoom - whole
    zoomed = np.array([r["zoom_zoomed_frames"] for r in rows])
    summary = {
        "clips": len(rows),
        "frames": args.frames,
        "step_px": args.step,
        "whole_mean": float(whole.mean()),
        "zoom_mean": float(zoom.mean()),
        "diff_mean": float(d.mean()),
        "diff_se": float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else None,
        "zoom_worse_clips": int((d > 0).sum()),
        "whole_max_pair_mean": float(np.mean([r["whole_max"] for r in rows])),
        "zoom_max_pair_mean": float(np.mean([r["zoom_max"] for r in rows])),
        "clips_with_zoom": int((zoomed > 0).sum()),
        "diff_mean_clips_with_zoom": float(d[zoomed > 0].mean()) if (zoomed > 0).any() else None,
        "clips_zoom_toggling": int(((zoomed > 0) & (zoomed < args.frames)).sum()),
        "zoom_outside_max_mean": float(np.mean([r["zoom_outside_max"] for r in rows])),
        "zoom_outside_max": float(max(r["zoom_outside_max"] for r in rows)),
        "clips_person_outside_box": int(sum(r["zoom_outside_max"] > 0 for r in rows)),
        "static_max": float(max(max(r["whole_static"], r["zoom_static"]) for r in rows)),
    }
    Path(args.out).write_text(json.dumps({"args": vars(args), "summary": summary, "rows": rows}, indent=1))
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
