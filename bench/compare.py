"""Compare two benchmark runs as one self-contained HTML page (paired per-image diffs and error maps).

usage: .venv/bin/python bench/compare.py BEFORE.json:MODEL AFTER.json:MODEL --data DATA_DIR OUT_HTML [--top 6]
A run is a result JSON from eval_person_iou.py plus a model name in it.
Only the top-K improved and top-K regressed images (by person_iou) are re-segmented to draw error maps:
red = kept but not person (pred & ~gt), blue = person cut away (gt & ~pred), the rest is the dimmed input.
DATA_DIR holds images/<id>.jpg and test/<id>.png (every class > 0 is person).
"""
import argparse
import base64
import html
import io
import json
from dataclasses import dataclass
from pathlib import Path

import mediapipe as mp
import numpy as np
from PIL import Image

from eval_person_iou import MODELS, segmenter

METRICS = {"person_iou": "per_image", "boundary_f": "per_image_boundary_f"}
RANK_BY = "person_iou"
THUMB_WIDTH = 320


@dataclass(frozen=True)
class Run:
    ref: str
    model: str
    threshold: float
    scores: dict[str, dict[str, float]]


def load_run(ref):
    path, model = ref.rsplit(":", 1)
    data = json.loads(Path(path).read_text())
    per_model = data["models"][model]
    return Run(ref, model, data["threshold"], {m: per_model[key] for m, key in METRICS.items()})


def paired(before, after, metric):
    ids = sorted(before.scores[metric].keys() & after.scores[metric].keys())
    b = np.array([before.scores[metric][i] for i in ids])
    a = np.array([after.scores[metric][i] for i in ids])
    return ids, b, a


def summary_row(metric, b, a):
    d = a - b
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    cells = [metric, f"{b.mean():.2%}", f"{a.mean():.2%}", f"{d.mean() * 100:+.2f}", f"{se * 100:.2f}",
             f"{(d > 0).sum()} / {(d < 0).sum()}", str(len(d))]
    return "<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>"


def data_uri(rgb):
    img = Image.fromarray(rgb)
    img = img.resize((THUMB_WIDTH, round(img.height * THUMB_WIDTH / img.width)), Image.BOX)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def error_map(rgb, pred, gt):
    out = (rgb * 0.35).astype(np.uint8)
    out[pred & ~gt] = (255, 0, 0)
    out[gt & ~pred] = (0, 80, 255)
    return out


class Segmenters:
    def __init__(self):
        self.cache = {}

    def predict(self, run, rgb):
        if run.model not in self.cache:
            self.cache[run.model] = segmenter(run.model)
        res = self.cache[run.model].segment(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        prob = np.squeeze(MODELS[run.model]([m.numpy_view() for m in res.confidence_masks]))
        return prob >= run.threshold

    def close(self):
        for seg in self.cache.values():
            seg.close()


def image_rows(ids, b, a, before, after, data, segs):
    rows = []
    for i, vb, va in zip(ids, b, a):
        rgb = np.array(Image.open(data / "images" / f"{i}.jpg").convert("RGB"))
        gt = np.array(Image.open(data / "test" / f"{i}.png")) > 0
        preds = [segs.predict(run, rgb) for run in (before, after)]
        stale = [run.ref for run, pred, v in zip((before, after), preds, (vb, va))
                 if abs((pred & gt).sum() / (pred | gt).sum() - v) > 1e-4]
        note = "".join(f"<br><b>map differs from JSON: {html.escape(r)}</b>" for r in stale)
        imgs = "".join(f'<td><img src="{data_uri(x)}"></td>' for x in [rgb, *(error_map(rgb, p, gt) for p in preds)])
        rows.append(f"<tr><td><code>{html.escape(i)}</code><br>{vb:.2%} → {va:.2%} ({(va - vb) * 100:+.2f}){note}</td>{imgs}</tr>")
    return "".join(rows)


def section(title, rows):
    head = f"<tr><th>id / {RANK_BY}</th><th>input</th><th>before</th><th>after</th></tr>"
    body = rows or '<tr><td colspan="4">none</td></tr>'
    return f"<h2>{title}</h2><table>{head}{body}</table>"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("before", type=load_run, metavar="BEFORE.json:MODEL")
    p.add_argument("after", type=load_run, metavar="AFTER.json:MODEL")
    p.add_argument("out", type=Path)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--top", type=int, default=6)
    args = p.parse_args()
    before, after = args.before, args.after

    table = "".join(summary_row(m, *paired(before, after, m)[1:]) for m in METRICS)
    ids, b, a = paired(before, after, RANK_BY)
    order = np.argsort(a - b, kind="stable")
    up = [k for k in order[::-1] if a[k] > b[k]][:args.top]
    down = [k for k in order if a[k] < b[k]][:args.top]

    segs = Segmenters()
    try:
        sections = [section(title, image_rows([ids[k] for k in ks], b[ks], a[ks], before, after, args.data, segs))
                    for title, ks in (("Largest improvements", up), ("Largest regressions", down))]
    finally:
        segs.close()

    refs = f"<code>{html.escape(before.ref)}</code> → <code>{html.escape(after.ref)}</code>"
    head = ("<tr><th>metric</th><th>before</th><th>after</th><th>diff (pt)</th><th>paired SE (pt)</th>"
            "<th>better / worse</th><th>n</th></tr>")
    args.out.write_text(f"""<!doctype html><meta charset="utf-8"><title>compare</title>
<style>body{{font-family:sans-serif;margin:2em}}table{{border-collapse:collapse;margin-bottom:2em}}
td,th{{border:1px solid #ccc;padding:4px 8px;vertical-align:top}}img{{display:block}}</style>
<h1>{refs}</h1>
<p>Error maps: red = kept but not person, blue = person cut away. Threshold {before.threshold} / {after.threshold}.</p>
<table>{head}{table}</table>
{"".join(sections)}
""")


if __name__ == "__main__":
    main()
