"""Compare two benchmark runs as one self-contained HTML page (paired per-image diffs and error maps).

usage: .venv/bin/python bench/compare.py BEFORE.json:MODEL AFTER.json:MODEL --data DATA_DIR OUT_HTML [--top 6]
         [--fingers BEFORE.jsonl:MODEL AFTER.jsonl:MODEL] [--latency BEFORE.json AFTER.json]
A run is a result JSON from eval_person_iou.py plus a model name in it.
--fingers adds finger survival (docs/finger-metric.md) from finger_core.py output, paired per finger
(image, hand, finger); the paired SE treats the image as the sampling unit, as in finger_summary.py.
--latency adds the per-frame time of each run's model from bench/browser/latency.html results (pass the same file twice
when one browser run measured both models). The pass gate (docs/finger-metric.md) is GPU_TEXTURE at p95.
Only the top-K improved and top-K regressed images (by person_iou) are re-segmented to draw error maps:
red = kept but not person (pred & ~gt), blue = person cut away (gt & ~pred), the rest is the dimmed input.
DATA_DIR holds images/<id>.jpg and test/<id>.png (every class > 0 is person).
"""
import argparse
import base64
import html
import io
import json
from collections import defaultdict
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
    if model not in data["models"]:
        raise ValueError(f"{model} not in {path}: {', '.join(data['models'])}")
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


def load_fingers(ref):
    path, model = ref.rsplit(":", 1)
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not rows or model not in rows[0]["kept"]:
        raise ValueError(f"{model} not in {path}")
    return ref, {(r["image"], r["hand"], r["finger"]): (r["kept"][model], r["recall"][model], r["core_px"]) for r in rows}


def clustered_paired_se(keys, d):
    per_image = defaultdict(lambda: [0.0, 0])
    for k, v in zip(keys, d):
        per_image[k[0]][0] += v
        per_image[k[0]][1] += 1
    g = len(per_image)
    if g < 2:
        return float("nan")
    mean = d.mean()
    resid = sum((s - mean * n) ** 2 for s, n in per_image.values())
    return np.sqrt(g / (g - 1) * resid) / len(d)


def finger_section(before, after):
    (bref, b), (aref, a) = before, after
    keys = sorted(b.keys() & a.keys())
    kb = np.array([b[k][0] for k in keys], dtype=float)
    ka = np.array([a[k][0] for k in keys], dtype=float)
    pb = np.array([b[k][2] for k in keys], dtype=float)
    pa = np.array([a[k][2] for k in keys], dtype=float)
    rb = np.array([b[k][1] for k in keys])
    ra = np.array([a[k][1] for k in keys])
    rows = [("finger survival (all)", keys, kb, ka)]
    for f in ("thumb", "index", "middle", "ring", "pinky"):
        sel = [i for i, k in enumerate(keys) if k[2] == f]
        if sel:
            rows.append((f"finger survival ({f})", [keys[i] for i in sel], kb[sel], ka[sel]))
    cells = []
    for name, ks, vb, va in rows:
        d = va - vb
        cells.append([name, f"{vb.mean():.2%}", f"{va.mean():.2%}", f"{d.mean() * 100:+.2f}",
                      f"{clustered_paired_se(ks, d) * 100:.2f}", f"{int((d > 0).sum())} / {int((d < 0).sum())}", str(len(d))])
    pooled_b, pooled_a = (rb * pb).sum() / pb.sum(), (ra * pa).sum() / pa.sum()
    cells.append(["finger-core recall (pooled)", f"{pooled_b:.2%}", f"{pooled_a:.2%}",
                  f"{(pooled_a - pooled_b) * 100:+.2f}", "", "", str(len(keys))])
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in cells)
    head = ("<tr><th>metric</th><th>before</th><th>after</th><th>diff (pt)</th><th>paired SE (pt, by image)</th>"
            "<th>kept now / lost now</th><th>fingers</th></tr>")
    images = len({k[0] for k in keys})
    return (f"<h2>Fingers</h2><p><code>{html.escape(bref)}</code> → <code>{html.escape(aref)}</code>, "
            f"{images} images. Unmatched fingers: {len(b.keys() ^ a.keys())}.</p><table>{head}{body}</table>")


GATE_MODE, GATE_MS = "GPU_TEXTURE", 8.0


def latency_rows(path, model):
    data = json.loads(Path(path).read_text())
    rows = {(r["mode"], r.get("input", "static")): r["with_readback_ms"] for r in data["results"] if r["model"] == model}
    if not rows:
        raise ValueError(f"{model} not in {path}")
    return rows


def latency_section(paths, before, after):
    b, a = latency_rows(paths[0], before.model), latency_rows(paths[1], after.model)
    keys = [k for k in b if k in a]
    gate = lambda ms: "pass" if ms <= GATE_MS else "fail"
    cells = []
    for mode, inp in keys:
        vb, va = b[(mode, inp)], a[(mode, inp)]
        verdict = f"{gate(vb['p95'])} → {gate(va['p95'])}" if mode == GATE_MODE else ""
        cells.append([mode, inp, f"{vb['p50']:g} / {vb['p95']:g}", f"{va['p50']:g} / {va['p95']:g}",
                      f"{va['p95'] - vb['p95']:+.1f}", verdict])
    body = "".join("<tr>" + "".join(f"<td>{c}</td>" for c in row) + "</tr>" for row in cells)
    head = ("<tr><th>mode</th><th>input</th><th>before p50 / p95 (ms)</th><th>after p50 / p95 (ms)</th>"
            f"<th>p95 diff (ms)</th><th>gate ({GATE_MODE} p95 ≤ {GATE_MS:g} ms)</th></tr>")
    src = " → ".join(f"<code>{html.escape(str(p))}</code>" for p in paths)
    return f"<h2>Latency</h2><p>{src}</p><table>{head}{body}</table>"


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
    p.add_argument("--fingers", nargs=2, type=load_fingers, metavar=("BEFORE.jsonl:MODEL", "AFTER.jsonl:MODEL"))
    p.add_argument("--latency", nargs=2, type=Path, metavar=("BEFORE.json", "AFTER.json"))
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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(f"""<!doctype html><meta charset="utf-8"><title>compare</title>
<style>body{{font-family:sans-serif;margin:2em}}table{{border-collapse:collapse;margin-bottom:2em}}
td,th{{border:1px solid #ccc;padding:4px 8px;vertical-align:top}}img{{display:block}}</style>
<h1>{refs}</h1>
<p>Error maps: red = kept but not person, blue = person cut away. Threshold {before.threshold} / {after.threshold}.</p>
<table>{head}{table}</table>
{finger_section(*args.fingers) if args.fingers else ""}
{latency_section(args.latency, before, after) if args.latency else ""}
{"".join(sections)}
""")


if __name__ == "__main__":
    main()
