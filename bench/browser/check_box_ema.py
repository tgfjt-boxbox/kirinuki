"""Check that the boxEma boxes a litert.html run recorded follow the same recurrence as bench/flicker.py.

usage: python3 bench/browser/check_box_ema.py OUT.json  (OUT.json from serve_latency.py)
Exits 1 if any box differs from the recurrence, or if no input moved the box (then the run proves nothing).
Each trace starts from its first recorded box: the page reuses one segmenter across inputs, so history carries over.
"""
import json
import sys

out = json.load(open(sys.argv[1]))
a = out["box_ema"]
worst, moved_total = 0.0, 0
for name, trace in out["ema_boxes"].items():
    err, moved, prev = 0.0, 0, trace[0]["box"]
    for f in trace[1:]:
        want = f["raw"] if prev is None or f["raw"] is None else [a * r + (1 - a) * p for r, p in zip(f["raw"], prev)]
        if want is None:
            err = max(err, 0.0 if f["box"] is None else 1.0)
        else:
            err = max(err, max(abs(x - y) for x, y in zip(want, f["box"])))
            moved += want != f["raw"]
        prev = f["box"]
    print(f"{name}: {len(trace)} frames, EMA changed the box in {moved}, max |box - recurrence| {err:.2e}")
    worst, moved_total = max(worst, err), moved_total + moved
sys.exit(0 if worst < 1e-9 and moved_total else 1)
