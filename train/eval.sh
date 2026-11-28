#!/bin/bash
# Evaluate a train/run.py student on the frozen benchmark: finger survival on HaGRID v1 peace_300 (crop43), and with
# a third argument `ep`, person IoU on all 4,000 EasyPortrait test images (whole frame, about 15 minutes).
# usage: train/eval.sh RUN_DIR OUT_DIR [ep]     (run from kirinuki/)
# Writes OUT_DIR/summary_peace300.json (the student is the litertjs_multiclass row, the name --masks uses) and
# OUT_DIR/ep4000.json; masks and overlays go to OUT_DIR/work/.
set -eu
RUN=$1
OUT=$2
W=$OUT/work
mkdir -p "$W"
export MPLCONFIGDIR=${TMPDIR:-/tmp}/mpl PYTHONPATH=bench
PY=.venv/bin/python

$PY -I train/predict.py "$RUN" "$W/peace" hagrid data/hagrid_v1/peace_300.txt crop43 > /dev/null
$PY bench/finger_core.py "$W/overlay" --masks "$W/peace" $(sed 's|^|data/hagrid_v1/|' data/hagrid_v1/peace_300.txt) \
  > "$OUT/fingers_peace300.jsonl" 2> "$W/fingers.err"
$PY bench/finger_summary.py "$OUT/fingers_peace300.jsonl" > "$OUT/summary_peace300.json"
$PY -c 'import json, sys; m = json.load(open(sys.argv[1]))["models"]; s = m["litertjs_multiclass"]; b = m["selfie_multiclass_256x256"]
print("student survival %.4f (se %.4f, %d fingers); multiclass %.4f" % (s["survival"], s["survival_se"], s["fingers"], b["survival"]))' "$OUT/summary_peace300.json"

if [ "${3-}" = ep ]; then
  $PY -I train/predict.py "$RUN" "$W/ep" ep data/easyportrait/ids_all.txt whole > /dev/null
  mkdir -p "$W/ep_root"
  ln -sfn "$PWD/data/easyportrait/images" "$W/ep_root/images"
  ln -sfn "$PWD/data/easyportrait/test" "$W/ep_root/test"
  cp data/easyportrait/ids_all.txt "$W/ep_root/ids.txt"
  $PY bench/eval_person_iou.py "$W/ep_root" "$OUT/ep4000.json" --models --masks "$W/ep" > /dev/null
  $PY -c 'import json, sys; s = json.load(open(sys.argv[1]))["models"]["litertjs_multiclass"]
print("student EP4000 mean image IoU %.4f, dataset IoU %.4f" % (s["mean_image_iou"], s["dataset_iou"]))' "$OUT/ep4000.json"
fi
