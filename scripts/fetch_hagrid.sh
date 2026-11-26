#!/bin/bash
# Fetch a fixed sample of one HaGRID 2.0 gesture class into data/hagrid/ for the finger-survival eval.
# The sample is the first N image names in sorted order. Names are UUIDs, so this is a reproducible random sample.
# Only the sampled entries are read from the remote zip (HTTP Range); the full class archive is about 40 GB.
# usage: scripts/fetch_hagrid.sh [CLASS] [N]   (default: peace 300)
set -euo pipefail
cd "$(dirname "$0")/.."
class="${1:-peace}" n="${2:-300}"
url="https://rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru/datasets/hagrid/hagrid_dataset_new_554800/hagrid_dataset/$class.zip"
list="data/hagrid/${class}_$n.txt"
mkdir -p data/hagrid
[ -f "$list" ] || python3 -I bench/remote_zip.py list "$url" "$class/*.jpg" | awk '{print $2}' | sort | head -n "$n" > "$list"
while read -r name; do [ -f "data/hagrid/$name" ] || echo "$name"; done < "$list" |
  xargs -r -n 10 -P 8 python3 -I bench/remote_zip.py get "$url" data/hagrid > /dev/null
echo "$(wc -l < "$list") listed, $(ls "data/hagrid/$class" | grep -c '\.jpg$') on disk in data/hagrid/$class"
