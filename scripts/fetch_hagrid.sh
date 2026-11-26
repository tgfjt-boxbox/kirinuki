#!/bin/bash
# Fetch a fixed sample of one HaGRID v1 gesture class into data/hagrid_v1/ for the finger-survival eval.
# Source: a 384p sample of the HaGRID v1 train_val split on Hugging Face (cj-mills/hagrid-sample-30k-384p, about 1 GB).
# The v1 links in the Hagrid_v1 README do not answer, and HaGRID 2.0 is non-commercial per its paper (see docs/datasets.md).
# The sample is the first N image names in sorted order. Names are UUIDs, so this is a reproducible random sample.
# usage: scripts/fetch_hagrid.sh [CLASS] [N]   (default: peace 300)
set -euo pipefail
cd "$(dirname "$0")/.."
class="${1:-peace}" n="${2:-300}"
out="data/hagrid_v1"
zip="$out/hagrid-sample-30k-384p.zip"
list="$out/${class}_$n.txt"
mkdir -p "$out"
if [ ! -f "$zip" ]; then
  curl -fsSL --retry 5 -o "$zip.part" https://huggingface.co/datasets/cj-mills/hagrid-sample-30k-384p/resolve/main/hagrid-sample-30k-384p.zip
  mv "$zip.part" "$zip"
fi
if [ ! -f "$list" ]; then
  zipinfo -1 "$zip" "*/train_val_$class/*.jpg" | sort | sed -n "1,${n}p" > "$list.part"
  mv "$list.part" "$list"
fi
xargs unzip -qo "$zip" -d "$out" < "$list"
echo "$(wc -l < "$list" | tr -d ' ') listed, $(find "$out" -path "*/train_val_$class/*.jpg" | wc -l | tr -d ' ') on disk in $out"
