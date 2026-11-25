#!/bin/bash
# Download the MediaPipe baseline segmenters (Apache 2.0) into models/ and verify their SHA-256.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p models

BASE=https://storage.googleapis.com/mediapipe-models/image_segmenter
while read -r name precision sha; do
  dest="models/$name.tflite"
  [ -f "$dest" ] || curl -fsSL -o "$dest" "$BASE/$name/$precision/latest/$name.tflite"
  echo "$sha  $dest" | shasum -a 256 -c -
done <<'EOF'
selfie_multiclass_256x256 float32 c6748b1253a99067ef71f7e26ca71096cd449baefa8f101900ea23016507e0e0
selfie_segmenter float16 191ac9529ae506ee0beefa6b2c945a172dab9d07d1e802a290a4e4038226658b
selfie_segmenter_landscape float16 490e9ea734313e0de10fa0cd9e3c6133e36ea4db2b7a49bde9ef019f72796b8e
EOF
