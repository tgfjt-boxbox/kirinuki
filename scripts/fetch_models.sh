#!/bin/bash
# Download the MediaPipe baseline segmenters and the hand landmarker into models/ and verify their SHA-256.
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

dest=models/hand_landmarker.task
[ -f "$dest" ] || curl -fsSL -o "$dest" https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task
echo "fbc2a30080c3c557093b5ddfc334698132eb341044ccee322ccf8bcf3607cde1  $dest" | shasum -a 256 -c -
