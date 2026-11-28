# 2段のズームの背景の誤り（#43）

EasyPortrait のテストの先頭 1,500枚（`head -1500 data/easyportrait/ids_all.txt`）を LiteRT.js（WebGPU、multiclass）で処理した。ズームが働いたのは 115枚。その 115枚で、1回の経路（`#ep/ids_head1500/whole/gpu`）と2段のズーム（`#…/zoom`）を比べた。

| | 1回 | 2段のズーム | 画像ごとの対の差 |
| --- | --- | --- | --- |
| 背景の誤り（背景の画素のうち人物とした割合、115枚の合計） | 0.706% | 0.812% | +0.18 ± 0.05 ポイント |
| 人物の IoU（115枚の合計） | 0.9786 | 0.9785 | |

- 115枚のすべてで背景の誤りが増え、減った画像はない。2回目の確率と `max` をとるので、構造上は増えるか同じかのどちらかになる
- 矩形の外では2段のズームは1回目と同じで、誤りの差は 0（ハーネスの確かめ）
- 4,000枚の全件は Chrome が20分の上限で止められ、途中で切った

再現:

```
python3 -I bench/browser/serve_latency.py OUT_zoom.json --mask-dir ZOOM_DIR    # と chrome-headless …/litert_masks.html#ep/ids_head1500/whole/zoom
python3 -I bench/browser/serve_latency.py OUT_gpu.json --mask-dir GPU_DIR      # と chrome-headless …/litert_masks.html#ep/ids_head1500/whole/gpu
.venv/bin/python -I bench/eval_zoom_background.py data/easyportrait GPU_DIR ZOOM_DIR OUT_zoom.json result.json
```
