# kirinuki

Browser person segmentation for video calls, measured on our own benchmark.

ビデオ通話の背景差し替えのための、ブラウザで動く人物切り抜き。自分たちで作ったベンチマークで測りながら改善する。

## 解きたい問題

ブラウザでの人物切り抜きは、MediaPipe の `selfie_multiclass_256x256` が実用上の基準になっている。公開データ（EasyPortrait）では人物 IoU が 98% を超える。それでも、実際のウェブカメラ映像では**指が消える**。肌と背景の色が近い部屋でピースサインをすると、multiclass でも指が半透明になり、背景が透ける。

このリポジトリの目標は、ブラウザで 24fps（640×480）を保ったまま、指を消さないこと。

## 進め方

1. 改善の前に、ベンチマークを作る。場面ごとにタグを付けた素材と、前後を比べられる結果を残す。設計は [docs/benchmark.md](docs/benchmark.md)
2. 基準モデルを同じ条件で測る。初回の結果は [docs/baselines.md](docs/baselines.md)
3. 改善は1つずつ PR にし、PR にベンチマークの前後を載せる

作業は Claude（AI エージェント）が自律的に行い、Issue と PR に記録している。

## 使い方

```sh
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
scripts/fetch_models.sh

# 比較画像（入力、確信度マスク、重ね合わせ、背景差し替え）をモデルごとに並べる
PYTHONPATH=bench .venv/bin/python bench/cutout.py out/ bench/materials/public/office-peace.png

# 指が残るかどうか（指の芯の再現率）を指ごとに測る。定義は docs/finger-metric.md
PYTHONPATH=bench .venv/bin/python bench/finger_core.py out/ bench/materials/public/office-peace.png
```

データセットの取得と評価の方法は [docs/datasets.md](docs/datasets.md) と各スクリプトの冒頭にある。

## ライセンス

- コード: MIT（[LICENSE](LICENSE)）
- 学習したモデルの重み（今後公開する）: CC BY-SA 4.0。学習に使う EasyPortrait のライセンス条件による
- 基準モデル（MediaPipe）: Apache 2.0。`scripts/fetch_models.sh` で Google の配布元から取得する
- `bench/materials/public/office-peace.png`: AI 生成画像（ChatGPT）
