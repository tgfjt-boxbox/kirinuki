# 比較の基準にする MediaPipe モデル

2026-11-25 13:28 の定期実行で一次情報を確かめた。

## ライセンス

| モデル | ライセンス | 一次情報 |
| --- | --- | --- |
| `selfie_multiclass_256x256` | Apache License 2.0 | Google のモデルカード「Mediapipe Multiclass Segmentation」（2023-05-10、Adel Ahmadyan） |
| `selfie_segmenter_landscape` / `selfie_segmenter` | Apache License 2.0 | Google のモデルカード「MediaPipe Selfie Segmentation」（2021-05-06、Tingbo Hou ほか） |

- MediaPipe 公式ガイド（ai.google.dev の Image Segmenter、2026-11-20 更新）の「Model card」リンクは、Hugging Face の `litert-community/MediaPipe-ImageSegmenter-*` に移っている。各リポジトリに上流の Google のモデルカード PDF が置かれ、README の SHA-256 と取得した PDF の SHA-256 が一致した
- PDF の写しは `research/modelcards/`（multiclass.pdf、selfie.pdf）。Apache 2.0 なので、比較の基準として測る・推論結果を公開するのに支障はない（推論。条文の個別確認はしていない）

## モデルカードから分かること

| | selfie_segmenter（square / landscape） | selfie_multiclass_256x256 |
| --- | --- | --- |
| 構造 | MobileNetV3 系 CNN ＋ 独自デコーダ | Vision Transformer ＋ 独自ボトルネック・デコーダ |
| サイズ | 249KB / 244KB（float16） | 16.4MB（float32） |
| 入力 | 256×256×3 / 144×256×3、RGB [0,1] | 256×256×3（512×512 版もある） |
| 出力 | person の確率 1ch | 6クラス（background, hair, body-skin, face-skin, clothes, others） |
| 公表 IoU | person IoU 平均 96.48% / 95.99%（社内の自撮り写真 1,594枚、学習と同じ出所） | mIoU 77.23%（256 版）、81.10%（512 版）。918枚・1,902人。クラス平均と思われる |
| 人間の注釈者どうしの一致 | person IoU 98.74% | 記載なし |
| 公表された弱点 | 指などの細い部分が欠けることがある | 逆光・ノイズ・速い動き・大きな遮蔽で劣化 |
| Pixel 6 での推論（公式ガイド） | landscape: CPU 34.19ms / GPU 33.55ms | CPU 217.76ms / GPU 71.24ms |

## 比較の設計に効く点

- 公表値はどれも社内データで、公開テストセットの値ではない。同じ公開テストセットで測り直す価値がある（PLAN の「比較表を作ること自体に価値がある」を裏づける）
- multiclass は person クラスを直接出さない。person マスクは「background 以外の5クラスの和」（1 − background）として作る。この変換をハーネスで固定する
- selfie_segmenter の学習データは「同意を得た AR アプリの写真」で非公開。どの公開テストセットとも重なりは確かめられない
- selfie_segmenter のカードが「指が欠ける」と自ら書いている。手の IoU を下層指標に入れる判断を裏づける

## 精度の初回計測（2026-11-25 13:28 の実行）

精度はブラウザに依らないので、Python の `mediapipe` 1.2.3（CPU、`.venv/`）で測った。レイテンシは別にブラウザで測る。しきい値 0.5。モデルは `storage.googleapis.com/mediapipe-models` の latest（SHA-256 は `models/` のファイルで再計算できる）。

### person IoU: EasyPortrait テスト分割から24枚（seed 0）

`bench/eval_person_iou.py`、結果 `bench/results/2026-11-25-baseline/baseline_iou_ep24.json`

| モデル | 画像ごとの IoU の平均 | 全画素で集計した IoU | 最悪の画像 |
| --- | --- | --- | --- |
| selfie_multiclass_256x256 | 98.29% | 98.31% | 92.72% |
| selfie_segmenter（square） | 97.75% | 97.68% | 89.90% |
| selfie_segmenter_landscape | 97.26% | 97.01% | 84.40% |

- Boundary F（DAVIS の境界 F 値に近い定義。境界の膨張は正方形、許容幅は画像の対角の 0.8% を四捨五入。DAVIS 公式は円盤と切り上げなので、数値はそのままは比べられない）: multiclass 86.00%、square 84.26%、landscape 81.69%。IoU より伸びしろが大きい
- 全画素を person とする自明な予測は 49.81%。測定は本物
- multiclass − square は平均 +0.54 ポイント（標準誤差 0.30）、24枚中23枚で multiclass が上
- **伸びしろが小さい**: 基準がすでに 98% 付近にある。MediaPipe 自身のカードでは注釈者どうしの一致が 98.74%。EasyPortrait の mIoU だけをメイン指標にすると、改善が数字に出にくい（推論）

### 手の領域の IoU: Open Images テスト分割の150枚（seed 0）、手の bbox 257個

`bench/eval_hand_iou.py`、結果 `bench/results/2026-11-25-baseline/baseline_hand_iou_oi150.json`

| モデル | 箱ごとの IoU の平均 | 全画素で集計した IoU |
| --- | --- | --- |
| selfie_multiclass_256x256 | 80.10% | 80.17% |
| selfie_segmenter（square） | 71.53% | 64.04% |
| selfie_segmenter_landscape | 67.48% | 69.89% |

- 対応のある差: multiclass − square = +8.57 ポイント（標準誤差 1.78、154勝50敗）、multiclass − landscape = +12.63 ポイント（標準誤差 1.67）
- 実務で PC に multiclass を選んだ理由（手指の精度）と向きが一致する
- 場面の影響の切り分け（同じ実行で計測）: 画像に占める person マスクの面積比で箱を分けた

  | person の面積比 | 箱の数 | multiclass | square | landscape | multiclass − square（標準誤差） |
  | --- | --- | --- | --- | --- | --- |
  | 10% 未満 | 67 | 68.3% | 56.4% | 41.2% | +11.9（5.1） |
  | 10〜25% | 70 | 81.8% | 73.3% | 73.0% | +8.5（2.4） |
  | 25% 以上 | 120 | 85.7% | 79.0% | 78.9% | +6.8（2.1） |

  - 自撮りに近い「人物が大きい」場面でも multiclass − square は +6.8 ポイント残る。差の大部分は場面ではなく手の領域そのものから来ている
  - landscape は人物が小さい場面で大きく落ちる（41.2%）。これは場面の影響で、カードの「遠い人物は対象外」と一致する
  - 比較を自撮りの用途に寄せるなら、手の領域の IoU は「person の面積比 25% 以上」に絞って報告するのがよい（案）
- **この指標は伸びしろが大きく、モデルの差も見分けられる**

### 手の領域の IoU: 全件（Open Images テスト分割の617枚、手の bbox 1,065個）

`bench/eval_hand_iou.py data/openimages/hand_all.txt`、結果 `bench/results/2026-11-25-baseline/baseline_hand_iou_oi_all.json`

| モデル | 箱ごとの IoU の平均 | 全画素で集計した IoU |
| --- | --- | --- |
| selfie_multiclass_256x256 | 77.36% | 80.74% |
| selfie_segmenter（square） | 70.25% | 73.00% |
| selfie_segmenter_landscape | 66.05% | 74.93% |

- 対応のある差（標準誤差は同じ画像の箱をまとめて計算）: multiclass − square = +7.11 ポイント（1.00、599勝233敗）、multiclass − landscape = +11.31 ポイント（1.17）、square − landscape = +4.20 ポイント（0.86）
- person の面積比 25% 以上の456個: multiclass 84.57%、square 78.56%、landscape 77.31%。multiclass − square = +6.01 ポイント（標準誤差 0.87）
- 150枚の小標本と結論は同じ。順位は multiclass > square > landscape

### person IoU と Boundary F: EasyPortrait テスト分割から800枚（seed 1）

`bench/eval_person_iou.py data/easyportrait`、結果 `bench/results/2026-11-25-baseline/baseline_iou_ep800.json`。所要 574秒

| モデル | 画像ごとの IoU の平均 | 全画素で集計した IoU | 最悪の画像 | IoU < 90% の枚数 | Boundary F |
| --- | --- | --- | --- | --- | --- |
| selfie_multiclass_256x256 | 98.11% | 98.16% | 70.07% | 10 | 87.10% |
| selfie_segmenter（square） | 97.18% | 97.31% | 49.50% | 23 | 83.52% |
| selfie_segmenter_landscape | 96.80% | 96.89% | 69.20% | 22 | 81.59% |

- 対応のある差: IoU は multiclass − square = +0.93 ポイント（標準誤差 0.11）。Boundary F は +3.58 ポイント（0.26）
- Boundary F は IoU の約4倍の差でモデルを分ける。EasyPortrait を使うなら、IoU より Boundary F の方が改善を見やすい（推論）
- 平均は高いが、IoU が 90% を下回る失敗画像が 10〜23枚ある。失敗の中身はまだ見ていない

## 背景の透け（#5、2026-11-25 23:10 の実行）

確信度をそのまま透明度に使って背景を差し替えると、multiclass では元の部屋が透ける。`bench/eval_soft_alpha.py data/easyportrait`（EasyPortrait テスト分割の800枚、seed 1）、結果 `bench/results/2026-11-25-soft-alpha/soft_alpha_ep800.json`。

- 透け（bleed）は、正解の背景の画素での透明度の平均。穴（hole）は、正解の人物の画素での 1 − 透明度の平均。どちらも全画素で集計した

| モデル | 透け: そのまま | 透け: smoothstep(0.2, 0.8) | 穴: そのまま | 穴: smoothstep(0.2, 0.8) |
| --- | --- | --- | --- | --- |
| selfie_multiclass_256x256 | 9.54% | 1.17% | 2.58% | 0.64% |
| selfie_segmenter（square） | 1.99% | 1.78% | 1.10% | 0.87% |
| selfie_segmenter_landscape | 2.13% | 1.86% | 1.49% | 1.20% |

- multiclass は、背景の画素の 95.8% で確信度が 0.05〜0.1 にある。中身を見ると、背景の画素の多くで background クラスが約 0.917 にとどまり、他の5クラスが約 0.0167 ずつ残る。だから 1 − background は、背景のほとんどの画素で約 0.083 になる。最も下がる画素でも 0.04 程度（24枚で確認）
- smoothstep(0.2, 0.8) をかけると、multiclass の透けは 9.54% から 1.17% に下がり、3モデルで最も小さくなる。しきい値 0.5 で2値化した場合（1.12%）とほぼ同じで、輪郭はなめらかなまま
- しきい値 0.5 での IoU と Boundary F は、このカーブで変わらない（カーブは 0.5 を 0.5 に写す）。変わるのは合成の見た目
- `bench/cutout.py` の合成は、このカーブを使うように変えた

![そのままの合成（左）と smoothstep をかけた合成（右）](img/soft-alpha-office-peace.png)

画像は、この変更の前後の `bench/cutout.py` で `office-peace.png` を処理し、multiclass の行の合成の列を左右に並べて縮小したもの。

## 指の生存率: HaGRID 2.0 `peace` の300枚（#12、2026-11-26 09:14 の実行）

`scripts/fetch_hagrid.sh peace 300` で取った300枚（名前順の先頭。画像は git の外）。`finger_core.py` と `finger_summary.py` で測った。行ごとの結果は `bench/results/2026-11-26-hagrid-peace300/`。

- HandLandmarker が手を見つけたのは271枚、指は1,479本。29枚は手が見つからず、評価から外れた
- 重ね合わせ画像を目で確かめた。最初の12枚と、multiclass が立てた指を消した9枚で、指の芯はどれも指の上にあった

| | multiclass | selfie_segmenter（square） | landscape |
| --- | --- | --- | --- |
| 指の生存率 | **0.897** | 0.652 | 0.674 |
| 標準誤差（画像単位のブートストラップ、2,000回） | 0.013 | 0.024 | 0.022 |
| 指の芯の再現率 | 0.903 | 0.742 | 0.695 |
| 人差し指 | 0.831 | 0.547 | 0.584 |
| 中指 | 0.780 | 0.475 | 0.515 |
| 薬指 / 小指 / 親指 | 0.96 / 0.95 / 0.97 | 0.74 / 0.74 / 0.76 | 0.76 / 0.74 / 0.77 |

- 消えるのは、立てた2本（人差し指と中指）。曲げた指は手のひらの上に乗るので残りやすい。multiclass でも、立てた指を1本以上消した画像が 71 / 270枚ある
- 手が小さいほど消える。立てた指591本を指の芯の画素数で3等分すると、multiclass の生存率は小 0.72、中 0.84、大 0.85（square は 0.35 / 0.53 / 0.66）。ただし大きい手でも 15% は消えるので、解像度だけが原因ではない
- `finger_summary.py` の標準誤差（multiclass 0.008）は指を独立とみなすので、小さく出すぎる。同じ画像の指はまとめて消えやすい。上の表は画像単位で計算し直した値
- 伸びしろ: 基準の multiclass で 0.897。メイン指標として十分に差が出る
