# 人物セグメンテーション用データセット候補（2026-11-25 調査）

目的: 学習と評価に使えるデータセットを、ライセンス付きで洗い出す。
区分:
- **学習可**: 公開するモデルの学習に使ってよい（商用利用の禁止がない）
- **評価のみ可**: 研究・非商用に限られる、またはライセンスがはっきりしない。手元での評価や比較だけに使い、学習には入れない。評価結果を公開してよいかは個別に確かめる
- **使わない**

「確認元」が一次情報（配布元の README や規約の PDF）のものは ✅、二次情報（ミラー・検索結果）だけのものは ⚠️ とする。

## 候補表

| データセット | 中身 | 規模 | ライセンス | 区分 | 確認元 |
| --- | --- | --- | --- | --- | --- |
| **EasyPortrait** | ビデオ会議を想定した室内のポートレート。person・skin・眉・目・唇・歯など9クラス | v1: 2万枚（学習1.4万/検証0.2万/テスト0.4万）、v3: 4万枚 | CC BY-SA 4.0 を書き直した独自ライセンス（"not a Creative Commons license"）。商用の禁止はない。**派生物（Adapted Material）は BY-SA 互換のライセンスで出す必要がある** | 学習可（条件付き。下の論点1） | ✅ [license/en_us.pdf](https://github.com/hukenovs/easyportrait/blob/master/license/en_us.pdf) |
| **Open Images V7**（Person と Human hand などのマスク） | 一般写真。インスタンスマスクは 350 クラス・280万件 | 大きい | アノテーションは Google が CC BY 4.0。画像は CC BY 2.0 として登録（1枚ずつの確認は利用者の責任） | 学習可 | ✅ [factsfigures_v7](https://storage.googleapis.com/openimages/web/factsfigures_v7.html) |
| **P3M-10k** | 顔をぼかしたポートレート＋手作業のアルファマット | 10,421枚（学習9,421 / P3M-500-P / P3M-500-NP） | 配布規約に「MIT license」と明記。画像の著作権は元の所有者にあり、「free use license」のサイトから集めたとされる | 学習可（画像の権利は元サイトに依存する点に注意） | ✅ [配布規約 PDF](https://jizhizili.github.io/files/p3m_dataset_agreement/P3M-10k_Dataset_Release_Agreement.pdf) |
| **COCO**（person のマスク） | 一般写真 | person が約26万件 | アノテーションは CC BY 4.0。画像は Flickr の規約と、1枚ごとのライセンスに従う（NC のものも混ざる） | 学習可（画像ごとのライセンスで NC を除く） | ⚠️ 二次情報のみ |
| **AISegment Matting Human** | 半身ポートレート 600×800 ＋ マット | 34,425枚 | README にライセンスの記載がない。ミラー（HF、Dataset Ninja）は MIT と表示しているが根拠がない。元画像は Flickr・百度・淘宝 | 評価のみ可 | ✅ README に記載がないことを確認 |
| **Supervisely Person** | 人物マスク（Pexels の画像） | 5,711枚 | 「non-commercial purposes」に限る | 評価のみ可 | ✅ [README](https://github.com/supervisely-ecosystem/persons) |
| **PPM-100** | ポートレートのマット（評価用） | 100枚 | マットは CC BY-NC-SA 4.0 | 評価のみ可 | ⚠️ |
| **PP-HumanSeg14K** | ビデオ会議の映像から切り出したもの | 約1.4万枚 | データ自体のライセンスが見つからない（ローダーのコードは Apache 2.0） | 評価のみ可（要確認） | ⚠️ |
| **VideoMatte240K** | 動画のマット（グリーンバック） | 484本 | BackgroundMattingV2 のコードは MIT だが、データのライセンスは明記されていない | 評価のみ可（要確認） | ⚠️ |
| **HaGRID**（手） | 手のジェスチャー（bbox。マスクではない）。v1 は FullHD、主に屋内でカメラから 0.5〜4m の距離で撮った画像、18クラス（`peace` を含む） | v1: 552,992枚（716GB）。各クラス100枚の subsample が 2.5GB | CC BY-SA 4.0 を書き直した独自ライセンス（"not a Creative Commons license"）。EasyPortrait と同じ GitHub アカウント（hukenovs）が配布し、同じ形式の文書。商用の禁止はない。派生物は BY-SA 互換のライセンスで出す | 指の評価用（#12） | ✅ [license/en_us.pdf](https://github.com/hukenovs/hagrid/blob/Hagrid_v1/license/en_us.pdf)（2026-11-25 23:17 に確認） |
| **EgoHands**（手） | 一人称視点の手のマスク | 4,800枚 | ミラーでは CC BY 4.0 と表示 | 保留（要確認） | ⚠️ |
| DAVIS 2017（ちらつきの評価） | 動画の物体マスク | 150本 | 今回は確かめられなかった | 保留（要確認） | — |
| LIP / CelebAMask-HQ / SA-1B など | — | — | 研究用途に限る | 使わない（学習）。必要なら評価だけ | 記憶ベース。使う段で確認する |

## 論点

1. **EasyPortrait の ShareAlike**: 学習したモデルの重みが「Adapted Material」にあたるかは、解釈が分かれる。安全側に倒すなら、EasyPortrait を学習に使ったモデルは CC BY-SA 4.0（または互換ライセンス）で公開する。npm パッケージのコード（MIT など）とモデルの重み（CC BY-SA）でライセンスを分ければ両立できる。→ 人間に相談する。
2. **手の IoU**: EasyPortrait には手のクラスがない（person に含まれる）。
   - **訂正（2026-11-25 13:28 の実行で確認）**: Open Images V7 のセグメンテーションの対象350クラス（`classes-segmentation.txt`）に Human hand は入っていない。手は bbox だけ。検証・テスト分割の手のマスクは0個だった
   - 代わりの案: Open Images の person 系マスク（Person / Man / Woman / Boy / Girl）と Human hand の bbox を組み合わせ、「手の bbox の中での person マスクの IoU」を手の領域の IoU とする。テスト分割で、手の bbox が person マスクの人物の bbox に収まる例は1,229個（753枚）、検証分割で408個（256枚）
   - 12例を目視した（作業ログの画像。リポジトリには入れていない）。9〜10例はマスクの輪郭が指に沿っていた。1例は、マスクのない別の人物の手だった。bbox の包含だけでは足りないので、「手の bbox の中で person マスクが一定の割合以上を占める」条件を足す
   - **確定（同じ実行）**: `bench/openimages_hand_index.py` で、テスト分割の手の bbox 2,630個（person マスクのある753枚）について、person マスクの和が bbox を覆う割合を出した（`data/openimages/hand_eval_index_test.csv`）。分布は二山で、0.1 未満が1,461個、0.5 以上が973個。目視で外れだった例は 0.0、良かった11例はすべて 0.5 以上。**しきい値 0.3 で 1,065個（617枚）**を手の領域のテストセットにする
   - 限界: 手が自分の体の前にある例（被覆率 1.0 付近、314個）は、bbox の中がほぼ全部 person なので指の輪郭の精度をあまり測れない
   - 注意: Open Images はビデオ通話の映像ではなく一般の写真。手の領域の IoU は、ドメインが違う補助の指標として扱う
3. **テストセットの案**: ビデオ会議という用途に最も近いのは EasyPortrait のテスト分割（4,000枚）。メイン指標の mIoU はここで測り、Open Images の person ＋ hand を補助のテストセットにする案が有力。MediaPipe のモデルが学習に何を使ったかは非公開なので、どのテストセットでもデータの重なりは確かめられない。

## 見かけた関連研究

- arXiv 2611.14558「Selecting Person Segmentation Models for Real-Time Video Call Backgrounds on the CPU」（2026年11月）。CPU でのビデオ通話用モデル選定。ブラウザ（WebGPU）での比較とは違うが、比較表を作るときに必ず読む。

## 出典

- EasyPortrait: https://github.com/hukenovs/easyportrait 、https://huggingface.co/datasets/gofixyourself/EasyPortrait 、https://arxiv.org/abs/2304.13509
- Open Images V7: https://storage.googleapis.com/openimages/web/factsfigures_v7.html
- P3M-10k: https://github.com/JizhiziLi/P3M
- Supervisely Person: https://github.com/supervisely-ecosystem/persons
- AISegment: https://github.com/aisegmentcn/matting_human_datasets
- PPM-100: https://github.com/ZHKKKe/PPM
- HaGRID: https://github.com/hukenovs/hagrid
- PP-HumanSeg: https://github.com/PaddlePaddle/PaddleSeg/tree/release/2.8/contrib/PP-HumanSeg
- BackgroundMattingV2: https://github.com/PeterL1n/BackgroundMattingV2
- MediaPipe モデル: `https://storage.googleapis.com/mediapipe-models/image_segmenter/<name>/...` から HTTP 200 で取得できることを確認（multiclass float32 は約16MB、landscape float16 は約250KB）

## HaGRID のライセンスの確認（2026-11-25 23:17）

- `Hagrid_v1` ブランチと `master` ブランチの `license/en_us.pdf` は、SHA-256 が同じ（`14f4845e…4d22`）。同じライセンスの文書
- 文書は CC BY-SA 4.0 の書き直しで、非商用の制限は無い。条件は、出典の表示、変更したことの表示、派生物を同じ条件か BY-SA 互換のライセンスで出すこと
- `Hagrid_v2` の README も、同じ PDF を指している。#1 のコメントにある「v2 は非商用研究に限る」は、この README と PDF からは確かめられなかった。v1 を使う方針は変えない
- 配布元のストレージ（sbercloud）は、この日の確認では HTTP 502 を返した。取得は次の回で試す

## HaGRID の取得元（2026-11-26 09:05）

- v1 の README にある取得先（`n-usr-2uzac.s3pd12.sbercloud.ru` と `n-ws-620xz-pd11.s3pd11.sbercloud.ru`）は、プロキシの CONNECT が 502 で失敗した。前日と同じ結果
- いまの README が指す取得先（`rndml-team-cv.obs.ru-moscow-1.hc.sbercloud.ru`）は応答する。HTTP Range も使える
- ここにあるジェスチャー別の zip（`hagrid_dataset_new_554800/`）は、v1（552,992枚、2022年）ではなく **HaGRID 2.0**（554,800枚、2023年、ブランチ `Hagrid_v2`）。`Hagrid_v2` の README も、v1 と同じライセンス PDF を指す。非商用の制限は、どの README と PDF にも見当たらない
- そこで、手元の評価には HaGRID 2.0 を使う。画像は git に入れず、`data/hagrid/` に置く。1M 枚の HaGRIDv2（2024年）は使わない
- 取得は `scripts/fetch_hagrid.sh`。クラスの zip（約40GB）から、名前順の先頭 N 枚だけを Range で読む。名前は UUID なので、並べて先頭を取るだけで再現できる無作為抽出になる
