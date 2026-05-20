このリポジトリでは **`python src/train.py`** を実行すると、Hydra が **`configs/train.yaml` を根（ルート）** にして、いくつかの YAML をくっつけた「1 つの設定オブジェクト」を作り、それを `train()` に渡します。あなたがコマンドで書く **`key=value` は、その完成済み設定の一部を上書きする**というイメージです。

---

## 1. 設定がどう組み立てられるか

`configs/train.yaml` 冒頭の `defaults:` が **読み込む順**を決めています。

```4:13:configs/train.yaml
defaults:
  - _self_
  - data: kd_data
  - model: kda
  - callbacks: default
  - logger: wandb # set logger here or use command line (e.g. `python train.py logger=tensorboard`)
  - trainer: ddp
  - paths: default
  - extras: default
  - hydra: default
```

意味の整理です。

| 行の形 | 意味（このプロジェクトで） |
|--------|------------------------------|
| `data: kd_data` | `configs/data/kd_data.yaml` を読み、全体の `data` にマージ |
| `model: kda` | `configs/model/kda.yaml` → `model` |
| `trainer: ddp` | `configs/trainer/ddp.yaml` → `trainer` |
| `experiment: null` | 最初は実験用設定なし。`experiment=名前` で差し替え |

**後から読んだものが前を上書き**するので、同じキーがあると後勝ちです。つまり「ベース + 上書き」を重ねていく仕組みです。

---

## 2. コマンドでの指定の基本形

すべて **`python src/train.py`** のあとに **スペース区切りで `キー=値`** を並べます。

```bash
python src/train.py キー1=値1 キー2=値2
```

**キーは設定の階層を `.` でつなぐ**のが基本です（ネストした YAML のパス）。

例（実在する構造に基づく）:

```bash
python src/train.py model.use_img_kd=false
python src/train.py trainer.max_epochs=50
python src/train.py data.batch_size=128
```

- **`model`** … `configs/model/kda.yaml` がマージされたあとのトップレベル `model`
- **`trainer`** … `configs/trainer/*.yaml` がマージされたあとの `trainer`

「どこに何があるか」は **`configs/` 以下の YAML を開いて木構造を見る**のが確実です。

---

## 3. 「グループ」を一発で差し替える書き方

`trainer: ddp` のように **「グループ名: 選ぶファイル」** で繋いでいる部分は、CLI でも短く書けます。

```bash
python src/train.py trainer=gpu
```

これは **`configs/trainer/gpu.yaml` を使う**という意味で、`ddp` の代わりに `gpu` に差し替わります（中身は基本 `ddp` の継承＋GPU 指定など）。

同様に:

- `logger=tensorboard` → `configs/logger/tensorboard.yaml`
- `data.attributes=6_StanfordCars` → `data` の下の `attributes` という**サブ設定**を、`configs/data/attributes/6_StanfordCars.yaml` にする、という指定方法（config group の指定＋名前）

---

## 4. `experiment=...` は何をするか

`train.yaml` には `experiment: null` が入っていて、デフォルトでは実験用 YAML は読みません。

```bash
python src/train.py experiment=ablation_full
```

とすると **`configs/experiment/ablation_full.yaml` がマージ**され、その中で `trainer` や `model`、`logger` などがまとめて上書きされます。  
「よく使う組み合わせを 1 ファイルにまとめる」ための仕組みです。

---

## 5. よくある書き方のルール

**リスト**（`tags: ["a", "b"]` のようなもの）は、シェルで解釈されないように **引用符で包む**ことが多いです。

```bash
python src/train.py 'tags=[dev, try1]'
```

**文字列にスペースがある**場合も引用が必要です。

**`null` を渡す**（オフにする）とき:

```bash
python src/train.py seed=null
```

**複数ジョブに分ける（multirun）** は `-m` です（カンマ区切りでスイープ）。

```bash
python src/train.py -m experiment=ablation_cls_only,ablation_full seed=42
```

---

## 6. 実行結果のログ・出力先

このプロジェクトの Hydra 設定では、実行のたびに **ログ用ディレクトリが 1 つ作られます**。

```9:12:configs/hydra/default.yaml
run:
  dir: ${paths.log_dir}/${task_name}/runs/${now:%Y-%m-%d}_${now:%H-%M-%S}
sweep:
  dir: ${paths.log_dir}/${task_name}/multiruns/${now:%Y-%m-%d}_${now:%H-%M-%S}
```

- **通常 1 回の実行** → `logs/train/runs/日時/` のような場所（`paths.log_dir` は `configs/paths/default.yaml` で定義）
- **`-m` の multirun** → `multiruns/日時/` の下に番号付きサブフォルダ

コンソールに **実際の絶対パス**が Hydra から表示されるので、「どこに出たか」はそこを見るとよいです。

---

## 7. 覚えておくと楽なコツ

1. **いま何が効いているか**を知りたい → 実行ログの先頭に、Hydra が **merged config** をよく出します。それが「最終的な 1 本の設定」です。
2. **上書きは CLI が勝つ**つもりでよい（細かい例外はあるが、実験ではその理解で十分なことが多い）。
3. **迷ったら** → `configs/train.yaml` の `defaults` で「どのファイルが土台か」を追い、**変えたい値がどの YAML のどのキーか**を探して `トップレベル.その下.…=値` とする。

---

## 8. このリポジトリ向けの最小例

```bash
# GPU・データだけ変える
python src/train.py trainer=gpu data.attributes=6_StanfordCars

# 実験プリセット＋シード
python src/train.py experiment=ablation_cls_only data.attributes=2_NABirds seed=123

# エポック数だけ変える
python src/train.py trainer=gpu trainer.max_epochs=20
```

まとめると、**Hydra は「YAML をくっつけて 1 つの辞書にし、CLI の `a.b=c` でその辞書を上書きする」仕組み**だと思うと分かりやすいです。