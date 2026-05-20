### Loss ablation (L_cls / L_vis / L_txt)

Toggle KD terms with `model.use_img_kd` (L_vis) and `model.use_txt_kd` (L_txt).
Keep `model.use_teacher=true` so all runs share the same student architecture.

| Setting | Command |
|---------|---------|
| L_cls only | `python src/train.py experiment=ablation_cls_only` |
| L_cls + L_vis | `python src/train.py experiment=ablation_cls_vis` |
| L_cls + L_txt | `python src/train.py experiment=ablation_cls_txt` |
| L_cls + L_vis + L_txt | `python src/train.py experiment=ablation_full` |

Run all four sequentially:
```bash
bash src/train_ablation.sh
# optional overrides, e.g. dataset and seed:
bash src/train_ablation.sh data.attributes=6_StanfordCars seed=42
```

Or override flags directly:
```bash
python src/train.py trainer=gpu model.use_img_kd=false model.use_txt_kd=false
```

----








実装しました。変更点と4条件の実行方法をまとめます。

## 変更内容

**`src/models/kd_module.py`**
- `use_img_kd`（L_vis）、`use_txt_kd`（L_txt）フラグを追加
- 有効な KD 項だけを平均して `loss = L_cls + mean(active KD)` に変更（片方だけオフでも重みが半分にならない）

**`configs/model/kda.yaml`**
- デフォルト `use_img_kd: true`, `use_txt_kd: true`

**`configs/experiment/ablation_*.yaml`**
- 4条件それぞれのプリセット（GPU・WandB 名付き）

**`src/train_ablation.sh`**
- 4条件を順に実行するスクリプト

**`README.md`**
- アブレーション手順を追記

---

## 4条件の対応

| 実験 | L_cls | L_vis | L_txt | 設定 |
|------|:-----:|:-----:|:-----:|------|
| cls only | ✓ | ✗ | ✗ | `use_img_kd=false`, `use_txt_kd=false` |
| cls + vis | ✓ | ✓ | ✗ | `use_img_kd=true`, `use_txt_kd=false` |
| cls + txt | ✓ | ✗ | ✓ | `use_img_kd=false`, `use_txt_kd=true` |
| full | ✓ | ✓ | ✓ | 両方 `true` |

いずれも **`model.use_teacher=true`** のままにし、Student アーキテクチャを揃えています（`use_teacher=false` は別ベースライン用です）。

---

## 実行方法

### 方法1: 実験プリセット（推奨）

```bash
# リポジトリルートで
python src/train.py experiment=ablation_cls_only    # L_cls のみ
python src/train.py experiment=ablation_cls_vis       # L_cls + L_vis
python src/train.py experiment=ablation_cls_txt       # L_cls + L_txt
python src/train.py experiment=ablation_full          # L_cls + L_vis + L_txt
```

### 方法2: 4本まとめて順実行

```bash
bash src/train_ablation.sh
```

データセットや seed を揃える例:

```bash
bash src/train_ablation.sh data.attributes=6_StanfordCars seed=42
```

### 方法3: CLI で直接指定

```bash
python src/train.py trainer=gpu model.use_img_kd=false model.use_txt_kd=false  # L_cls
python src/train.py trainer=gpu model.use_img_kd=true  model.use_txt_kd=false  # + L_vis
python src/train.py trainer=gpu model.use_img_kd=false model.use_txt_kd=true   # + L_txt
python src/train.py trainer=gpu model.use_img_kd=true  model.use_txt_kd=true   # full
```

### Hydra multirun（並列・グリッド）

```bash
python src/train.py -m \
  experiment=ablation_cls_only,ablation_cls_vis,ablation_cls_txt,ablation_full \
  seed=42
```

---

## 評価

各 run で従来どおり **epoch ごとの validation** と、学習後の **test（ベスト ckpt）** が走ります。WandB の run 名には `_visTrue_txtFalse` のような suffix が付きます。

比較のため、可能なら **同じ `seed`・データセット** で4条件を揃えてください。