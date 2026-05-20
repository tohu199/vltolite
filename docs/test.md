# src/eval.py

`src/eval.py` は **学習はせず、指定したチェックポイントで `trainer.test` だけ実行する** Hydra エントリです。

---

## 何をするか

```48:75:src/eval.py
    assert cfg.ckpt_path
    ...
    trainer.test(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt_path)
```

- **`ckpt_path`** … 読み込む `.ckpt`（**必須**。未指定だと `eval.yaml` の `???` で落ちます）
- **`data`** … テスト用 `test_dataloader()` を持つ DataModule
- **`model`** … `KDModule` 等、学習時と**同じ構造のモデル**（重みは ckpt から読み込み）

ログは任意（`logger: null` なしでもよい）。

---

## 基本的な使い方（VL2Lite / KD 向け）

`configs/eval.yaml` はテンプレのまま **`data: mnist` / `model: mnist`** なので、このプロジェクトで評価するときは **上書き**します。

```bash
python src/eval.py \
  data=kd_data \
  model=kda \
  ckpt_path=/絶対または相対パス/to/last.ckpt
```

GPU なら例:

```bash
python src/eval.py data=kd_data model=kda trainer=gpu \
  ckpt_path=logs/train/runs/.../checkpoints/last.ckpt
```

データセットやアブレーションを学習時と揃える例:

```bash
python src/eval.py data=kd_data model=kda trainer=gpu \
  data.attributes=6_StanfordCars \
  model.use_img_kd=true model.use_txt_kd=true \
  ckpt_path=path/to.ckpt
```

**注意:** `model` のハイパラ（`use_teacher`、投影層の深さなど）は **学習時と同じ**にしないと、チェックポイントの形と合わず読み込みに失敗します。学習ログや `.hydra/config.yaml` を参照して揃えるのが安全です。

---

## ログを付けたいとき

```bash
python src/eval.py data=kd_data model=kda \
  ckpt_path=... logger=csv
```

---

## `train.py` との違い

| | `train.py` | `eval.py` |
|---|------------|-----------|
| 学習 (`fit`) | あり | **なし** |
| テスト (`test`) | 設定 `test: True` のときのみ | **常に**（`ckpt` 付き） |

学習の最後にすでに `test` まで回しているなら、`eval.py` は **別データでの再評価・別 ckpt の比較・本番推論前の確認**などに使います。

---

## テスト・プログラムから呼ぶ

`tests/test_eval.py` のように、`evaluate(cfg)` に **Hydra の `DictConfig`** を渡して直接呼ぶこともできます（`ckpt_path` と `data`/`model` が揃っている必要があります）。

---

**まとめ:** `python src/eval.py data=kd_data model=kda ckpt_path=<path>` を基本形にし、**学習時と同じ `data` / `model` 設定**に揃えるのがコツです。