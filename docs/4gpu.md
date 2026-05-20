このリポジトリでは **4 GPU 用の設定がすでにあります**。

## 推奨: `trainer=ddp`

`configs/trainer/ddp.yaml` は次のようになっています。

```1:9:configs/trainer/ddp.yaml
defaults:
  - default

strategy: ddp

accelerator: gpu
devices: 4
num_nodes: 1
sync_batchnorm: True
```

そのため **次で 4 GPU の DDP 学習**になります。

```bash
python src/train.py trainer=ddp
```

なお **`configs/train.yaml` のデフォルトはすでに `trainer: ddp`** なので、引数を付けずに

```bash
python src/train.py
```

でも、デフォルトどおり **4 GPU（ddp）** で動きます（README の `trainer=gpu` は **1 GPU 用**の別プリセットです）。

---

## `trainer=gpu` から 4 GPU にしたい場合

`trainer=gpu` は `devices: 1` なので、次のように **台数と戦略**を上書きします。

```bash
python src/train.py trainer=gpu trainer.devices=4 trainer.strategy=ddp
```

（4 枚で学習するなら **`ddp` 設定をそのまま使う方が設定と揃っていて分かりやすい**です。）

---

## バッチサイズについて

`KDDataModule` はワールドサイズ（GPU 数）で `batch_size` を割るので、**設定の `data.batch_size` は「全体の合計」**のイメージです。`512` なら GPU あたり 128 になります。  

**`batch_size` は GPU 数で割り切れる必要があります**（4 で `512 % 4 == 0` なので問題ありません）。

---

## GPU の指定（任意）

特定の物理 GPU だけ使う場合:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python src/train.py trainer=ddp
```

---

**まとめ:** 4 GPU なら **`trainer=ddp`**、または何も付けずにデフォルトのまま。README の **`trainer=gpu` は 1 GPU 向け**なので、4 枚にしたいときは `ddp` を使うのがこのコードベースでは自然です。