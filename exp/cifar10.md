# データセット

VL2Lite では Hydra の **属性設定**（`configs/data/attributes/`）でクラス数・クラス名・テキストプロンプトを切り替え、同じ `src/train.py` で各データセットを学習できます。

## ディレクトリ構成

デフォルトのデータルートは `configs/paths/default.yaml` で次のように定義されています。

```
${PROJECT_ROOT}/data/kd_datasets/<data_name>/
```

`<data_name>` は各属性 YAML の `name` フィールドと一致します（例: `0_CUB_200_2011`, `10_CIFAR10`）。

別の場所にデータがある場合は、シンボリックリンクを張るか `paths.data_dir` を上書きします。

```bash
# 例: 既存データをリンク
ln -s /path/to/KD_datasets ./data/kd_datasets

# 例: パスを直接指定（末尾は kd_datasets/）
python src/train.py data/attributes=0_CUB_200_2011 paths.data_dir=/path/to/kd_datasets/
```

## データセットの切り替え（Hydra）

`configs/data/kd_data.yaml` のデフォルトは **CUB-200-2011**（`0_CUB_200_2011`）です。

別データセットに切り替えるときは **config group** として指定します（ドットではなくスラッシュ）。

```bash
python src/train.py data/attributes=<属性ファイル名> trainer=gpu
```

| 属性ファイル名 | データセット | 備考 |
|----------------|--------------|------|
| `0_CUB_200_2011` | CUB-200-2011 | 既定 |
| `1_FGVC_AIRCRAFT` | FGVC Aircraft | 手動配置 |
| `2_NABirds` | NABirds | 手動配置 |
| `3_DTD` | DTD | 手動配置 |
| `4_OxfordIIITPet` | Oxford-IIIT Pet | 手動配置 |
| `5_StanfordDogs` | Stanford Dogs | 手動配置 |
| `6_StanfordCars` | Stanford Cars | 手動配置 |
| `7_CALTECH101` | Caltech-101 | 手動配置 |
| `8_CALTECH256` | Caltech-256 | 手動配置 |
| `9_GTSRB` | GTSRB | 手動配置 |
| `10_CIFAR10` | CIFAR-10 | **初回実行時に自動ダウンロード** |
| `11_MNIST` | MNIST | **初回実行時に自動ダウンロード** |

CUB の実行例:

```bash
python src/train.py data/attributes=0_CUB_200_2011 trainer=gpu
```

アブレーション実験プリセットを使う例:

```bash
python src/train.py experiment=ablation_full data/attributes=6_StanfordCars
```

複数データセットを順に回す（multirun）:

```bash
python src/train.py -m experiment=ablation_full \
  data/attributes=1_FGVC_AIRCRAFT,2_NABirds,6_StanfordCars
```

## 手動配置が必要なデータセット

CIFAR-10 / MNIST 以外は、公式サイト等から取得したデータを
`data/kd_datasets/<data_name>/` に配置してください。  
各データセットの期待するフォルダ構成は `src/data/components/kd_dataloader.py` 内の Dataset クラスを参照してください。

## 新しいデータセットを追加する

1. `configs/data/attributes/` に YAML を追加（`name`, `class_num`, `classes`, `prompt_tmpl`, `sub_dir` など）
2. `src/data/components/kd_dataloader.py` の `get_dataloader` にローダを追加（未実装の場合）

詳細: [docs/Change_Dataset.md](../docs/Change_Dataset.md)

---

## CIFAR-10

CIFAR-10 は **手動ダウンロード不要** です。初回実行時に次のパスへ自動取得されます。

```
data/kd_datasets/10_CIFAR10/
```

### 学習コマンド

```bash
python src/train.py data/attributes=10_CIFAR10 trainer=gpu
```

### スモークテスト（CPU・2 train steps）

環境構築後の動作確認向け:

```bash
python src/train.py experiment=smoke_test data/attributes=10_CIFAR10 logger=local trainer=cpu
```

### よく使う上書き例

```bash
python src/train.py data/attributes=10_CIFAR10 \
  trainer.max_epochs=100 \
  data.batch_size=128 \
  model.net.student.arch=resnet18 \
  logger=tensorboard
```

CIFAR-10 は画像が 32×32 ですが、既存パイプラインの transform により 224×224 にリサイズして CLIP 教師モデルへ入力します。

### データのみ取得する

学習前に CIFAR-10 だけダウンロードする場合:

```bash
python scripts/download_cifar10.py
```

### トラブルシューティング

**`RuntimeError: File not found or corrupted.`**  
ダウンロードが途中で中断されると `cifar-10-python.tar.gz` が壊れた状態で残ることがあります。次で削除してから再実行してください（再ダウンロードされます）。

```bash
rm -f data/kd_datasets/10_CIFAR10/cifar-10-python.tar.gz
python src/train.py experiment=smoke_test data/attributes=10_CIFAR10 logger=local trainer=cpu
```

---

## MNIST

MNIST も **手動ダウンロード不要** です。初回実行時に次のパスへ自動取得されます。

```
data/kd_datasets/11_MNIST/
```

### 学習コマンド

```bash
python src/train.py data/attributes=11_MNIST trainer=gpu
```

### スモークテスト（CPU・2 train steps）

```bash
python src/train.py experiment=smoke_test data/attributes=11_MNIST logger=local trainer=cpu
```

### よく使う上書き例

```bash
python src/train.py data/attributes=11_MNIST \
  trainer.max_epochs=100 \
  data.batch_size=128 \
  model.net.student.arch=resnet18 \
  logger=tensorboard
```

MNIST は 28×28 のグレースケール画像ですが、読み込み時に RGB に変換し、既存パイプラインの transform により 224×224 にリサイズして CLIP 教師モデルへ入力します。

### データのみ取得する

```bash
python scripts/download_mnist.py
```

### トラブルシューティング

**ダウンロードが途中で中断された場合**  
`data/kd_datasets/11_MNIST/MNIST/raw/` 内の壊れた `.gz` は次回実行時に再取得されます。うまくいかない場合は raw フォルダを削除してから再実行してください。

```bash
rm -rf data/kd_datasets/11_MNIST/MNIST/raw
python src/train.py experiment=smoke_test data/attributes=11_MNIST logger=local trainer=cpu
```
