はい、**可能です**。実装を入れました。

## 現在の構造

`AlignNet`（論文の condensation / alignment に相当）は、**画像ブランチと言語ブランチそれぞれ**で次のどちらかになっています。

| `align_num_layers` | 中身 |
|--------------------|------|
| **2**（デフォルト） | `Linear → ReLU → Linear`（従来どおりの 2 層 MLP） |
| **1** | **`Linear` のみ**（ブランチあたり 1 つの全結合層） |

正規化はこれまでどおり `forward` 末尾の `feature_norm` です。

## 切り替え方

Hydra で上書きします。

```bash
# 投影を 1 層のみにする
python src/train.py model.net.align_num_layers=1
```

GPU や実験プリセットと組み合わせる例:

```bash
python src/train.py trainer=gpu model.net.align_num_layers=1 experiment=ablation_full
```

## 設定ファイル

`configs/model/kda.yaml` にデフォルトを追加してあります。

- 省略時は **`align_num_layers: 2`**（論文実装に近い 2 段 MLP）
- 比較実験では CLI や別 YAML で **`align_num_layers: 1`** を指定してください

**注意:** Teacher が凍結のまま、`align_num_layers` を変えると**学習可能パラメータ数**が変わるため、公平な比較には学習率や訓練長の揃え方も検討するとよいです。