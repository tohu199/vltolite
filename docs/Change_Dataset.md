CUB 以外でも、**データ用の属性設定（クラス数・プロンプトなど）を差し替えれば**同じコマンドで実験できます。

## 1. データセットの切り替え（Hydra）

`configs/data/kd_data.yaml` のデフォルトは `attributes: 0_CUB_200_2011` です。別データセットにするには **`data/attributes` を上書き**します（Hydra の config group 指定。`data.attributes` ではなく `data/attributes`）。

利用可能な定義ファイルは `configs/data/attributes/` 内の名前に対応します（例）：

| ファイル名 | データセット例 |
|-----------|------------------|
| `0_CUB_200_2011` | CUB（既定） |
| `1_FGVC_AIRCRAFT` | FGVC Aircraft |
| `2_NABirds` | NABirds |
| `3_DTD` | DTD |
| `4_OxfordIIITPet` | Oxford-IIIT Pet |
| `5_StanfordDogs` | Stanford Dogs |
| `6_StanfordCars` | Stanford Cars |
| `7_CALTECH101` | Caltech-101 |
| `8_CALTECH256` | Caltech-256 |
| `9_GTSRB` | GTSRB |
| `10_CIFAR10` | CIFAR-10（自動ダウンロード） |
| `11_MNIST` | MNIST（自動ダウンロード） |

実行例（Stanford Cars のとき）:

```bash
python src/train.py data/attributes=6_StanfordCars trainer=gpu
```

アブレーション実験プリセットを使う場合:

```bash
python src/train.py experiment=ablation_full data/attributes=6_StanfordCars
```

スクリプトでまとめて回す場合:

```bash
bash src/train_ablation.sh data/attributes=6_StanfordCars seed=42
```

複数データセットを順番に回す（Hydra のリストではなく、`-m` でスイープする例）:

```bash
python src/train.py -m experiment=ablation_full \
  data/attributes=1_FGVC_AIRCRAFT,2_NABirds,6_StanfordCars
```

## 2. データの置き場所

`configs/paths/default.yaml` では、データは次のようなパス前提です。

- **`${PROJECT_ROOT}/data/kd_datasets/<data_name>/`**  
  ここで `<data_name>` は各属性 YAML の `name` フィールドと一致（例: `6_StanfordCars`）。

README のとおり、別パスにデータがある場合はシンボリックリンクするか、`paths.data_dir` を上書きしてください。

```bash
python src/train.py data/attributes=9_GTSRB paths.data_dir=/your/path/kd_datasets/
```

（末尾は `kd_datasets/` のまま、`data_name` サブフォルダがその下に来る形です。）

## 3. 新しいデータセットを追加するとき

1. `configs/data/attributes/` に、`name`・`class_num`・`classes`・`prompt_tmpl`・`sub_dir` などを定義した YAML を追加する。  
2. `src/data/components/kd_dataloader.py` の `get_dataloader` に、その `data_name` 用のクラスが既にあるか確認（なければローダを追加）。  

---

**まとめ:** CUB 以外は **`data/attributes=<属性ファイル名>`** を付けるだけでよく、データは **`data/kd_datasets/<その名前>/`** に置くのがデフォルトの前提です。手順の詳細は [exp/cifar10.md](../exp/cifar10.md) も参照してください。