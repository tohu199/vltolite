# テスト時の混同行列・失敗画像について

`trainer.test`（`src/train.py` の `test: True` 後のテスト、または `src/eval.py`）で `KDModule` が **test** ループを回したときの挙動です。

---

## 混同行列

### 何が計算されているか

- **集計**: `torchmetrics.MulticlassConfusionMatrix` で、テスト全バッチを通じた **クラス数 \(C\) の平方行列**（非正規化の件数）。
- **行列の意味（実装の解釈）**: 行 **i** = 真クラス **i**、列 **j** = 予測クラス **j** に対応するカウントがよく使われます（torchmetrics の multiclass 定義に準拠）。

### 「保存」されるもの

- **数値行列そのもの**（`.npy` / `.csv` など）を **ディスクに自動書き出しはしていません**。
- 代わりに次が出力されます。

| 出力先 | 内容 |
|--------|------|
| **TensorBoard**（`logger=tensorboard` または `logger=local` など TB 含有） | **`test/confusion_matrix_row_norm`**: 行方向に正規化したヒートマップ（**真クラスごとの条件付き分布 \(p(\text{予測} \mid \text{真})\)** のイメージで読むと分かりやすい）。対角が大きいほど、その真クラスでは正解率が高い。 |
| 同上 | **`test/per_class_recall`**: クラス別 recall を **ヒストグラム**として記録。 |
| 同上 | **`test/per_class_recall_bar`**: クラス ID 横軸の **棒グラフ**。 |
| **任意のロガー**（CSV 含む） | **`test/macro_recall`**: クラス別 recall ベクトルの **平均**（スカラー）。 |
| すべてのロガー経由の共通メトリクス | **`test/loss`**, **`test/acc`**: 従来どおり。 |

混同行列を **ファイルで残したい**場合は、テスト終了後に TensorBoard からエクスポートするか、`KDModule` 側に保存処理を足す必要があります。

### TensorBoard で確認する

```bash
tensorboard --logdir=<Hydra 実行ディレクトリ内の tensorboard パス>
```

実行ディレクトリはコンソールに Hydra が表示する **`logs/train/runs/日時_時刻`** などです（`configs/hydra/default.yaml` の `run.dir`）。

---

## 失敗画像（誤分類の可視化）

### 有効化

`configs/model/kda.yaml` のフラグ、またはコマンドラインで上書きします。

```yaml
save_test_failures: false   # true で有効
test_failure_max_images: 64 # 収集する最大枚数
```

例:

```bash
python src/train.py model.save_test_failures=true model.test_failure_max_images=48 logger=tensorboard
```

### 何を集めるか

- **test** の各バッチで、**予測クラス ≠ 正解ラベル** のサンプルのみ。
- 先頭から最大 **`test_failure_max_images`** 枚で打ち切り（全誤答を保存するわけではない）。

### ディスクへの保存

Hydra の **その run の出力ルート**（通常 `Trainer.default_root_dir` = 当該 run のディレクトリ）の下:

```
test_failures/rank_XX/
  manifest.txt          # index, target, pred, クラス名（可能なら属性 YAML の classes から解決）
  fail_0000_true{t}_pred{p}.png
  ...
  failures_grid.png     # 収集した失敗画像のグリッド
```

- **`rank_XX`**: 分散学習時はプロセス番号ごとに別フォルダ。**各 rank が担当したテスト分割内**の失敗だけが入る（全件を 1 フォルダに集約はしない）。
- 画像はデータローダの **CLIP 用正規化を逆変換**したうえで RGB として保存。

### TensorBoard

- **global rank 0** のみ、**`test/misclassified_grid`** としてグリッド画像を 1 枚送信（複数プロセスで同じ TB に重ねないため）。

### ロガーなしでも

`logger=null` でも **`save_test_failures=true`** なら **PNG と `manifest.txt` は保存**されます（TB 用の混同行列図などは、ロガー設定に依存）。

### 注意（データの扱い）

このリポジトリでは **validation と test が同じ split を参照する実装**になっている点に注意してください（開発・デバッグ用の理解として）。本番的な「汎化性能」として報告する場合は、split の設計を別途検討してください。
