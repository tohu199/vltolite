# 環境構築（WSL2）

WSL2（Ubuntu）上で VL2Lite を動かす手順です。GPU 学習を使う場合は、**Windows 側に NVIDIA ドライバ**を入れたうえで WSL2 を使ってください。

## 1. 前提

| 項目 | 推奨 |
|------|------|
| OS | WSL2 + Ubuntu 22.04 など |
| Python | 3.9 以上（3.10 / 3.11 でも可） |
| GPU（任意） | CUDA 対応 NVIDIA GPU + Windows ドライバ |

WSL 内で GPU が見えるか確認:

```bash
nvidia-smi
```

## 2. リポジトリの取得

```bash
git clone https://github.com/jsjangAI/VL2Lite
cd VL2Lite
```

## 3. Python 仮想環境

### venv（推奨）

```bash
sudo apt update
sudo apt install -y python3-venv python3-pip

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip setuptools wheel
```

### conda（任意）

```bash
conda create -n vl2lite python=3.10 -y
conda activate vl2lite
```

## 4. PyTorch のインストール

[PyTorch 公式](https://pytorch.org/get-started/locally/) から **WSL + CUDA** 用のコマンドを選んで実行します。例（CUDA 12.1）:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

CPU のみで試す場合:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 5. 依存パッケージ

```bash
pip install -r requirements.txt
```

`requirements.txt` に含まれていないが、学習に必要なパッケージ:

```bash
pip install open_clip_torch pandas scipy
```

## 6. 動作確認

プロジェクトルート（`VL2Lite/`）で実行します。

```bash
# CPU・数ステップだけ（CIFAR-10、初回はデータ自動 DL）
python src/train.py experiment=smoke_test data/attributes=10_CIFAR10 logger=local trainer=cpu
```

GPU がある場合:

```bash
python src/train.py experiment=smoke_test data/attributes=10_CIFAR10 logger=local trainer=gpu
```

ログは `logs/train/runs/` 以下に出力されます。Weights & Biases を使わない場合は `logger=local` または `logger=tensorboard` を指定してください。

## 7. よくあるトラブル

**`ModuleNotFoundError: No module named 'hydra'`**  
仮想環境が有効化されていない、または `pip install -r requirements.txt` 未実行です。

**`ModuleNotFoundError: No module named 'open_clip'`**  
`pip install open_clip_torch` を実行してください。

**GPU が認識されない**  
Windows 側の NVIDIA ドライバを更新し、WSL を再起動してから `nvidia-smi` を再確認してください。

**Hydra のデータセット切り替えでエラー**  
`data.attributes=...` ではなく **`data/attributes=...`**（スラッシュ）を使います。詳細は [cifar10.md](./cifar10.md) を参照してください。

## 8. 次のステップ

- データセットの準備・切り替え: [exp/cifar10.md](./cifar10.md)
- 実験設定の一覧: `configs/experiment/`
- データセット追加の詳細: [docs/Change_Dataset.md](../docs/Change_Dataset.md)
