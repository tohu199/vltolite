## 環境構築
```bash
source .venv_smoke/bin/activate
pip install -r requirements.txt
pip install open_clip_torch pandas scipy 'wandb>=0.12.10'
```

## データセット準備
```bash
python scripts/download_mnist.py
```


## 学習
```bash
python src/train.py data/attributes=11_MNIST trainer=gpu logger=local \
  data.batch_size=128 data.num_workers=4
```

10 epoch のみ、モデルが8GBに載るように
```bash
python src/train.py data/attributes=11_MNIST trainer=gpu logger=local trainer.max_epochs=10 \
  model.net.teacher.arch=ViT-B-32 model.net.teacher.pretrained=openai \
  data.batch_size=128 data.num_workers=4
```


### cls_txt
```bash
python src/train.py \
  experiment=ablation_cls_txt_ramp_kd \
  data/attributes=11_MNIST \
  logger=local \
  model.net.teacher.arch=ViT-B-32 model.net.teacher.pretrained=openai \
  data.batch_size=64 data.num_workers=4 \
  trainer.max_epochs=10

```