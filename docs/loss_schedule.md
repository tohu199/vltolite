# Loss weight schedule (L_cls / KD)

Teacher 使用時、`L_cls` と KD 項（L_vis / L_txt）にはエポック依存の重み `w_cls`, `w_kd` が掛かります。スケジュールは Hydra の config group `model/loss_schedule` で切り替えます。

## Presets

| Config | `w_cls` (epoch 0) | `w_kd` (epoch 0) | 挙動 |
|--------|-------------------|------------------|------|
| `linear_slow`（デフォルト） | 0 | 1 | 序盤 KD 優位。`w_cls` が `epoch / (max_epochs × 8)` まで増加 |
| `linear_ramp_kd` | 1 | 0 | 序盤分類優位。`w_kd` を同じ式で 0 から増加 |
| `constant` | 1 | 1 | スケジュールなし |

## CLI

```bash
python src/train.py model/loss_schedule=linear_slow
python src/train.py model/loss_schedule=linear_ramp_kd
python src/train.py model/loss_schedule=constant
python src/train.py experiment=ablation_cls_vis model/loss_schedule=linear_ramp_kd
```

`ramp_epochs_factor` の上書き（`linear_slow` / `linear_ramp_kd` のみ）:

```bash
python src/train.py model.loss_schedule.ramp_epochs_factor=4
```

KD 項の強さ（L_vis / L_txt それぞれ）: `model.img_kd_scale` / `model.txt_kd_scale`（デフォルト `1.0`、`0.1` で 1/10）。

実装: `src/models/components/loss_schedule.py`
