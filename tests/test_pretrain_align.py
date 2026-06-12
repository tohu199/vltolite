from pathlib import Path

import pytest
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, open_dict

from src.pretrain_align import pretrain


def test_pretrain_align_fast_dev_run(cfg_pretrain_align: DictConfig) -> None:
    HydraConfig().set_config(cfg_pretrain_align)
    with open_dict(cfg_pretrain_align):
        cfg_pretrain_align.trainer.fast_dev_run = True
        cfg_pretrain_align.trainer.accelerator = "cpu"
    pretrain(cfg_pretrain_align)


def test_pretrain_align_writes_encoder_checkpoint(
    cfg_pretrain_align: DictConfig, tmp_path: Path
) -> None:
    HydraConfig().set_config(cfg_pretrain_align)
    with open_dict(cfg_pretrain_align):
        cfg_pretrain_align.trainer.max_epochs = 1
        cfg_pretrain_align.trainer.limit_train_batches = 1
        cfg_pretrain_align.trainer.limit_val_batches = 1
        cfg_pretrain_align.trainer.accelerator = "cpu"
        cfg_pretrain_align.data.batch_size = 2
        cfg_pretrain_align.data.num_workers = 0
        cfg_pretrain_align.model.teacher.arch = "ViT-B-32"
        cfg_pretrain_align.model.teacher.pretrained = "openai"
    pretrain(cfg_pretrain_align)
    encoder_path = tmp_path / "align_encoder.ckpt"
    assert encoder_path.exists()
