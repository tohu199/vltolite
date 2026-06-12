import tempfile
from pathlib import Path

import pytest
import torch

from src.models.components.align_sae import AlignSAE, load_align_encoder_checkpoint
from src.models.components.campus import AlignNet


@pytest.mark.parametrize("num_layers", [1, 2])
def test_align_sae_forward_shapes(num_layers: int) -> None:
    teacher_dim = 64
    latent_dim = 16
    batch_size = 4
    num_classes = 5

    sae = AlignSAE(teacher_dim, latent_dim, num_layers=num_layers)
    img = torch.randn(batch_size, teacher_dim)
    nlp = torch.randn(num_classes, teacher_dim)
    img = img / img.norm(dim=-1, keepdim=True)
    nlp = nlp / nlp.norm(dim=-1, keepdim=True)

    pretrain_out = sae.forward_pretrain(img, nlp)
    assert pretrain_out["recon_img"].shape == img.shape
    assert pretrain_out["recon_nlp"].shape == nlp.shape
    assert pretrain_out["latent_img"].shape == (batch_size, latent_dim)
    assert pretrain_out["latent_nlp"].shape == (num_classes, latent_dim)

    aligned_img, aligned_nlp = sae.forward_kd(img, nlp)
    assert aligned_img.shape == (batch_size, latent_dim)
    assert aligned_nlp.shape == (num_classes, latent_dim)
    assert torch.allclose(aligned_img.norm(dim=-1), torch.ones(batch_size), atol=1e-5)


def test_encoder_checkpoint_loads_into_alignnet() -> None:
    teacher_dim = 32
    latent_dim = 8
    sae = AlignSAE(teacher_dim, latent_dim, num_layers=1)
    align = AlignNet(teacher_dim, latent_dim, num_layers=1)

    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "align_encoder.ckpt"
        torch.save(sae.encoder_state_dict(), ckpt_path)
        load_align_encoder_checkpoint(align, ckpt_path)

    img = torch.randn(2, teacher_dim)
    nlp = torch.randn(3, teacher_dim)
    sae_out = sae.forward_kd(img, nlp)
    align_out = align(img, nlp)
    for left, right in zip(sae_out, align_out):
        assert torch.allclose(left, right, atol=1e-6)


def test_load_from_lightning_state_dict() -> None:
    teacher_dim = 16
    latent_dim = 4
    sae = AlignSAE(teacher_dim, latent_dim, num_layers=1)
    align = AlignNet(teacher_dim, latent_dim, num_layers=1)

    state_dict = {}
    for key, value in sae.state_dict().items():
        if key.startswith("align_"):
            state_dict[f"align_sae.{key}"] = value

    load_align_encoder_checkpoint(align, {"state_dict": state_dict})
    img = torch.randn(1, teacher_dim)
    nlp = torch.randn(2, teacher_dim)
    sae_out = sae.forward_kd(img, nlp)
    align_out = align(img, nlp)
    assert torch.allclose(sae_out[0], align_out[0], atol=1e-6)
