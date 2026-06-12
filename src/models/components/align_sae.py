from pathlib import Path
from typing import Any, Dict, Tuple, Union

import torch
from torch import nn

feature_norm = lambda x: x / (x.norm(dim=-1, keepdim=True) + 1e-10)


def _build_align_mlp(in_features: int, out_features: int, num_layers: int) -> nn.Module:
    if num_layers not in (1, 2):
        raise ValueError(f"align num_layers must be 1 or 2, got {num_layers}")
    if num_layers == 1:
        return nn.Linear(in_features, out_features)
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.ReLU(),
        nn.Linear(out_features, out_features),
    )


def _build_decoder_mlp(latent_features: int, out_features: int, num_layers: int) -> nn.Module:
    if num_layers not in (1, 2):
        raise ValueError(f"align num_layers must be 1 or 2, got {num_layers}")
    if num_layers == 1:
        return nn.Linear(latent_features, out_features)
    return nn.Sequential(
        nn.Linear(latent_features, latent_features),
        nn.ReLU(),
        nn.Linear(latent_features, out_features),
    )


class AlignSAE(nn.Module):
    """Sparse autoencoder on frozen teacher embeddings.

    Encoder weights use the same module names as ``AlignNet`` so they can be
    loaded directly for phase-2 knowledge distillation.
    """

    def __init__(self, in_features: int, latent_features: int, num_layers: int = 2):
        super().__init__()
        self.align_img_layer = _build_align_mlp(in_features, latent_features, num_layers)
        self.align_nlp_layer = _build_align_mlp(in_features, latent_features, num_layers)
        self.decode_img_layer = _build_decoder_mlp(latent_features, in_features, num_layers)
        self.decode_nlp_layer = _build_decoder_mlp(latent_features, in_features, num_layers)

    def encode(self, img_features: torch.Tensor, nlp_features: torch.Tensor):
        return self.align_img_layer(img_features), self.align_nlp_layer(nlp_features)

    def decode(
        self, latent_img: torch.Tensor, latent_nlp: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.decode_img_layer(latent_img), self.decode_nlp_layer(latent_nlp)

    def forward_kd(
        self, img_features: torch.Tensor, nlp_features: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        latent_img, latent_nlp = self.encode(img_features, nlp_features)
        return feature_norm(latent_img), feature_norm(latent_nlp)

    def forward_pretrain(
        self, img_features: torch.Tensor, nlp_features: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        latent_img, latent_nlp = self.encode(img_features, nlp_features)
        recon_img, recon_nlp = self.decode(latent_img, latent_nlp)
        return {
            "latent_img": latent_img,
            "latent_nlp": latent_nlp,
            "recon_img": recon_img,
            "recon_nlp": recon_nlp,
            "aligned_img": feature_norm(latent_img),
            "aligned_nlp": feature_norm(latent_nlp),
        }

    def encoder_state_dict(self) -> Dict[str, Dict[str, torch.Tensor]]:
        return {
            "align_img_layer": self.align_img_layer.state_dict(),
            "align_nlp_layer": self.align_nlp_layer.state_dict(),
        }


def load_align_encoder_checkpoint(
    align_net: nn.Module, checkpoint: Union[str, Path, Dict[str, Any]]
) -> None:
    """Load SAE encoder weights into an ``AlignNet`` instance."""
    if isinstance(checkpoint, (str, Path)):
        checkpoint = torch.load(checkpoint, map_location="cpu")

    if isinstance(checkpoint, dict) and "align_img_layer" in checkpoint:
        encoder_state = checkpoint
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        encoder_state: Dict[str, Dict[str, torch.Tensor]] = {}
        for key, value in state_dict.items():
            if ".align_img_layer." in key:
                encoder_state.setdefault("align_img_layer", {})[
                    key.split(".align_img_layer.", 1)[1]
                ] = value
            elif key.startswith("align_img_layer."):
                encoder_state.setdefault("align_img_layer", {})[
                    key[len("align_img_layer.") :]
                ] = value
            elif ".align_nlp_layer." in key:
                encoder_state.setdefault("align_nlp_layer", {})[
                    key.split(".align_nlp_layer.", 1)[1]
                ] = value
            elif key.startswith("align_nlp_layer."):
                encoder_state.setdefault("align_nlp_layer", {})[
                    key[len("align_nlp_layer.") :]
                ] = value
        if "align_img_layer" not in encoder_state or "align_nlp_layer" not in encoder_state:
            raise KeyError(
                "Could not find align encoder weights in Lightning checkpoint."
            )
    else:
        raise KeyError(
            "Unsupported align encoder checkpoint format. "
            "Expected encoder_state dict or Lightning checkpoint."
        )

    align_net.align_img_layer.load_state_dict(encoder_state["align_img_layer"])
    align_net.align_nlp_layer.load_state_dict(encoder_state["align_nlp_layer"])
