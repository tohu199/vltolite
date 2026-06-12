from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric

from src.models.components.align_sae import AlignSAE
from src.models.components.campus import StudentNet, TeacherNet, feature_norm


class AlignPretrainModule(LightningModule):
    """Phase-1 training: frozen teacher + AlignSAE reconstruction/sparsity."""

    def __init__(
        self,
        teacher,
        student,
        data_attributes,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        recon_criterion,
        align_num_layers: int = 2,
        sparsity_weight: float = 1e-3,
        text_recon_weight: float = 1.0,
        compile: bool = False,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.teacher = TeacherNet(teacher)
        student_stub = StudentNet(student, data_attributes.class_num, use_teacher=True)
        self.align_sae = AlignSAE(
            self.teacher.last_features_dim,
            student_stub.num_features,
            num_layers=align_num_layers,
        )
        self.frozen_nlp_features = self._build_frozen_nlp_features(data_attributes)
        self.recon_criterion = recon_criterion

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.train_recon_img = MeanMetric()
        self.train_recon_nlp = MeanMetric()
        self.train_sparse = MeanMetric()
        self.val_recon_img = MeanMetric()
        self.val_recon_nlp = MeanMetric()
        self.val_sparse = MeanMetric()

    def _build_frozen_nlp_features(self, attributes):
        prompt_tmpl = attributes.prompt_tmpl
        classes_list = list(attributes.classes.values())
        text_tokens = self.teacher.tokenizer(
            [prompt_tmpl.format(word) for word in classes_list]
        )
        nlp_features = self.teacher.encode_text(text_tokens).detach()
        return feature_norm(nlp_features)

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        x, _y = batch
        clip_img_features = self.teacher(x)
        nlp_features = self.frozen_nlp_features.to(clip_img_features.device)

        latent_img = self.align_sae.align_img_layer(clip_img_features)
        recon_img = self.align_sae.decode_img_layer(latent_img)
        img_recon = self.recon_criterion(recon_img, clip_img_features)
        img_sparse = latent_img.abs().mean()

        latent_nlp = self.align_sae.align_nlp_layer(nlp_features)
        recon_nlp = self.align_sae.decode_nlp_layer(latent_nlp)
        nlp_recon = self.recon_criterion(recon_nlp, nlp_features)
        nlp_sparse = latent_nlp.abs().mean()

        sparse_term = self.hparams.sparsity_weight * (img_sparse + nlp_sparse)
        text_term = self.hparams.text_recon_weight * nlp_recon
        loss = img_recon + text_term + sparse_term

        loss_dict = {
            "loss": loss,
            "recon_img": img_recon,
            "recon_nlp": nlp_recon,
            "sparse": img_sparse + nlp_sparse,
        }
        return loss, loss_dict

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        loss, loss_dict = self.model_step(batch)
        self.train_loss(loss)
        self.train_recon_img(loss_dict["recon_img"])
        self.train_recon_nlp(loss_dict["recon_nlp"])
        self.train_sparse(loss_dict["sparse"])
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log(
            "train/recon_img",
            self.train_recon_img,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/recon_nlp",
            self.train_recon_nlp,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "train/sparse",
            self.train_sparse,
            on_step=False,
            on_epoch=True,
        )
        return loss

    def validation_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> None:
        loss, loss_dict = self.model_step(batch)
        self.val_loss(loss)
        self.val_recon_img(loss_dict["recon_img"])
        self.val_recon_nlp(loss_dict["recon_nlp"])
        self.val_sparse(loss_dict["sparse"])

    def on_validation_epoch_end(self) -> None:
        self.log("val/loss", self.val_loss, prog_bar=True)
        self.log("val/recon_img", self.val_recon_img)
        self.log("val/recon_nlp", self.val_recon_nlp)
        self.log("val/sparse", self.val_sparse)

    def on_fit_end(self) -> None:
        if self.trainer is None or not self.trainer.is_global_zero:
            return
        save_dir = Path(self.trainer.default_root_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        encoder_path = save_dir / "align_encoder.ckpt"
        torch.save(self.align_sae.encoder_state_dict(), encoder_path)

    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.align_sae = torch.compile(self.align_sae)

    def configure_optimizers(self) -> Dict[str, Any]:
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = self.hparams.optimizer(params=params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}
