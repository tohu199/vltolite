from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from lightning import LightningModule
from lightning.pytorch.loggers import TensorBoardLogger
from torchmetrics import MaxMetric, MeanMetric
from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassConfusionMatrix,
    MulticlassRecall,
)
from torchmetrics.classification.accuracy import Accuracy


class KDModule(LightningModule):
    """Example of a `LightningModule` for MNIST classification.

    A `LightningModule` implements 8 key methods:

    ```python
    def __init__(self):
    # Define initialization code here.

    def setup(self, stage):
    # Things to setup before each stage, 'fit', 'validate', 'test', 'predict'.
    # This hook is called on every process when using DDP.

    def training_step(self, batch, batch_idx):
    # The complete training step.

    def validation_step(self, batch, batch_idx):
    # The complete validation step.

    def test_step(self, batch, batch_idx):
    # The complete test step.

    def predict_step(self, batch, batch_idx):
    # The complete predict step.

    def configure_optimizers(self):
    # Define and configure optimizers and LR schedulers.
    ```

    Docs:
        https://lightning.ai/docs/pytorch/latest/common/lightning_module.html
    """

    def __init__(
        self,
        net: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        use_teacher: bool,
        kd_criterion,
        compile: bool,
        use_img_kd: bool = True,
        use_txt_kd: bool = True,
        img_kd_scale: float = 1.0,
        txt_kd_scale: float = 1.0,
        loss_schedule=None,
        save_test_failures: bool = False,
        test_failure_max_images: int = 64,
        save_test_tsne: bool = False,
        test_tsne_max_samples: int = 2000,
        test_tsne_perplexity: int = 30,
    ) -> None:
        """Initialize a `MNISTLitModule`.

        :param net: The model to train.
        :param optimizer: The optimizer to use for training.
        :param scheduler: The learning rate scheduler to use for training.
        """
        super().__init__()

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(logger=False)

        self.net = net
        self.use_teacher = use_teacher
        if use_teacher:
            self.kd_criterion = kd_criterion
            if loss_schedule is None:
                raise ValueError(
                    "loss_schedule is required when use_teacher=true "
                    "(set model/loss_schedule in config)."
                )
            self.loss_schedule = loss_schedule
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # metric objects for calculating and averaging accuracy across batches
        num_classes = self.hparams.net.data_attributes.class_num
        self.train_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_acc_per_class = MulticlassAccuracy(
            num_classes=num_classes,
            average="none",
        )
        self.test_acc = Accuracy(task="multiclass", num_classes=num_classes)
        self.test_confusion_matrix = MulticlassConfusionMatrix(num_classes=num_classes)
        # Per-class recall (true class k: how often k is predicted as k)
        self.test_recall_per_class = MulticlassRecall(
            num_classes=num_classes, average="none"
        )

        # for averaging loss across batches
        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()
        self.test_loss = MeanMetric()
        if use_teacher:
            self.kd_loss = MeanMetric()
            self.cls_loss = MeanMetric()
            self.img_loss = MeanMetric()

        # for tracking best so far validation accuracy
        self.val_acc_best = MaxMetric()

        # misclassified test samples for optional export (reset each test epoch)
        self._test_failure_samples: List[Dict[str, Any]] = []
        # test-time embeddings for optional t-SNE (reset each test epoch)
        self._test_tsne_student: List[np.ndarray] = []
        self._test_tsne_teacher_img: List[np.ndarray] = []
        self._test_tsne_teacher_txt: List[np.ndarray] = []
        self._test_tsne_labels: List[int] = []

    def on_test_epoch_start(self) -> None:
        self.test_confusion_matrix.reset()
        self.test_recall_per_class.reset()
        self._test_failure_samples = []
        self._test_tsne_student = []
        self._test_tsne_teacher_img = []
        self._test_tsne_teacher_txt = []
        self._test_tsne_labels = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a forward pass through the model `self.net`.

        :param x: A tensor of images.
        :return: A tensor of logits.
        """
        return self.net(x)

    def on_train_start(self) -> None:
        """Lightning hook that is called when training begins."""
        # by default lightning executes validation step sanity checks before training starts,
        # so it's worth to make sure validation metrics don't store results from these checks
        self.val_loss.reset()
        self.val_acc.reset()
        self.val_acc_per_class.reset()
        self.val_acc_best.reset()

    def on_validation_epoch_start(self) -> None:
        self.val_acc_per_class.reset()

    def model_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform a single model step on a batch of data.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target labels.

        :return: A tuple containing (in order):
            - A tensor of losses.
            - A tensor of predictions.
            - A tensor of target labels.
        """

        x, y = batch
        loss_dict = {}
        if self.use_teacher:
            outputs = self.forward(x)
            img_loss, kd_loss = self.kd_criterion(outputs)
            cls_loss = self.criterion(outputs[1], y)

            cls_loss_weight, kd_loss_weight = self.loss_schedule(
                self.current_epoch, self.trainer.max_epochs
            )
            cls_loss = cls_loss_weight * cls_loss
            if self.hparams.use_img_kd:
                img_loss = (
                    kd_loss_weight * self.hparams.img_kd_scale * img_loss
                )
            else:
                img_loss = img_loss.detach() * 0
            if self.hparams.use_txt_kd:
                kd_loss = (
                    kd_loss_weight * self.hparams.txt_kd_scale * kd_loss
                )
            else:
                kd_loss = kd_loss.detach() * 0

            kd_terms = []
            if self.hparams.use_img_kd:
                kd_terms.append(img_loss)
            if self.hparams.use_txt_kd:
                kd_terms.append(kd_loss)
            kd_total = sum(kd_terms) / len(kd_terms) if kd_terms else cls_loss * 0
            loss = cls_loss + kd_total

            preds = torch.argmax(outputs[1], dim=1)
            loss_dict["loss"] = loss
            loss_dict["cls_loss"] = cls_loss
            loss_dict["img_loss"] = img_loss
            loss_dict["kd_loss"] = kd_loss
        else:
            logits = self.forward(x)
            loss = self.criterion(logits, y)
            preds = torch.argmax(logits, dim=1)
            loss_dict["loss"] = loss
        return loss, loss_dict, preds, y

    def training_step(
        self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        """Perform a single training step on a batch of data from the training set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        :return: A tensor of losses between model predictions and targets.
        """
        loss, loss_dicts, preds, targets = self.model_step(batch)

        # update and log metrics
        self.train_loss(loss)
        self.train_acc(preds, targets)
        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train/acc", self.train_acc, on_step=False, on_epoch=True, prog_bar=True)

        if self.use_teacher:
            self.cls_loss(loss_dicts["cls_loss"])
            self.img_loss(loss_dicts["img_loss"])
            self.kd_loss(loss_dicts["kd_loss"])
            self.log("train/cls_loss", self.cls_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/img_loss", self.img_loss, on_step=False, on_epoch=True, prog_bar=True)
            self.log("train/kd_loss", self.kd_loss, on_step=False, on_epoch=True, prog_bar=True)
        # return loss or backpropagation will fail
        return loss

    def on_train_epoch_start(self) -> None:
        if not self.use_teacher:
            return
        w_cls, w_kd = self.loss_schedule(
            self.current_epoch, self.trainer.max_epochs
        )
        self.log("train/w_cls", w_cls, on_step=False, on_epoch=True)
        self.log("train/w_kd", w_kd, on_step=False, on_epoch=True)

    def on_train_epoch_end(self) -> None:
        "Lightning hook that is called when a training epoch ends."
        pass

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single validation step on a batch of data from the validation set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, loss_dicts, preds, targets = self.model_step(batch)

        # update and log metrics
        self.val_loss(loss)
        self.val_acc(preds, targets)
        self.val_acc_per_class(preds, targets)
        self.log("val/loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val/acc", self.val_acc, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        "Lightning hook that is called when a validation epoch ends."
        acc = self.val_acc.compute()  # get current val acc
        self.val_acc_best(acc)  # update best so far val acc
        # log `val_acc_best` as a value through `.compute()` method, instead of as a metric object
        # otherwise metric would be reset by lightning after each epoch
        self.log("val/acc_best", self.val_acc_best.compute(), sync_dist=True, prog_bar=True)

        per_class = self.val_acc_per_class.compute().detach()  # (C,)
        self.log(
            "val/mean_per_class_acc",
            per_class.mean(),
            sync_dist=True,
            prog_bar=False,
        )

        if self.trainer and getattr(self.trainer, "loggers", None) and self.trainer.is_global_zero:
            step = int(self.trainer.global_step)
            self._log_val_per_class_tensorboard(per_class, step)

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        """Perform a single test step on a batch of data from the test set.

        :param batch: A batch of data (a tuple) containing the input tensor of images and target
            labels.
        :param batch_idx: The index of the current batch.
        """
        loss, loss_dicts, preds, targets = self.model_step(batch)

        # update and log metrics
        self.test_loss(loss)
        self.test_acc(preds, targets)
        self.log("test/loss", self.test_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test/acc", self.test_acc, on_step=False, on_epoch=True, prog_bar=True)

        self.test_confusion_matrix(preds, targets)
        self.test_recall_per_class(preds, targets)

        if self.hparams.save_test_failures:
            self._collect_test_failures(batch, preds, targets)

        if self.hparams.save_test_tsne and self.use_teacher:
            self._collect_test_tsne_features(batch)

    def on_test_epoch_end(self) -> None:
        """Log per-class recall and confusion matrix (figures / scalars) after test epoch."""
        recall_pc = self.test_recall_per_class.compute().detach()  # (C,)
        cm = self.test_confusion_matrix.compute().detach()  # (C, C)

        macro_rec = recall_pc.mean()
        self.log(
            "test/macro_recall",
            macro_rec,
            prog_bar=False,
            sync_dist=True,
        )

        step = int(getattr(self.trainer, "global_step", 0) or 0)
        if self.hparams.save_test_failures:
            self._export_test_failure_visualizations(step)
        if self.hparams.save_test_tsne and self.use_teacher:
            self._export_test_tsne_visualizations(step)

        if not self.trainer or not getattr(self.trainer, "loggers", None):
            return

        self._log_test_extended_metrics(
            confusion_matrix=cm,
            recall_per_class=recall_pc,
            step=step,
        )

    def _log_test_extended_metrics(
        self,
        confusion_matrix: torch.Tensor,
        recall_per_class: torch.Tensor,
        step: int,
    ) -> None:
        """Log confusion matrix image / histograms to TensorBoard and scalars to all loggers."""
        if not self.trainer.is_global_zero:
            return

        num_classes = confusion_matrix.shape[0]
        cm_np = confusion_matrix.float().cpu().numpy()
        rec_np = recall_per_class.float().cpu().numpy()

        row_sum = cm_np.sum(axis=1, keepdims=True).clip(min=1.0)
        cm_row_norm = cm_np / row_sum

        loggers: List = list(self.trainer.loggers or [])
        for lg in loggers:
            if isinstance(lg, TensorBoardLogger):
                exp = lg.experiment
                exp.add_histogram(
                    "test/per_class_recall",
                    recall_per_class,
                    global_step=step,
                )
                try:
                    import matplotlib.pyplot as plt

                    fig_size = float(min(28.0, 6.0 + num_classes * 0.12))
                    fig, ax = plt.subplots(figsize=(fig_size, fig_size))
                    im = ax.imshow(cm_row_norm, aspect="auto", cmap="Blues", vmin=0.0, vmax=1.0)
                    ax.set_xlabel("predicted")
                    ax.set_ylabel("true")
                    ax.set_title("test confusion (row-normalized: p(pred | true))")
                    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                    plt.tight_layout()
                    exp.add_figure(
                        "test/confusion_matrix_row_norm",
                        fig,
                        global_step=step,
                    )
                    plt.close(fig)

                    fig2, ax2 = plt.subplots(figsize=(max(8.0, num_classes * 0.06), 4.0))
                    ax2.bar(np.arange(num_classes), rec_np, width=1.0, align="edge")
                    ax2.set_xlabel("class id (0-based)")
                    ax2.set_ylabel("per-class recall")
                    ax2.set_title("test per-class recall")
                    ax2.set_xlim(0, num_classes)
                    fig2.tight_layout()
                    exp.add_figure(
                        "test/per_class_recall_bar",
                        fig2,
                        global_step=step,
                    )
                    plt.close(fig2)
                except ImportError:
                    pass

    def _denormalize_clip_image(self, x_chw: torch.Tensor) -> torch.Tensor:
        """Invert CLIP-style normalization used in `KDDataset` transforms."""
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], device=x_chw.device, dtype=x_chw.dtype)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711], device=x_chw.device, dtype=x_chw.dtype)
        mean = mean.view(3, 1, 1)
        std = std.view(3, 1, 1)
        return (x_chw * std + mean).clamp(0.0, 1.0)

    def _class_display_name(self, class_idx: int) -> str:
        classes = self.hparams.net.data_attributes.classes
        if classes is None:
            return str(class_idx)
        for key in (class_idx + 1, class_idx, str(class_idx + 1), str(class_idx)):
            try:
                if hasattr(classes, "get"):
                    v = classes.get(key)
                else:
                    v = classes[key] if key in classes else None
                if v is not None:
                    return str(v)
            except (KeyError, TypeError, IndexError, RuntimeError):
                continue
        return str(class_idx)

    def _collect_test_tsne_features(
        self, batch: Tuple[torch.Tensor, torch.Tensor]
    ) -> None:
        """Accumulate student / teacher image / teacher text (true-class) embeddings."""
        max_n = int(self.hparams.test_tsne_max_samples)
        if len(self._test_tsne_labels) >= max_n:
            return

        x, y = batch
        with torch.no_grad():
            outputs = self.net(x)
        hidden, _out, clip_img, clip_nlp, _aligned_img, _aligned_nlp = outputs
        # Per-image teacher text: frozen class embedding of the ground-truth label
        txt_per_image = clip_nlp[y]

        for i in range(x.shape[0]):
            if len(self._test_tsne_labels) >= max_n:
                break
            self._test_tsne_student.append(hidden[i].detach().cpu().numpy())
            self._test_tsne_teacher_img.append(clip_img[i].detach().cpu().numpy())
            self._test_tsne_teacher_txt.append(txt_per_image[i].detach().cpu().numpy())
            self._test_tsne_labels.append(int(y[i].item()))

    def _export_test_tsne_visualizations(self, step: int) -> None:
        """Run t-SNE per feature space and save PNG / npz under the run directory."""
        if not self._test_tsne_labels or not self.trainer:
            return

        n = len(self._test_tsne_labels)
        if n < 4:
            return

        try:
            from sklearn.manifold import TSNE
        except ImportError:
            return

        labels = np.array(self._test_tsne_labels, dtype=np.int64)
        student = np.stack(self._test_tsne_student, axis=0)
        teacher_img = np.stack(self._test_tsne_teacher_img, axis=0)
        teacher_txt = np.stack(self._test_tsne_teacher_txt, axis=0)

        perplexity = min(
            int(self.hparams.test_tsne_perplexity),
            max(5, (n - 1) // 3),
            n - 1,
        )

        root = Path(self.trainer.default_root_dir or ".")
        out_dir = root / "test_tsne" / f"rank_{self.trainer.global_rank:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        coords: Dict[str, np.ndarray] = {}
        for name, feats in (
            ("student", student),
            ("teacher_image", teacher_img),
            ("teacher_text_true_class", teacher_txt),
        ):
            tsne = TSNE(
                n_components=2,
                perplexity=perplexity,
                init="pca",
                learning_rate="auto",
                random_state=42,
            )
            coords[name] = tsne.fit_transform(feats).astype(np.float32)

        np.savez(
            out_dir / "tsne_coords.npz",
            labels=labels,
            student=coords["student"],
            teacher_image=coords["teacher_image"],
            teacher_text_true_class=coords["teacher_text_true_class"],
            perplexity=perplexity,
            num_samples=n,
        )

        titles = {
            "student": "Student hidden features (t-SNE)",
            "teacher_image": "Teacher image features (CLIP, t-SNE)",
            "teacher_text_true_class": "Teacher text features (true-class embedding, t-SNE)",
        }
        tb_names = {
            "student": "test/tsne_student",
            "teacher_image": "test/tsne_teacher_image",
            "teacher_text_true_class": "test/tsne_teacher_text",
        }

        for key in ("student", "teacher_image", "teacher_text_true_class"):
            fig = self._tsne_scatter_figure(coords[key], labels, titles[key])
            fig.savefig(out_dir / f"{key}_tsne.png", dpi=150, bbox_inches="tight")
            if self.trainer.is_global_zero:
                for lg in list(self.trainer.loggers or []):
                    if isinstance(lg, TensorBoardLogger):
                        lg.experiment.add_figure(tb_names[key], fig, global_step=step)
            try:
                import matplotlib.pyplot as plt

                plt.close(fig)
            except ImportError:
                pass

    def _tsne_scatter_figure(self, xy: np.ndarray, labels: np.ndarray, title: str):
        import matplotlib.pyplot as plt

        num_classes = int(labels.max()) + 1 if labels.size else 1
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=labels,
            s=8,
            alpha=0.65,
            cmap="tab20",
            vmin=0,
            vmax=max(num_classes - 1, 1),
        )
        ax.set_title(title)
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        fig.tight_layout()
        return fig

    def _collect_test_failures(
        self,
        batch: Tuple[torch.Tensor, torch.Tensor],
        preds: torch.Tensor,
        targets: torch.Tensor,
    ) -> None:
        max_n = int(self.hparams.test_failure_max_images)
        if len(self._test_failure_samples) >= max_n:
            return
        x, _y = batch
        wrong = preds != targets
        idxs = torch.where(wrong)[0]
        for i in idxs.tolist():
            if len(self._test_failure_samples) >= max_n:
                break
            self._test_failure_samples.append(
                {
                    "x": x[i].detach().cpu(),
                    "pred": int(preds[i].item()),
                    "target": int(targets[i].item()),
                }
            )

    def _export_test_failure_visualizations(self, step: int) -> None:
        """Write misclassified images under the run directory and log a grid to TensorBoard."""
        if not self._test_failure_samples or not self.trainer:
            return

        from torchvision.utils import make_grid, save_image

        root = Path(self.trainer.default_root_dir or ".")
        out_dir = root / "test_failures" / f"rank_{self.trainer.global_rank:02d}"
        out_dir.mkdir(parents=True, exist_ok=True)

        for j, item in enumerate(self._test_failure_samples):
            img = self._denormalize_clip_image(item["x"])
            t, p = item["target"], item["pred"]
            fn = f"fail_{j:04d}_true{t}_pred{p}.png"
            save_image(img, out_dir / fn)

        manifest_path = out_dir / "manifest.txt"
        with open(manifest_path, "w", encoding="utf-8") as f:
            f.write("index\ttarget\tpred\ttarget_name\tpred_name\n")
            for j, item in enumerate(self._test_failure_samples):
                tn = self._class_display_name(item["target"])
                pn = self._class_display_name(item["pred"])
                f.write(
                    f"{j}\t{item['target']}\t{item['pred']}\t{tn}\t{pn}\n"
                )

        stack = torch.stack(
            [self._denormalize_clip_image(s["x"]) for s in self._test_failure_samples],
            dim=0,
        )
        nrow = min(8, max(1, stack.shape[0]))
        grid = make_grid(stack, nrow=nrow, padding=2)
        save_image(grid, out_dir / "failures_grid.png")

        if not self.trainer.is_global_zero:
            return

        for lg in list(self.trainer.loggers or []):
            if isinstance(lg, TensorBoardLogger):
                lg.experiment.add_image("test/misclassified_grid", grid, global_step=step)

    def _log_val_per_class_tensorboard(
        self, per_class_acc: torch.Tensor, step: int
    ) -> None:
        """Log per-class validation accuracy to TensorBoard (histogram + bar chart)."""
        num_classes = per_class_acc.shape[0]
        acc_np = per_class_acc.float().cpu().numpy()

        for lg in list(self.trainer.loggers or []):
            if not isinstance(lg, TensorBoardLogger):
                continue
            exp = lg.experiment
            exp.add_histogram("val/per_class_accuracy", per_class_acc, global_step=step)
            try:
                import matplotlib.pyplot as plt

                fig, ax = plt.subplots(figsize=(max(8.0, num_classes * 0.06), 4.0))
                ax.bar(np.arange(num_classes), acc_np, width=1.0, align="edge")
                ax.set_xlabel("class id (0-based)")
                ax.set_ylabel("per-class accuracy")
                ax.set_title("val per-class accuracy")
                ax.set_xlim(0, num_classes)
                ax.set_ylim(0.0, 1.05)
                fig.tight_layout()
                exp.add_figure("val/per_class_accuracy_bar", fig, global_step=step)
                plt.close(fig)
            except ImportError:
                pass

    def setup(self, stage: str) -> None:
        """Lightning hook that is called at the beginning of fit (train + validate), validate,
        test, or predict.

        This is a good hook when you need to build models dynamically or adjust something about
        them. This hook is called on every process when using DDP.

        :param stage: Either `"fit"`, `"validate"`, `"test"`, or `"predict"`.
        """
        if self.hparams.compile and stage == "fit":
            self.net = torch.compile(self.net)

    def configure_optimizers(self) -> Dict[str, Any]:
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Examples:
            https://lightning.ai/docs/pytorch/latest/common/lightning_module.html#configure-optimizers

        :return: A dict containing the configured optimizers and learning-rate schedulers to be used for training.
        """
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
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


if __name__ == "__main__":
    _ = MNISTLitModule(None, None, None, None)
