from typing import Literal, Tuple

import numpy as np


class LinearRampSchedule:
    """Ramp one loss weight from 0→1 over max_epochs * factor; the other is 1 - ramp."""

    def __init__(
        self,
        ramp: Literal["cls", "kd"] = "cls",
        ramp_epochs_factor: float = 8.0,
    ) -> None:
        self.ramp = ramp
        self.ramp_epochs_factor = ramp_epochs_factor

    def __call__(self, epoch: int, max_epochs: int) -> Tuple[float, float]:
        if max_epochs <= 0:
            return (1.0, 1.0)
        t = float(
            np.clip(epoch / (max_epochs * self.ramp_epochs_factor), 0.0, 1.0)
        )
        if self.ramp == "cls":
            return t, 1.0 - t
        return 1.0 - t, t


class ConstantSchedule:
    def __init__(self, w_cls: float = 1.0, w_kd: float = 1.0) -> None:
        self.w_cls = w_cls
        self.w_kd = w_kd

    def __call__(self, epoch: int, max_epochs: int) -> Tuple[float, float]:
        return self.w_cls, self.w_kd
