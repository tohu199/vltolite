import numpy as np
import pytest

from src.models.components.loss_schedule import ConstantSchedule, LinearRampSchedule


def _legacy_weights(epoch: int, max_epochs: int) -> tuple[float, float]:
    weight1 = epoch / (max_epochs * 8)
    weight1 = float(np.clip(weight1, 0, 1))
    weight2 = float(np.clip(1 - weight1, 0, 1))
    return weight1, weight2


@pytest.mark.parametrize("epoch", [0, 149, 299])
def test_linear_slow_matches_legacy(epoch: int) -> None:
    schedule = LinearRampSchedule(ramp="cls", ramp_epochs_factor=8.0)
    assert schedule(epoch, 300) == pytest.approx(_legacy_weights(epoch, 300))


def test_linear_ramp_kd_at_boundaries() -> None:
    schedule = LinearRampSchedule(ramp="kd", ramp_epochs_factor=8.0)
    assert schedule(0, 300) == (1.0, 0.0)
    w_cls, w_kd = schedule(299, 300)
    assert w_cls == pytest.approx(1.0 - 299 / (300 * 8))
    assert w_kd == pytest.approx(299 / (300 * 8))


def test_linear_ramp_kd_complements_linear_slow() -> None:
    slow = LinearRampSchedule(ramp="cls", ramp_epochs_factor=8.0)
    kd_ramp = LinearRampSchedule(ramp="kd", ramp_epochs_factor=8.0)
    for epoch in (0, 50, 150, 299):
        assert slow(epoch, 300) == kd_ramp(epoch, 300)[::-1]


def test_constant() -> None:
    schedule = ConstantSchedule()
    assert schedule(0, 300) == (1.0, 1.0)
    assert schedule(299, 300) == (1.0, 1.0)
