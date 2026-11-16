import pytest

from thermalpipe.config import CameraConfig
from thermalpipe.metrics import apparent_delta_t, geometric_range, task_range
from thermalpipe.physics import RadianceLUT

CAM = CameraConfig()
LUT = RadianceLUT(CAM.band)


def test_geometric_johnson_range():
    """2.3 m vehicle, 1 cycle = 2 pixels, IFOV 120 µrad -> 2.3 / 240e-6 = 9583 m."""
    assert geometric_range("detection", 2.3, CAM) == pytest.approx(9583, rel=1e-3)


def test_apparent_contrast_at_zero_distance():
    assert apparent_delta_t(0.0, 1.0, LUT) == pytest.approx(1.0, rel=0.03)


def test_range_never_beats_geometry():
    for task in ["detection", "recognition", "identification"]:
        assert task_range(task, 0.75, 8.0, CAM, LUT) <= geometric_range(task, 0.75, CAM) + 10


def test_haze_never_helps_and_less_noise_never_hurts():
    clear = task_range("detection", 0.75, 1.0, CAM, LUT, extinction_per_km=0.1)
    haze = task_range("detection", 0.75, 1.0, CAM, LUT, extinction_per_km=0.5)
    quiet = task_range("detection", 0.75, 1.0, CAM, LUT, extinction_per_km=0.5, netd=0.02)
    assert haze <= clear
    assert quiet >= haze