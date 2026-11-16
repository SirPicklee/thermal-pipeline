import numpy as np
import pytest

from thermalpipe.config import CameraConfig
from thermalpipe.detector import DetectorConfig, Microbolometer
from thermalpipe.metrics import measured_netd, spatial_noise, temporal_noise
from thermalpipe.optics import apply_optics
from thermalpipe.physics import RadianceLUT
from thermalpipe.scene import flat_field

CAM = CameraConfig(n_rows=64, n_cols=80)
LUT = RadianceLUT(CAM.band)


def flat_frames(detector, temperature, n=30):
    irradiance = apply_optics(flat_field(CAM, LUT, temperature), CAM)
    return np.array([detector.read(irradiance) for _ in range(n)])


def test_output_is_14_bit():
    frames = flat_frames(Microbolometer(CAM, lut=LUT), 300.0, n=3)
    assert frames.dtype == np.uint16
    assert frames.max() <= 2 ** 14 - 1


def test_same_seed_same_detector():
    a, b = Microbolometer(CAM, lut=LUT, seed=7), Microbolometer(CAM, lut=LUT, seed=7)
    assert np.array_equal(a.offset, b.offset)


def test_fixed_pattern_dominates_raw_image():
    det = Microbolometer(CAM, lut=LUT)
    frames = flat_frames(det, 300.0)
    good = ~(det.dead_mask | det.hot_mask)
    assert spatial_noise(frames, good) > 20 * temporal_noise(frames, good)


def test_measured_netd_matches_specification():
    """Validation: measuring NETD the way a lab does must give back the 50 mK we designed."""
    det = Microbolometer(CAM, DetectorConfig(one_over_f_std=0.0), lut=LUT)
    good = ~(det.dead_mask | det.hot_mask)
    netd = measured_netd(flat_frames(det, 299.0, 60), flat_frames(det, 301.0, 60), 2.0, good)
    assert netd == pytest.approx(0.050, rel=0.1)


def test_dead_pixel_fraction():
    det = Microbolometer(CameraConfig(), lut=LUT)
    fraction = (det.dead_mask | det.hot_mask).mean()
    assert fraction == pytest.approx(0.001, rel=0.5)   # random, so allow some spread


def test_thermal_lag():
    """After one frame the bolometer has only partly followed a step in irradiance."""
    det = Microbolometer(CAM, DetectorConfig(netd=1e-6, one_over_f_std=0.0), lut=LUT)
    cold = apply_optics(flat_field(CAM, LUT, 290.0), CAM)
    hot = apply_optics(flat_field(CAM, LUT, 310.0), CAM)
    good = ~(det.dead_mask | det.hot_mask)
    c0 = det.read(cold)[good].astype(float).mean()
    c1 = det.read(hot)[good].astype(float).mean()
    for _ in range(20):
        c_end = det.read(hot)[good].astype(float).mean()
    fraction = (c1 - c0) / (c_end - c0)
    assert 0.5 < fraction < 0.99