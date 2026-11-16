import numpy as np
import pytest

from thermalpipe.config import CameraConfig
from thermalpipe.detector import Microbolometer
from thermalpipe.metrics import spatial_noise, temporal_noise
from thermalpipe.nuc import apply_nuc, calibrate_two_point, update_offsets_ffc
from thermalpipe.optics import apply_optics
from thermalpipe.physics import RadianceLUT
from thermalpipe.scene import flat_field

CAM = CameraConfig(n_rows=64, n_cols=80)
LUT = RadianceLUT(CAM.band)


def flat_frames(det, temperature, n=32, fpa_temp=None, settle=5):
    irradiance = apply_optics(flat_field(CAM, LUT, temperature), CAM)
    frames = [det.read(irradiance, fpa_temp=fpa_temp) for _ in range(settle + n)]
    return frames[settle:]          # drop frames while the bolometer settles


@pytest.fixture
def calibrated():
    det = Microbolometer(CAM, lut=LUT, seed=3)
    table = calibrate_two_point(flat_frames(det, 290.0), flat_frames(det, 320.0))
    return det, table


def test_nuc_flattens_a_third_temperature(calibrated):
    """Calibrated at 290 K and 320 K, the image of a 305 K blackbody must be flat."""
    det, table = calibrated
    raw = flat_frames(det, 305.0)
    corrected = [apply_nuc(f, table) for f in raw]
    good = ~table.bad_pixels
    assert spatial_noise(raw, good) > 100
    assert spatial_noise(corrected, good) < 0.5 * temporal_noise(corrected, good)


def test_nuc_finds_defective_pixels(calibrated):
    det, table = calibrated
    defects = det.dead_mask | det.hot_mask
    assert np.all(table.bad_pixels[defects])
    assert table.bad_pixels.mean() < 0.01        # and does not throw away good pixels


def test_drift_breaks_nuc_and_ffc_repairs_it(calibrated):
    det, table = calibrated
    good = ~table.bad_pixels
    warm = [apply_nuc(f, table) for f in flat_frames(det, 305.0, fpa_temp=305.0)]
    assert spatial_noise(warm, good) > 20          # camera warmed up 5 K: pattern is back

    shutter = flat_frames(det, 305.0, fpa_temp=305.0)
    table2 = update_offsets_ffc(table, shutter)
    fixed = [apply_nuc(f, table2) for f in flat_frames(det, 305.0, fpa_temp=305.0)]
    assert spatial_noise(fixed, good) < 0.5 * temporal_noise(fixed, good)