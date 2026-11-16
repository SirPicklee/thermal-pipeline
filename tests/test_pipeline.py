import numpy as np
import pytest

from thermalpipe import filters
from thermalpipe.config import CameraConfig
from thermalpipe.detector import Microbolometer
from thermalpipe.nuc import apply_nuc, calibrate_two_point
from thermalpipe.optics import apply_optics
from thermalpipe.physics import RadianceLUT
from thermalpipe.pipeline import PipelineConfig, ThermalPipeline
from thermalpipe.scene import flat_field, render_scene

CAM = CameraConfig(n_rows=64, n_cols=80)
LUT = RadianceLUT(CAM.band)


@pytest.fixture
def setup():
    det = Microbolometer(CAM, lut=LUT, seed=5)

    def flats(t, n=16, settle=5):
        irradiance = apply_optics(flat_field(CAM, LUT, t), CAM)
        return [det.read(irradiance) for _ in range(settle + n)][settle:]

    table = calibrate_two_point(flats(290.0), flats(320.0))
    raw = det.read(apply_optics(render_scene(CAM, LUT).radiance, CAM))
    return table, raw


def test_pipeline_outputs(setup):
    table, raw = setup
    data, display = ThermalPipeline(table).process(raw)
    assert data.shape == display.shape == raw.shape
    assert data.dtype == float and display.dtype == np.uint8


def test_pipeline_equals_manual_chain(setup):
    table, raw = setup
    data, _ = ThermalPipeline(table).process(raw)
    expected = filters.replace_bad_pixels(apply_nuc(raw, table), table.bad_pixels)
    assert np.allclose(data, expected)


def test_pipeline_reports_speed(setup):
    table, raw = setup
    pipe = ThermalPipeline(table, PipelineConfig(temporal_alpha=0.5))
    for _ in range(3):
        pipe.process(raw)
    report = pipe.report()
    assert report["FPS"] > 0 and "Temporal filter" in report


def test_temporal_filter_lags_a_moving_target():
    """Theory: a recursive filter follows a ramp with a delay of (1 - alpha) / alpha frames."""
    alpha, speed = 0.25, 0.5
    tf = filters.TemporalFilter(alpha)
    for k in range(200):
        out = tf(np.array([[speed * k]]))
    lag_frames = (speed * k - out[0, 0]) / speed
    assert lag_frames == pytest.approx((1 - alpha) / alpha, rel=1e-3)