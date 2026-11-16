import numpy as np
import pytest

from thermalpipe.config import CameraConfig
from thermalpipe.optics import apply_optics, focal_plane_irradiance, mtf_at, total_blur_px
from thermalpipe.physics import RadianceLUT
from thermalpipe.scene import render_scene

CFG = CameraConfig()
LUT = RadianceLUT(CFG.band)


def test_ifov_is_120_microradians():
    assert CFG.ifov == pytest.approx(120e-6)


def test_person_size_in_pixels_follows_geometry():
    """A 1.8 m person at 1 km with 120 µrad IFOV must be 1.8 / 0.12 = 15 pixels tall."""
    scene = render_scene(CFG, LUT, distance_m=1000.0)
    rows = np.where(scene.masks["person"].any(axis=1))[0]
    assert len(rows) == pytest.approx(15, abs=1)


def test_objects_have_their_temperatures():
    scene = render_scene(CFG, LUT)
    assert np.all(scene.temperature[scene.masks["person"]] == 310.0)
    assert np.all(scene.temperature[scene.masks["engine"]] == 340.0)


def test_person_is_brighter_than_ground():
    scene = render_scene(CFG, LUT)
    assert scene.radiance[scene.masks["person"]].mean() > scene.radiance[scene.masks["ground"]].mean()


def test_farther_targets_have_less_contrast():
    def contrast(distance):
        s = render_scene(CFG, LUT, distance_m=distance)
        return s.radiance[s.masks["person"]].mean() - s.radiance[s.masks["road"]].mean()
    assert contrast(500.0) > contrast(2000.0) > 0


def test_camera_equation():
    """With tau = 1 and F = 1: E = pi · L / 5."""
    cfg = CameraConfig(optics_transmission=1.0, f_number=1.0)
    assert focal_plane_irradiance(10.0, cfg) == pytest.approx(np.pi * 10.0 / 5.0)


def test_blur_conserves_energy():
    image = np.zeros((64, 64))
    image[32, 32] = 1.0
    assert apply_optics(image, CFG).sum() == pytest.approx(focal_plane_irradiance(1.0, CFG), rel=1e-6)


def test_mtf_limits():
    assert mtf_at(0.0, CFG) == pytest.approx(1.0)
    assert 0.0 < mtf_at(0.5, CFG) < 1.0
    assert total_blur_px(CFG) > CFG.aberration_blur_px