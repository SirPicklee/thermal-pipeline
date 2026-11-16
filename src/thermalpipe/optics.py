"""Lens model: radiance at the lens -> irradiance on the detector, with blur."""

import numpy as np
from scipy.ndimage import gaussian_filter

from thermalpipe.config import CameraConfig


def focal_plane_irradiance(radiance, cfg: CameraConfig):
    """Camera equation: E = tau · pi · L / (1 + 4 F^2)   [W/m^2]."""
    return cfg.optics_transmission * np.pi * radiance / (1.0 + 4.0 * cfg.f_number ** 2)


def diffraction_blur_px(cfg: CameraConfig) -> float:
    """Gaussian approximation of the Airy spot: sigma ~ 0.42 · lambda · F  [pixels]."""
    return 0.42 * cfg.wavelength_center * cfg.f_number / cfg.pixel_pitch


def total_blur_px(cfg: CameraConfig) -> float:
    """Diffraction and aberrations combined (independent blurs add in quadrature)."""
    return float(np.hypot(diffraction_blur_px(cfg), cfg.aberration_blur_px))


def mtf_at(frequency_cyc_per_px, cfg: CameraConfig):
    """Optical MTF of the Gaussian blur at a spatial frequency [cycles/pixel]."""
    sigma = total_blur_px(cfg)
    return np.exp(-2.0 * np.pi ** 2 * sigma ** 2 * np.asarray(frequency_cyc_per_px) ** 2)


def apply_optics(radiance, cfg: CameraConfig):
    """Full lens model: blur the scene, then convert to focal-plane irradiance."""
    blurred = gaussian_filter(radiance, sigma=total_blur_px(cfg), mode="nearest")
    return focal_plane_irradiance(blurred, cfg)