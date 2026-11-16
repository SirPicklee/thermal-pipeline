import numpy as np
import pytest

from thermalpipe.physics import (
    LWIR, SIGMA, RadianceLUT, apparent_radiance, atmospheric_transmission,
    band_radiance, planck_spectral_radiance,
)


def test_wien_peak_at_room_temperature():
    """The Planck curve at 300 K must peak at 2898 / 300 = 9.66 µm (Wien's law)."""
    wavelengths = np.linspace(1e-6, 30e-6, 30000)
    peak = wavelengths[np.argmax(planck_spectral_radiance(wavelengths, 300.0))]
    assert peak == pytest.approx(2897.77e-6 / 300.0, rel=1e-3)


def test_total_radiance_matches_stefan_boltzmann():
    """Integrating over (almost) all wavelengths must give sigma·T^4 / pi."""
    total = band_radiance(300.0, band=(0.5e-6, 1000e-6), n_points=200000)
    assert total == pytest.approx(SIGMA * 300.0 ** 4 / np.pi, rel=1e-2)


def test_hotter_is_brighter():
    radiances = band_radiance(np.array([280.0, 300.0, 320.0]), LWIR)
    assert np.all(np.diff(radiances) > 0)


def test_lut_matches_direct_integration():
    lut = RadianceLUT(LWIR)
    temps = np.array([250.0, 300.0, 310.3, 600.0])
    assert lut.radiance(temps) == pytest.approx(band_radiance(temps, LWIR), rel=1e-5)


def test_lut_round_trip():
    lut = RadianceLUT(LWIR)
    temps = np.array([273.15, 300.0, 310.0])
    assert lut.temperature(lut.radiance(temps)) == pytest.approx(temps, abs=1e-3)


def test_atmosphere_reduces_contrast_with_distance():
    lut = RadianceLUT(LWIR)
    target, background = lut.radiance(310.0), lut.radiance(300.0)
    air = lut.radiance(295.0)
    contrasts = []
    for distance in [0, 1000, 5000]:
        tau = atmospheric_transmission(distance)
        contrasts.append(apparent_radiance(target, air, tau) - apparent_radiance(background, air, tau))
    assert atmospheric_transmission(0) == 1.0
    assert contrasts[0] > contrasts[1] > contrasts[2] > 0