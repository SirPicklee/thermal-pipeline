"""Blackbody radiation and atmospheric transmission."""

import numpy as np

# Physical constants (SI)
H = 6.62607015e-34       # Planck constant [J·s]
C = 2.99792458e8         # speed of light [m/s]
KB = 1.380649e-23        # Boltzmann constant [J/K]
SIGMA = 5.670374419e-8   # Stefan-Boltzmann constant [W/(m^2·K^4)]

LWIR = (8e-6, 14e-6)     # long-wave infrared band [m]
MWIR = (3e-6, 5e-6)      # mid-wave infrared band [m]


def planck_spectral_radiance(wavelength, temperature):
    """Blackbody spectral radiance L(lambda, T) [W / (m^2 · sr · m)].

    wavelength [m] and temperature [K] can be scalars or numpy arrays (broadcasting).
    """
    wavelength = np.asarray(wavelength, dtype=float)
    temperature = np.asarray(temperature, dtype=float)
    a = 2.0 * H * C ** 2 / wavelength ** 5
    b = H * C / (wavelength * KB * temperature)
    return a / np.expm1(b)


def band_radiance(temperature, band=LWIR, n_points=400):
    """In-band radiance [W / (m^2 · sr)]: Planck curve integrated over a wavelength band."""
    temperature = np.asarray(temperature, dtype=float)
    wavelengths = np.linspace(band[0], band[1], n_points)
    spectral = planck_spectral_radiance(wavelengths, temperature[..., None])
    return np.trapezoid(spectral, wavelengths, axis=-1)


class RadianceLUT:
    """Look-up table: temperature <-> in-band radiance.

    Integrating the Planck curve for every pixel of every frame is slow.
    Real cameras use a pre-computed table, and so do we.
    """

    def __init__(self, band=LWIR, t_min=200.0, t_max=1000.0, step=0.05):
        self.temperatures = np.arange(t_min, t_max + step, step)
        self.radiances = band_radiance(self.temperatures, band)

    def radiance(self, temperature):
        """Temperature [K] -> in-band radiance [W/(m^2·sr)]."""
        return np.interp(temperature, self.temperatures, self.radiances)

    def temperature(self, radiance):
        """In-band radiance [W/(m^2·sr)] -> temperature [K] (radiometry)."""
        return np.interp(radiance, self.radiances, self.temperatures)


def atmospheric_transmission(distance_m, extinction_per_km=0.2):
    """Beer-Lambert law: fraction of radiation that survives the path (0...1)."""
    return np.exp(-extinction_per_km * np.asarray(distance_m, dtype=float) / 1000.0)


def apparent_radiance(target_radiance, air_radiance, transmission):
    """What the camera sees: attenuated target + glow of the air in between."""
    return transmission * target_radiance + (1.0 - transmission) * air_radiance