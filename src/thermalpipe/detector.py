"""Uncooled microbolometer model: turns focal-plane irradiance into raw 14-bit counts."""

from dataclasses import dataclass

import numpy as np

from thermalpipe.config import CameraConfig
from thermalpipe.optics import focal_plane_irradiance
from thermalpipe.physics import RadianceLUT


@dataclass(frozen=True)
class DetectorConfig:
    """Imperfections of a typical uncooled VOx microbolometer."""

    fill_factor: float = 0.8              # fraction of the pixel area that absorbs light
    responsivity: float = 1.0e12          # counts per watt (readout + ADC)
    global_offset: float = 4000.0         # counts added by the readout electronics
    netd: float = 0.050                   # temporal noise expressed in kelvin [K]
    gain_nonuniformity: float = 0.05      # pixel-to-pixel gain spread (std, relative)
    offset_nonuniformity: float = 300.0   # pixel-to-pixel offset spread (std) [counts]
    column_offset_std: float = 80.0       # column stripes from shared readout circuits [counts]
    dead_pixel_fraction: float = 0.001    # stuck or hot pixels
    drift_counts_per_k: float = 40.0      # offset change per kelvin of camera temperature
    drift_nonuniformity: float = 0.3      # how differently pixels drift (relative)
    one_over_f_std: float = 0.5           # slow random offset wander, in units of temporal noise
    one_over_f_corr: float = 0.995        # frame-to-frame correlation of that wander
    time_constant: float = 0.010          # bolometer thermal time constant [s]
    frame_rate: float = 50.0              # frames per second [Hz]
    adc_bits: int = 14


class Microbolometer:
    """A detector with its own fixed imperfections and internal state (thermal lag, 1/f)."""

    def __init__(self, cam: CameraConfig, det: DetectorConfig = DetectorConfig(),
                 lut: RadianceLUT | None = None, seed: int = 0, calib_fpa_temp: float = 300.0):
        self.cam, self.det = cam, det
        self.lut = lut if lut is not None else RadianceLUT(cam.band)
        self.rng = np.random.default_rng(seed)
        self.calib_fpa_temp = calib_fpa_temp
        shape = (cam.n_rows, cam.n_cols)

        # Fixed pattern: every pixel is a little different, forever
        self.gain = 1.0 + det.gain_nonuniformity * self.rng.standard_normal(shape)
        self.offset = (det.global_offset
                       + det.offset_nonuniformity * self.rng.standard_normal(shape)
                       + det.column_offset_std * self.rng.standard_normal((1, cam.n_cols)))
        self.drift = det.drift_counts_per_k * (1.0 + det.drift_nonuniformity
                                               * self.rng.standard_normal(shape))

        # Defective pixels: half stuck (no response), half hot (always saturated)
        defects = self.rng.random(shape) < det.dead_pixel_fraction
        hot = self.rng.random(shape) < 0.5
        self.dead_mask = defects & ~hot
        self.hot_mask = defects & hot
        self.gain[self.dead_mask] = 0.0

        # Temporal noise level chosen so that the detector meets its NETD specification
        self.noise_counts = det.netd * self.counts_per_kelvin()

        # Internal state
        self._absorbed = None
        self._wander = np.zeros(shape)
        self._alpha = 1.0 - np.exp(-1.0 / (det.frame_rate * det.time_constant))

    def pixel_power(self, irradiance):
        """Optical power absorbed by each pixel [W]."""
        return irradiance * self.det.fill_factor * self.cam.pixel_pitch ** 2

    def counts_per_kelvin(self, scene_temp=300.0) -> float:
        """Signal change (counts) of an average pixel for +1 K of scene temperature."""
        d_radiance = self.lut.radiance(scene_temp + 0.5) - self.lut.radiance(scene_temp - 0.5)
        d_power = self.pixel_power(focal_plane_irradiance(d_radiance, self.cam))
        return float(self.det.responsivity * d_power)

    def read(self, irradiance, fpa_temp: float | None = None) -> np.ndarray:
        """One frame: irradiance [W/m^2] -> raw counts (uint16, 0 ... 2^bits - 1)."""
        power = self.pixel_power(irradiance)

        # Thermal lag: the membrane only moves part of the way towards the new power
        if self._absorbed is None:
            self._absorbed = power.copy()
        else:
            self._absorbed += self._alpha * (power - self._absorbed)

        # Offsets drift when the camera itself warms up or cools down
        fpa_temp = self.calib_fpa_temp if fpa_temp is None else fpa_temp
        offset = self.offset + self.drift * (fpa_temp - self.calib_fpa_temp)

        # Slow 1/f wander (AR(1) process) + white temporal noise
        rho = self.det.one_over_f_corr
        self._wander = (rho * self._wander + np.sqrt(1 - rho ** 2) * self.det.one_over_f_std
                        * self.noise_counts * self.rng.standard_normal(self._wander.shape))
        noise = self.noise_counts * self.rng.standard_normal(power.shape)

        counts = self.gain * self.det.responsivity * self._absorbed + offset + self._wander + noise
        max_count = 2 ** self.det.adc_bits - 1
        counts[self.hot_mask] = max_count
        return np.clip(np.round(counts), 0, max_count).astype(np.uint16)