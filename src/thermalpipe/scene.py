"""Synthetic thermal scenes: background, sky, people and vehicles."""

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

from thermalpipe.config import CameraConfig
from thermalpipe.physics import RadianceLUT, apparent_radiance, atmospheric_transmission


@dataclass
class Scene:
    """Result of rendering: what is out there and what reaches the lens."""

    temperature: np.ndarray            # true temperature of every pixel [K]
    radiance: np.ndarray               # apparent in-band radiance at the lens [W/(m^2·sr)]
    masks: dict = field(default_factory=dict)   # where each object is (True/False images)


def _box(cfg, center_row, center_col, height_m, width_m, distance_m):
    """Boolean mask of a rectangle with real-world size [m] seen at a distance."""
    h_px = height_m / (distance_m * cfg.ifov)
    w_px = width_m / (distance_m * cfg.ifov)
    rows, cols = np.mgrid[0:cfg.n_rows, 0:cfg.n_cols]
    return (np.abs(rows - center_row) < h_px / 2) & (np.abs(cols - center_col) < w_px / 2)


def render_scene(cfg: CameraConfig, lut: RadianceLUT, distance_m=1000.0,
                 extinction_per_km=0.2, air_temp=293.0, seed=0, person_shift_m=0.0) -> Scene:
    """Sky, textured ground, a road, a person and a vehicle with a hot engine."""
    rng = np.random.default_rng(seed)
    rows = np.arange(cfg.n_rows)[:, None] * np.ones((1, cfg.n_cols))
    horizon = int(0.35 * cfg.n_rows)

    # Ground: 293 K with smooth random texture (+-1.5 K), slightly warmer near the camera
    texture = gaussian_filter(rng.standard_normal((cfg.n_rows, cfg.n_cols)), sigma=6)
    texture *= 1.5 / texture.std()
    temp = 293.0 + texture + 2.0 * (rows - horizon) / cfg.n_rows

    # Road: a warm asphalt band
    ground_row = 0.75 * cfg.n_rows
    road = np.abs(rows - ground_row) < 0.06 * cfg.n_rows
    temp[road] = 298.0 + 0.3 * texture[road]

    # Objects standing on the road
    person_col = 0.35 * cfg.n_cols + person_shift_m / (distance_m * cfg.ifov)
    person = _box(cfg, ground_row - 0.9 / (distance_m * cfg.ifov), person_col,
                  1.8, 0.5, distance_m)
    vehicle = _box(cfg, ground_row - 0.75 / (distance_m * cfg.ifov), 0.65 * cfg.n_cols,
                   1.5, 4.5, distance_m)
    engine = vehicle & _box(cfg, ground_row - 0.75 / (distance_m * cfg.ifov),
                            0.65 * cfg.n_cols - 1.6 / (distance_m * cfg.ifov),
                            0.8, 1.2, distance_m)
    temp[person] = 310.0
    temp[vehicle] = 302.0
    temp[engine] = 340.0

    # Atmosphere: everything below the horizon is seen through distance_m of air
    tau = atmospheric_transmission(distance_m, extinction_per_km)
    radiance = apparent_radiance(lut.radiance(temp), lut.radiance(air_temp), tau)

    # Sky: cold, getting colder upwards (no atmosphere correction: it IS the atmosphere)
    sky = rows < horizon
    temp[sky] = 250.0 + 25.0 * rows[sky] / horizon
    radiance[sky] = lut.radiance(temp[sky])

    masks = {"person": person, "vehicle": vehicle, "engine": engine,
             "road": road, "sky": sky, "ground": ~sky & ~road & ~person & ~vehicle}
    return Scene(temperature=temp, radiance=radiance, masks=masks)



def flat_field(cfg: CameraConfig, lut: RadianceLUT, temperature: float) -> np.ndarray:
    """Uniform blackbody filling the whole view (used for calibration)."""
    return np.full((cfg.n_rows, cfg.n_cols), float(lut.radiance(temperature)))



def point_target(cfg: CameraConfig, lut: RadianceLUT, offset_px=(0.0, 0.0),
                 delta_t=2.0, sigma_px=1.0, background_temp=293.0) -> np.ndarray:
    """A small distant hot target (blob) on a uniform background, at a sub-pixel position."""
    rows, cols = np.mgrid[0:cfg.n_rows, 0:cfg.n_cols]
    r0 = cfg.n_rows / 2 + offset_px[0]
    c0 = cfg.n_cols / 2 + offset_px[1]
    blob = np.exp(-((rows - r0) ** 2 + (cols - c0) ** 2) / (2 * sigma_px ** 2))
    return lut.radiance(background_temp + delta_t * blob)