"""Non-uniformity correction (NUC): two-point calibration and shutter-based offset update."""

from dataclasses import dataclass

import numpy as np


@dataclass
class NUCTable:
    """Per-pixel correction: corrected = gain * raw + offset."""

    gain: np.ndarray
    offset: np.ndarray
    bad_pixels: np.ndarray       # True where the pixel cannot be trusted


def _robust_outliers(values, good, n_sigma=6.0):
    """Pixels far from the median, measured with the median absolute deviation (MAD)."""
    median = np.median(values[good])
    mad = 1.4826 * np.median(np.abs(values[good] - median))
    return np.abs(values - median) > n_sigma * mad


def calibrate_two_point(frames_cold, frames_hot, max_count=2 ** 14 - 1) -> NUCTable:
    """Two flat blackbodies (cold and hot) -> per-pixel gain and offset."""
    y_cold = np.mean(np.asarray(frames_cold, dtype=float), axis=0)
    y_hot = np.mean(np.asarray(frames_hot, dtype=float), axis=0)
    response = y_hot - y_cold

    # Pixels that do not respond or are saturated are bad
    bad = (response <= 0) | (y_hot >= max_count) | (y_cold <= 0)
    bad |= _robust_outliers(response, ~bad)

    good = ~bad
    mu_cold, mu_hot = np.median(y_cold[good]), np.median(y_hot[good])

    gain = np.ones_like(y_cold)
    gain[good] = (mu_hot - mu_cold) / response[good]
    offset = mu_cold - gain * y_cold
    gain[bad], offset[bad] = 0.0, mu_cold          # bad pixels: replaced later (filters.py)
    return NUCTable(gain=gain, offset=offset, bad_pixels=bad)


def apply_nuc(raw, table: NUCTable) -> np.ndarray:
    """Correct one raw frame (returns float counts)."""
    return table.gain * np.asarray(raw, dtype=float) + table.offset


def update_offsets_ffc(table: NUCTable, shutter_frames) -> NUCTable:
    """Flat Field Correction: look at the closed shutter and re-flatten the offsets.

    Gains are kept (they hardly change); only offsets are re-measured, because they drift
    when the camera temperature changes.
    """
    corrected = np.mean([apply_nuc(f, table) for f in shutter_frames], axis=0)
    good = ~table.bad_pixels
    new_offset = table.offset + (np.median(corrected[good]) - corrected)
    new_offset[table.bad_pixels] = table.offset[table.bad_pixels]
    return NUCTable(gain=table.gain, offset=new_offset, bad_pixels=table.bad_pixels)