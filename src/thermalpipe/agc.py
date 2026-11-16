"""Automatic gain control (AGC): 14-bit sensor data -> 8-bit display image."""

import numpy as np

from thermalpipe.filters import bilateral


def to_8bit(image) -> np.ndarray:
    return np.clip(np.round(image), 0, 255).astype(np.uint8)


def _stretch(image, lo, hi):
    return to_8bit(255.0 * (np.asarray(image, dtype=float) - lo) / max(hi - lo, 1e-9))


def linear_minmax(image) -> np.ndarray:
    """Coldest pixel -> black, hottest pixel -> white."""
    return _stretch(image, np.min(image), np.max(image))


def linear_percentile(image, low=1.0, high=99.0) -> np.ndarray:
    """Like min-max, but ignores the coldest and hottest few percent of pixels."""
    lo, hi = np.percentile(image, [low, high])
    return _stretch(image, lo, hi)


def histogram_equalization(image, plateau=None, n_bins=2048) -> np.ndarray:
    """Give gray levels to values in proportion to how many pixels have them.

    plateau: limit for how many pixels one histogram bin may count.
             None = classic histogram equalization; small = close to linear.
    """
    image = np.asarray(image, dtype=float)
    hist, edges = np.histogram(image, bins=n_bins)
    hist = hist.astype(float)
    if plateau is not None:
        hist = np.minimum(hist, plateau)
    cdf = np.cumsum(hist)
    cdf = (cdf - cdf[0]) / max(cdf[-1] - cdf[0], 1e-9)
    centres = 0.5 * (edges[:-1] + edges[1:])
    return to_8bit(255.0 * np.interp(image, centres, cdf))


def plateau_equalization(image, plateau_fraction=0.5, n_bins=2048) -> np.ndarray:
    """Plateau histogram equalization, the standard AGC of thermal cameras.

    plateau_fraction scales the plateau relative to an evenly filled histogram.
    """
    plateau = plateau_fraction * np.size(image) / n_bins
    return histogram_equalization(image, plateau=plateau, n_bins=n_bins)


def dde(image, agc=plateau_equalization, detail_gain=3.0, sigma_space=2.0,
        sigma_range=40.0, radius=4) -> np.ndarray:
    """Digital detail enhancement: compress the big picture, boost the fine details."""
    image = np.asarray(image, dtype=float)
    base = bilateral(image, sigma_space=sigma_space, sigma_range=sigma_range, radius=radius)
    detail = image - base
    lo, hi = np.percentile(image, [1, 99])
    detail_display = detail * 255.0 / max(hi - lo, 1e-9)     # counts -> display levels
    return to_8bit(agc(base).astype(float) + detail_gain * detail_display)


class TemporalAGC:
    """Percentile AGC whose black and white levels change slowly from frame to frame."""

    def __init__(self, alpha=0.1, low=1.0, high=99.0):
        self.alpha, self.low, self.high = alpha, low, high
        self.levels = None

    def __call__(self, image) -> np.ndarray:
        target = np.percentile(image, [self.low, self.high])
        if self.levels is None:
            self.levels = target
        else:
            self.levels = self.levels + self.alpha * (target - self.levels)
        return _stretch(image, *self.levels)