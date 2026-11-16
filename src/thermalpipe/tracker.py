"""Sub-pixel target tracker (template matching), the same idea as in Project A."""

import numpy as np
from scipy.signal import fftconvolve


def gaussian_template(sigma_px, half_size=4):
    """Template of a blurred point-like hot target."""
    y, x = np.mgrid[-half_size:half_size + 1, -half_size:half_size + 1]
    t = np.exp(-(x ** 2 + y ** 2) / (2 * sigma_px ** 2))
    return t - t.mean()          # zero mean: the background level does not matter


def _parabola_peak(left, mid, right):
    denom = left - 2 * mid + right
    return 0.0 if denom == 0 else 0.5 * (left - right) / denom


def locate(image, template):
    """Target position relative to the image centre [pixels] as (d_row, d_col)."""
    image = np.asarray(image, dtype=float)
    corr = fftconvolve(image - image.mean(), template[::-1, ::-1], mode="same")
    r, c = np.unravel_index(np.argmax(corr), corr.shape)
    r = int(np.clip(r, 1, corr.shape[0] - 2))
    c = int(np.clip(c, 1, corr.shape[1] - 2))
    dr = _parabola_peak(corr[r - 1, c], corr[r, c], corr[r + 1, c])
    dc = _parabola_peak(corr[r, c - 1], corr[r, c], corr[r, c + 1])
    return r + dr - image.shape[0] / 2, c + dc - image.shape[1] / 2