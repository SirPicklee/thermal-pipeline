"""Bad pixel replacement and noise reduction filters."""

import numpy as np
from scipy.ndimage import gaussian_filter, median_filter


def _neighbour_stack(image, radius):
    """All shifted copies of the image within a (2r+1) x (2r+1) window (edges repeated)."""
    padded = np.pad(image, radius, mode="edge")
    rows, cols = image.shape
    return np.stack([padded[radius + dy: radius + dy + rows, radius + dx: radius + dx + cols]
                     for dy in range(-radius, radius + 1)
                     for dx in range(-radius, radius + 1)])


def replace_bad_pixels(image, bad_pixels, radius=1):
    """Replace each bad pixel with the median of its GOOD neighbours.

    Only the few bad pixels are visited, not the whole image.
    """
    out = np.array(image, dtype=float)
    rows, cols = np.nonzero(bad_pixels)
    if rows.size == 0:
        return out
    steps = np.arange(-radius, radius + 1)
    d_row, d_col = np.meshgrid(steps, steps, indexing="ij")
    n_rows = np.clip(rows[:, None] + d_row.ravel(), 0, out.shape[0] - 1)   # (n_bad, n_neighbours)
    n_cols = np.clip(cols[:, None] + d_col.ravel(), 0, out.shape[1] - 1)
    values = np.where(bad_pixels[n_rows, n_cols], np.nan, out[n_rows, n_cols])
    with np.errstate(all="ignore"):
        medians = np.nanmedian(values, axis=1)
    fixable = np.isfinite(medians)
    out[rows[fixable], cols[fixable]] = medians[fixable]
    return out


def gaussian(image, sigma=1.0):
    """Weighted average of neighbours: removes noise, but also blurs edges."""
    return gaussian_filter(np.asarray(image, dtype=float), sigma=sigma, mode="nearest")


def median(image, size=3):
    """Median of the neighbourhood: good against spikes, keeps edges better."""
    return median_filter(np.asarray(image, dtype=float), size=size, mode="nearest")


def bilateral(image, sigma_space=1.5, sigma_range=6.0, radius=2):
    """Average only with neighbours that are close in position AND in value (keeps edges)."""
    image = np.asarray(image, dtype=float)
    values = _neighbour_stack(image, radius)
    offsets = np.array([(dy, dx) for dy in range(-radius, radius + 1)
                        for dx in range(-radius, radius + 1)], dtype=float)
    w_space = np.exp(-(offsets ** 2).sum(axis=1) / (2 * sigma_space ** 2))[:, None, None]
    w_range = np.exp(-(values - image) ** 2 / (2 * sigma_range ** 2))
    weights = w_space * w_range
    return (weights * values).sum(axis=0) / weights.sum(axis=0)


class TemporalFilter:
    """Recursive average over frames: y += alpha * (x - y). Small alpha = strong smoothing."""

    def __init__(self, alpha=0.25):
        self.alpha = alpha
        self.state = None

    def __call__(self, frame):
        frame = np.asarray(frame, dtype=float)
        if self.state is None:
            self.state = frame.copy()
        else:
            self.state += self.alpha * (frame - self.state)
        return self.state.copy()

    @property
    def noise_reduction(self) -> float:
        """How much white noise std is reduced in steady state."""
        return float(np.sqrt(self.alpha / (2 - self.alpha)))