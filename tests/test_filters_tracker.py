import numpy as np
import pytest

from thermalpipe import filters
from thermalpipe.tracker import gaussian_template, locate

RNG = np.random.default_rng(0)


def test_bad_pixel_gets_median_of_good_neighbours():
    image = np.full((5, 5), 100.0)
    image[2, 2] = 16383.0                      # hot pixel
    bad = np.zeros((5, 5), dtype=bool)
    bad[2, 2] = True
    fixed = filters.replace_bad_pixels(image, bad)
    assert fixed[2, 2] == 100.0
    assert np.array_equal(fixed[~bad], image[~bad])     # good pixels untouched


@pytest.mark.parametrize("smoother", [
    lambda im: filters.gaussian(im, 1.0),
    lambda im: filters.median(im, 3),
    lambda im: filters.bilateral(im, sigma_range=6.0),
])
def test_spatial_filters_reduce_noise(smoother):
    noise = 2.0 * RNG.standard_normal((64, 64))
    assert smoother(noise).std() < 0.7 * noise.std()


def test_bilateral_keeps_edges_better_than_gaussian():
    step = np.zeros((32, 32))
    step[:, 16:] = 100.0                       # a sharp edge, much larger than sigma_range
    edge_gauss = filters.gaussian(step, 1.5)[:, 16] - filters.gaussian(step, 1.5)[:, 15]
    edge_bilat = filters.bilateral(step)[:, 16] - filters.bilateral(step)[:, 15]
    assert edge_bilat.mean() > 1.5 * edge_gauss.mean()


def test_temporal_filter_noise_reduction():
    tf = filters.TemporalFilter(alpha=0.25)
    outputs = [tf(2.0 * RNG.standard_normal((32, 32))) for _ in range(300)]
    measured = np.std(outputs[50:]) / 2.0
    assert measured == pytest.approx(tf.noise_reduction, rel=0.15)


def test_tracker_finds_sub_pixel_position():
    rows, cols = np.mgrid[0:48, 0:48]
    true_r, true_c = 1.3, -2.6
    blob = np.exp(-((rows - 24 - true_r) ** 2 + (cols - 24 - true_c) ** 2) / (2 * 1.2 ** 2))
    r, c = locate(blob, gaussian_template(1.2))
    assert r == pytest.approx(true_r, abs=0.05)
    assert c == pytest.approx(true_c, abs=0.05)



def _replace_bad_pixels_reference(image, bad_pixels, radius=1):
    """The first, slow version (median for EVERY pixel), kept as a reference."""
    padded = np.pad(np.asarray(image, float), radius, mode="edge")
    padded_bad = np.pad(bad_pixels, radius, mode="edge")
    out = np.array(image, dtype=float)
    for r, c in zip(*np.nonzero(bad_pixels)):
        window = padded[r:r + 2 * radius + 1, c:c + 2 * radius + 1]
        ok = ~padded_bad[r:r + 2 * radius + 1, c:c + 2 * radius + 1]
        if ok.any():
            out[r, c] = np.median(window[ok])
    return out


def test_fast_bad_pixel_replacement_matches_reference():
    image = RNG.normal(6000, 50, (120, 160))
    bad = RNG.random((120, 160)) < 0.01
    bad[0, 0] = bad[0, 1] = bad[1, 0] = True               # a cluster in the corner
    image[bad] = 16383.0
    assert np.allclose(filters.replace_bad_pixels(image, bad),
                       _replace_bad_pixels_reference(image, bad))