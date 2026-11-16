import numpy as np
import pytest

from thermalpipe import agc

RNG = np.random.default_rng(0)
IMAGE = 6000 + 300 * RNG.standard_normal((64, 64)) ** 2       # skewed, like a real scene


@pytest.mark.parametrize("method", [agc.linear_minmax, agc.linear_percentile,
                                    agc.histogram_equalization, agc.plateau_equalization])
def test_hotter_is_never_darker(method):
    """An AGC may stretch or squeeze, but must keep the order of temperatures."""
    out = method(IMAGE)
    order = np.argsort(IMAGE.ravel())
    assert out.dtype == np.uint8
    assert np.all(np.diff(out.ravel()[order].astype(int)) >= 0)


def test_minmax_uses_full_range():
    out = agc.linear_minmax(IMAGE)
    assert out.min() == 0 and out.max() == 255


def test_one_hot_pixel_ruins_minmax_but_not_percentile():
    hot = IMAGE.copy()
    hot[0, 0] = 16383.0
    assert agc.linear_minmax(hot)[1:, 1:].std() < 0.5 * agc.linear_minmax(IMAGE).std()
    assert agc.linear_percentile(hot)[1:, 1:].std() == pytest.approx(
        agc.linear_percentile(IMAGE)[1:, 1:].std(), rel=0.05)


def test_equalization_flattens_the_histogram():
    linear_hist = np.histogram(agc.linear_minmax(IMAGE), bins=16, range=(0, 256))[0]
    eq_hist = np.histogram(agc.histogram_equalization(IMAGE), bins=16, range=(0, 256))[0]
    assert eq_hist.std() < 0.3 * linear_hist.std()


def test_huge_plateau_equals_classic_equalization():
    assert np.array_equal(agc.histogram_equalization(IMAGE, plateau=1e12),
                          agc.histogram_equalization(IMAGE))


def test_temporal_agc_changes_slowly():
    t_agc = agc.TemporalAGC(alpha=0.1)
    t_agc(IMAGE)
    hot_scene = IMAGE + 2000.0                                  # everything suddenly warmer
    first = t_agc(hot_scene).mean()
    for _ in range(60):
        last = t_agc(hot_scene).mean()
    assert first > last + 50          # first frame too bright, then it adapts