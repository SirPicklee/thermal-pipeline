"""Experiment 5: bad pixels, noise filters and what they do to a sub-pixel tracker."""

import time

import matplotlib.pyplot as plt
import numpy as np

from thermalpipe import filters
from thermalpipe.config import CameraConfig
from thermalpipe.detector import Microbolometer
from thermalpipe.nuc import apply_nuc, calibrate_two_point
from thermalpipe.optics import apply_optics, total_blur_px
from thermalpipe.physics import RadianceLUT
from thermalpipe.scene import flat_field, point_target
from thermalpipe.tracker import gaussian_template, locate

cam = CameraConfig(n_rows=96, n_cols=96)      # small window around the target
lut = RadianceLUT(cam.band)
TARGET_SIGMA = 0.8                            # target size [px]
TARGET_DT = 0.5                               # apparent temperature difference [K] (far away)
template = gaussian_template(np.hypot(TARGET_SIGMA, total_blur_px(cam)))
N_DETECTORS, N_POSITIONS = 20, 8


def flats(det, temperature, n=32, settle=5):
    irradiance = apply_optics(flat_field(cam, lut, temperature), cam)
    frames = [det.read(irradiance) for _ in range(settle + n)]
    return frames[settle:]


def pipelines(raw, table, temporal):
    """Every processing option we want to compare, for one raw frame."""
    nuc = apply_nuc(raw, table)
    fixed = filters.replace_bad_pixels(nuc, table.bad_pixels)
    return {
        "Raw (no NUC)": raw.astype(float),
        "NUC": nuc,
        "NUC + bad pixel": fixed,
        "+ Gaussian": filters.gaussian(fixed, 1.0),
        "+ Median 3x3": filters.median(fixed, 3),
        "+ Bilateral": filters.bilateral(fixed),
        "+ Temporal": temporal(fixed),
    }


errors, noise_mk, examples = {}, {}, {}
start = time.time()
for s in range(N_DETECTORS):
    det = Microbolometer(cam, lut=lut, seed=100 + s)
    table = calibrate_two_point(flats(det, 290.0), flats(det, 320.0))
    good = ~table.bad_pixels
    rng = np.random.default_rng(s)

    # Noise on a flat scene after each option [mK]
    temporal = filters.TemporalFilter(alpha=0.25)
    for frame in flats(det, 293.0, n=12):
        outputs = pipelines(frame, table, temporal)
    if s == 0:
        for name, img in outputs.items():
            noise_mk[name] = 1000 * np.std(img[good]) / det.counts_per_kelvin()

    # Tracking accuracy on a small, faint, static target
    for _ in range(N_POSITIONS):
        offset = rng.uniform(-3, 3, size=2)
        irradiance = apply_optics(point_target(cam, lut, offset, TARGET_DT, TARGET_SIGMA), cam)
        temporal = filters.TemporalFilter(alpha=0.25)
        for _ in range(12):                                  # let the temporal filter settle
            outputs = pipelines(det.read(irradiance), table, temporal)
        for name, img in outputs.items():
            r, c = locate(img, template)
            errors.setdefault(name, []).append(np.hypot(r - offset[0], c - offset[1]) * cam.ifov * 1e6)
        if s == 0 and not examples:
            examples = {name: img for name, img in outputs.items()}
print(f"({N_DETECTORS * N_POSITIONS} tracking trials in {time.time() - start:.0f} s)\n")

print(f"{'Processing':18s} {'Noise':>9s} {'Tracker RMS':>12s} {'Lost target':>12s}")
names = list(errors)
rms = []
for name in names:
    e = np.array(errors[name])
    ok = e < 120.0                                          # more than 1 pixel off = lost
    rms.append(np.sqrt(np.mean(e[ok] ** 2)) if ok.any() else np.nan)
    tracker = f"{rms[-1]:7.1f} µrad" if ok.any() else "      -     "
    print(f"{name:18s} {noise_mk[name]:6.0f} mK {tracker:>12s} {100 * (1 - ok.mean()):10.1f} %")

fig = plt.figure(figsize=(15, 6))
for i, name in enumerate(names):
    ax = fig.add_subplot(2, 7, i + 1)
    crop = examples[name][32:64, 32:64]
    lo, hi = np.percentile(crop, [1, 99.5])
    ax.imshow(crop, cmap="gray", vmin=lo, vmax=hi)
    ax.set_title(name, fontsize=9)
    ax.axis("off")
ax = fig.add_subplot(2, 1, 2)
ax.bar(names, rms, color=["gray"] * 3 + ["tab:blue"] * 3 + ["tab:green"])
ax.set_ylabel("Tracker error (µrad RMS)")
ax.set_title(f"Faint target ({TARGET_DT} K): which processing helps the tracker?")
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("results/exp05_filters.png", dpi=150)
plt.show()