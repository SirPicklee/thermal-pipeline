"""Experiment 6: AGC - turning 14-bit data into an image an operator can use."""

import matplotlib.pyplot as plt
import numpy as np

from thermalpipe import agc
from thermalpipe.config import CameraConfig
from thermalpipe.detector import Microbolometer
from thermalpipe.filters import replace_bad_pixels
from thermalpipe.nuc import apply_nuc, calibrate_two_point
from thermalpipe.optics import apply_optics
from thermalpipe.physics import RadianceLUT
from thermalpipe.scene import flat_field, render_scene

cam = CameraConfig()
lut = RadianceLUT(cam.band)
det = Microbolometer(cam, lut=lut, seed=1)


def flats(temperature, n=32, settle=5):
    irradiance = apply_optics(flat_field(cam, lut, temperature), cam)
    frames = [det.read(irradiance) for _ in range(settle + n)]
    return frames[settle:]


table = calibrate_two_point(flats(290.0), flats(320.0))
scene = render_scene(cam, lut, distance_m=1000.0)
m = scene.masks

# Scene B: same scene, plus a tiny but very hot exhaust pipe (600 K, 2 x 3 pixels)
radiance_b = scene.radiance.copy()
engine_rows, engine_cols = np.where(m["engine"])
r0, c0 = engine_rows.max() + 1, engine_cols.min() - 3
radiance_b[r0:r0 + 2, c0:c0 + 3] = lut.radiance(600.0)


def process(radiance):
    """Detector -> NUC -> bad pixels: the 14-bit image the AGC receives."""
    return replace_bad_pixels(apply_nuc(det.read(apply_optics(radiance, cam)), table),
                              table.bad_pixels)


methods = {
    "Linear min-max": agc.linear_minmax,
    "Linear 1-99 %": agc.linear_percentile,
    "Histogram eq.": agc.histogram_equalization,
    "Plateau HE": agc.plateau_equalization,
    "Plateau HE + DDE": agc.dde,
}

fig, axes = plt.subplots(2, len(methods), figsize=(18, 6.5))
for row, (label, radiance) in enumerate([("normal scene", scene.radiance),
                                         ("+ tiny 600 K exhaust", radiance_b)]):
    image = process(radiance)
    print(f"\n--- {label} ---")
    print(f"{'AGC':18s} {'Person vs road':>15s} {'Engine vs body':>15s} {'Ground texture':>15s}")
    for col, (name, method) in enumerate(methods.items()):
        out = method(image).astype(float)
        person = out[m["person"]].mean() - out[m["road"]].mean()
        engine = out[m["engine"]].mean() - out[m["vehicle"] & ~m["engine"]].mean()
        texture = out[m["ground"]].std()
        print(f"{name:18s} {person:12.0f} DL {engine:12.0f} DL {texture:12.0f} DL")
        ax = axes[row, col]
        ax.imshow(out, cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"{name}\n({label})", fontsize=10)
        ax.axis("off")
print("\nDL = display levels (0 = black, 255 = white). The eye needs roughly 5-10 DL to notice.")
plt.tight_layout()
plt.savefig("results/exp06_agc.png", dpi=150)
plt.show()