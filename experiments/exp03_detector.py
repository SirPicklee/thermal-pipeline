"""Experiment 3: what a real microbolometer does to the perfect image."""

import matplotlib.pyplot as plt
import numpy as np

from thermalpipe.config import CameraConfig
from thermalpipe.detector import Microbolometer
from thermalpipe.metrics import measured_netd, spatial_noise, temporal_noise
from thermalpipe.optics import apply_optics
from thermalpipe.physics import RadianceLUT
from thermalpipe.scene import flat_field, render_scene

cam = CameraConfig()
lut = RadianceLUT(cam.band)
det = Microbolometer(cam, lut=lut, seed=1)
good = ~(det.dead_mask | det.hot_mask)


def stretch(image):
    """Show only the middle 98 % of values (so a few hot pixels do not hide everything)."""
    lo, hi = np.percentile(image, [1, 99])
    return np.clip((image - lo) / (hi - lo), 0, 1)


# --- Lab measurement: look at flat blackbodies like a test engineer would ---
def flat_stack(temperature, n=50):
    irradiance = apply_optics(flat_field(cam, lut, temperature), cam)
    return np.array([det.read(irradiance) for _ in range(n)])


cold, hot = flat_stack(299.0), flat_stack(301.0)
print(f"Signal: {det.counts_per_kelvin():.1f} counts per kelvin")
print(f"Temporal noise : {temporal_noise(cold, good):6.2f} counts")
print(f"Fixed pattern  : {spatial_noise(cold, good):6.1f} counts "
      f"({spatial_noise(cold, good) / temporal_noise(cold, good):.0f}x the temporal noise)")
print(f"Measured NETD  : {1000 * measured_netd(cold, hot, 2.0, good):.1f} mK (designed: 50 mK)")
print(f"Defective pixels: {(~good).sum()} of {good.size}")

# --- The real scene ---
scene = render_scene(cam, lut, distance_m=1000.0)
ideal = apply_optics(scene.radiance, cam)
raw = det.read(ideal)

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0, 0].imshow(stretch(ideal), cmap="gray")
axes[0, 0].set_title("Ideal image (what the lens delivers)")
axes[0, 1].imshow(stretch(raw.astype(float)), cmap="gray")
axes[0, 1].set_title("Raw 14-bit detector output")
axes[1, 0].imshow(stretch(cold[0].astype(float)), cmap="gray")
axes[1, 0].set_title("Raw output looking at a FLAT 26 °C blackbody")
axes[1, 1].hist(raw[good].ravel(), bins=200, color="gray")
axes[1, 1].set_title("Histogram of raw counts (0 ... 16383)")
axes[1, 1].set_xlabel("Counts")
for ax in axes.flat[:3]:
    ax.axis("off")
plt.tight_layout()
plt.savefig("results/exp03_detector.png", dpi=150)
plt.show()