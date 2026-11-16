"""Experiment 2: a synthetic thermal scene seen through the atmosphere and the lens."""

import matplotlib.pyplot as plt
import numpy as np

from thermalpipe.config import CameraConfig
from thermalpipe.optics import apply_optics, diffraction_blur_px, mtf_at, total_blur_px
from thermalpipe.physics import RadianceLUT
from thermalpipe.scene import render_scene

cfg = CameraConfig()
lut = RadianceLUT(cfg.band)

print(f"IFOV: {cfg.ifov * 1e6:.0f} µrad | field of view: "
      f"{np.degrees(cfg.n_cols * cfg.ifov):.2f}° x {np.degrees(cfg.n_rows * cfg.ifov):.2f}°")
print(f"Blur: diffraction {diffraction_blur_px(cfg):.2f} px, total {total_blur_px(cfg):.2f} px (sigma)")
print(f"Optical MTF at Nyquist (0.5 cycles/px): {float(mtf_at(0.5, cfg)):.2f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 8))

# 1) True temperatures at 1 km
scene = render_scene(cfg, lut, distance_m=1000.0)
im = axes[0, 0].imshow(scene.temperature - 273.15, cmap="inferno")
axes[0, 0].set_title("True temperature (°C)")
plt.colorbar(im, ax=axes[0, 0])

# 2) What reaches the detector: atmosphere + lens, at three distances
for ax, distance in zip([axes[0, 1], axes[1, 0], axes[1, 1]], [1000.0, 3000.0, 6000.0]):
    s = render_scene(cfg, lut, distance_m=distance)
    irradiance = apply_optics(s.radiance, cfg)
    person_px = s.masks["person"].any(axis=1).sum()
    contrast = s.radiance[s.masks["person"]].mean() - s.radiance[s.masks["road"]].mean()
    one_kelvin = lut.radiance(300.5) - lut.radiance(299.5)
    print(f"{distance / 1000:.0f} km: person is {person_px:2d} px tall, "
          f"apparent contrast vs road {contrast / one_kelvin:5.2f} K")
    im = ax.imshow(irradiance, cmap="gray")
    ax.set_title(f"Focal-plane irradiance at {distance / 1000:.0f} km (W/m²)")
    plt.colorbar(im, ax=ax)

for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
plt.savefig("results/exp02_scene.png", dpi=150)
plt.show()