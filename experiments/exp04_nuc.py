"""Experiment 4: two-point NUC, camera warm-up drift and shutter FFC."""

import matplotlib.pyplot as plt
import numpy as np

from thermalpipe.config import CameraConfig
from thermalpipe.detector import Microbolometer
from thermalpipe.metrics import spatial_noise, temporal_noise
from thermalpipe.nuc import apply_nuc, calibrate_two_point, update_offsets_ffc
from thermalpipe.optics import apply_optics
from thermalpipe.physics import RadianceLUT
from thermalpipe.scene import flat_field, render_scene

cam = CameraConfig()
lut = RadianceLUT(cam.band)
det = Microbolometer(cam, lut=lut, seed=1)
k_per_count = 1.0 / det.counts_per_kelvin()


def flats(temperature, n=32, fpa_temp=None, settle=5):
    """Look at a flat blackbody; drop the first frames while the bolometer settles."""
    irradiance = apply_optics(flat_field(cam, lut, temperature), cam)
    frames = [det.read(irradiance, fpa_temp=fpa_temp) for _ in range(settle + n)]
    return frames[settle:]


def stretch(image, good, reference=None):
    """Map the 1-99 % range of `reference` (default: the image itself) to black...white."""
    reference = image if reference is None else reference
    lo, hi = np.percentile(reference[good], [1, 99])
    return np.clip((image - lo) / (hi - lo), 0, 1)


# --- 1) Factory calibration: two blackbodies, camera at 300 K ---
table = calibrate_two_point(flats(290.0), flats(320.0))
good = ~table.bad_pixels
truly_bad = det.dead_mask | det.hot_mask
print(f"Bad pixels found: {table.bad_pixels.sum()} (real defects: {truly_bad.sum()}, "
      f"all found: {bool(np.all(table.bad_pixels[truly_bad]))})")

raw = flats(305.0)
corrected = [apply_nuc(f, table) for f in raw]
fpn_raw, fpn_nuc = spatial_noise(raw, good), spatial_noise(corrected, good)
print(f"Fixed pattern on a 305 K blackbody: raw {fpn_raw:6.1f} counts ({fpn_raw * k_per_count:5.2f} K)"
      f" -> after NUC {fpn_nuc:5.2f} counts ({1000 * fpn_nuc * k_per_count:4.1f} mK)")
print(f"Temporal noise for comparison: {temporal_noise(corrected, good):.2f} counts")

# --- 2) The camera warms up: offsets drift ---
fpa_temps = np.arange(300.0, 310.5, 1.0)
fpn_no_ffc, fpn_ffc = [], []
for fpa in fpa_temps:
    frames = flats(305.0, n=16, fpa_temp=fpa)
    fpn_no_ffc.append(spatial_noise([apply_nuc(f, table) for f in frames], good) * k_per_count * 1000)
    ffc_table = update_offsets_ffc(table, flats(fpa, n=16, fpa_temp=fpa))   # shutter at camera temp
    fpn_ffc.append(spatial_noise([apply_nuc(f, ffc_table) for f in frames], good) * k_per_count * 1000)
print(f"Camera warmed up by 6 K: residual pattern {fpn_no_ffc[6]:.0f} mK without FFC, "
      f"{fpn_ffc[6]:.0f} mK with FFC")

# --- 3) Real scene ---
scene = apply_optics(render_scene(cam, lut, distance_m=1000.0).radiance, cam)
scene_raw = det.read(scene).astype(float)
scene_nuc = apply_nuc(scene_raw, table)
scene_warm = apply_nuc(det.read(scene, fpa_temp=306.0), table)

fig, axes = plt.subplots(2, 3, figsize=(15, 8))
flat_raw = raw[0].astype(float)
flat_nuc = corrected[0] - np.median(corrected[0]) + np.median(flat_raw)
panels = [(flat_raw, flat_raw, "Flat 32 °C blackbody: RAW"),
          (flat_nuc, flat_raw, "Same frame after NUC (same gray scale)"),
          (scene_raw, None, "Scene: RAW"),
          (scene_nuc, None, "Scene: after NUC"),
          (scene_warm, None, "Scene: camera +6 K, no FFC")]
for ax, (img, ref, title) in zip(axes.flat, panels):
    ax.imshow(stretch(img, good, ref), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
ax = axes[1, 2]
ax.plot(fpa_temps - 300, fpn_no_ffc, "o-", label="Factory NUC only")
ax.plot(fpa_temps - 300, fpn_ffc, "s-", label="NUC + shutter FFC")
ax.axhline(50, color="r", linestyle="--", label="NETD (50 mK)")
ax.set_xlabel("Camera warm-up since calibration (K)")
ax.set_ylabel("Residual fixed pattern (mK)")
ax.set_title("Why thermal cameras 'click'")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/exp04_nuc.png", dpi=150)
plt.show()