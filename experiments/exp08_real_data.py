"""Experiment 8: our pipeline on REAL thermal images (Teledyne FLIR ADAS v2, Tau 2 camera).

Run tools/fetch_flir_samples.py --get 20 --video 60 first.
"""

import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from thermalpipe import agc, filters
from thermalpipe.datasets import load_raw16

DATA = Path("data/flir")
raw_paths = sorted((DATA / "val" / "raw16").glob("*.tif*"))
video_paths = sorted((DATA / "video" / "raw16").glob("*.tif*"),
                     key=lambda p: int(p.name.split("-frame-")[1].split("-")[0]))
frames = [load_raw16(p).astype(float) for p in raw_paths]

# ---------------------------------------------------------------------------
# 1) What do the numbers mean? (the camera was in "T-linear" mode)
# ---------------------------------------------------------------------------
all_values = np.concatenate([f.ravel() for f in frames])
lo, med, hi = np.percentile(all_values, [1, 50, 99])
print(f"{len(frames)} frames of {frames[0].shape[1]} x {frames[0].shape[0]} pixels")
print(f"Raw values: 1 % = {lo:.0f}, median = {med:.0f}, 99 % = {hi:.0f}")
for name, kelvin_per_count in [("0.01 K / count", 0.01), ("0.04 K / count", 0.04)]:
    c = [v * kelvin_per_count - 273.15 for v in (lo, med, hi)]
    print(f"  if {name}: {c[0]:6.1f} °C ... {c[1]:6.1f} °C ... {c[2]:6.1f} °C")

# ---------------------------------------------------------------------------
# 2) Noise measured on real video: frame-to-frame differences
# ---------------------------------------------------------------------------
video = [load_raw16(p).astype(float) for p in video_paths]
diffs = np.concatenate([(b - a).ravel() for a, b in zip(video[:-1], video[1:])])
noise_counts = 1.4826 * np.median(np.abs(diffs - np.median(diffs))) / np.sqrt(2)
print(f"\nVideo: {len(video)} consecutive frames")
print(f"Temporal noise (robust, from frame differences): {noise_counts:.2f} counts "
      f"= {1000 * 0.04 * noise_counts:.0f} mK (T-linear: 0.04 K / count)")

# ---------------------------------------------------------------------------
# 3) Our AGCs vs. FLIR's own 8-bit images
# ---------------------------------------------------------------------------
examples = np.linspace(0, len(frames) - 1, 4).astype(int)
methods = {"Linear 1-99 %": agc.linear_percentile,
           "Plateau HE (ours)": agc.plateau_equalization,
           "Plateau HE + DDE (ours)": agc.dde}
fig, axes = plt.subplots(len(examples), 1 + len(methods), figsize=(16, 3.2 * len(examples)))
timings = {name: [] for name in methods}
for row, i in enumerate(examples):
    jpeg = DATA / "val" / "flir8" / (raw_paths[i].stem + ".jpg")
    flir = np.array(Image.open(jpeg).convert("L")) if jpeg.exists() else np.zeros_like(frames[i])
    axes[row, 0].imshow(flir, cmap="gray", vmin=0, vmax=255)
    axes[row, 0].set_title("FLIR's own 8-bit" if row == 0 else "")
    for col, (name, method) in enumerate(methods.items(), start=1):
        start = time.perf_counter()
        out = method(frames[i])
        timings[name].append(time.perf_counter() - start)
        axes[row, col].imshow(out, cmap="gray", vmin=0, vmax=255)
        axes[row, col].set_title(name if row == 0 else "")
for ax in axes.flat:
    ax.axis("off")
plt.tight_layout()
plt.savefig("results/exp08_real_agc.png", dpi=120)

print("\nAGC time per 640 x 512 frame:")
for name, t in timings.items():
    print(f"  {name:25s} {1000 * np.mean(t):7.1f} ms")

# ---------------------------------------------------------------------------
# 4) Temporal filter on real video: less noise, but moving objects smear
# ---------------------------------------------------------------------------
temporal = filters.TemporalFilter(alpha=0.25)
gif = []
filtered_diffs = []
previous = None
for frame in video:
    smooth = temporal(frame)
    if previous is not None:
        filtered_diffs.append((smooth - previous).ravel())
    previous = smooth
    left, right = agc.plateau_equalization(frame), agc.plateau_equalization(smooth)
    gif.append(Image.fromarray(np.hstack([left, np.zeros((frame.shape[0], 8), np.uint8), right])))
fd = np.concatenate(filtered_diffs[10:])
noise_filtered = 1.4826 * np.median(np.abs(fd - np.median(fd))) / np.sqrt(2)
print(f"\nTemporal filter (alpha 0.25) on real video: frame-to-frame flicker "
      f"{noise_counts:.2f} -> {noise_filtered:.2f} counts")
gif[0].save("results/exp08_real_video.gif", save_all=True, append_images=gif[1:], duration=33, loop=0)
print("Video saved: results/exp08_real_video.gif (left: no filter, right: temporal filter)")
plt.show()