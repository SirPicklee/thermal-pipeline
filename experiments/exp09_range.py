"""Experiment 9: how far can this camera see a person? What limits it, and what helps?"""

import matplotlib.pyplot as plt
import numpy as np

from thermalpipe.config import CameraConfig
from thermalpipe.metrics import (JOHNSON_CYCLES, apparent_delta_t, geometric_range,
                                 required_delta_t, task_range)
from thermalpipe.physics import RadianceLUT

cam = CameraConfig()
lut = RadianceLUT(cam.band)
PERSON = 0.75                      # critical size of a person [m] (standard value)
NETD = 0.050                       # detector, as designed in step 3 (42 mK measured on real data)
NETD_TEMPORAL = NETD * 0.38        # with the temporal filter (alpha 0.25), slow targets only

print(f"Pure geometry (Johnson), person {PERSON} m, IFOV {cam.ifov * 1e6:.0f} µrad:")
for task in JOHNSON_CYCLES:
    print(f"  {task:15s}: {geometric_range(task, PERSON, cam) / 1000:5.2f} km")

scenarios = [
    ("Normal day, person 8 K warmer", 8.0),
    ("Thermal crossover, person 1 K warmer", 1.0),
]
options = [
    ("Our camera", cam, NETD),
    ("+ temporal filter", cam, NETD_TEMPORAL),
    ("+ 300 mm lens instead", CameraConfig(focal_length=0.3), NETD),
]
for title, dt0 in scenarios:
    print(f"\n{title}: DETECTION range (km)   clear / average / haze")
    for name, c, netd in options:
        r = [task_range("detection", PERSON, dt0, c, lut, netd, e) / 1000 for e in (0.1, 0.2, 0.5)]
        print(f"  {name:24s} {r[0]:6.2f} {r[1]:7.2f} {r[2]:7.2f}")

# --- Classic range-performance chart: what the target offers vs. what the camera needs ---
ranges = np.linspace(100, 6000, 600)
pixels = PERSON / (ranges * cam.ifov)
freq = JOHNSON_CYCLES["detection"] / pixels
fig, ax = plt.subplots(figsize=(9, 5.5))
for dt0, style in [(8.0, "-"), (1.0, "--")]:
    for ext, color in [(0.1, "tab:blue"), (0.5, "tab:gray")]:
        ax.semilogy(ranges / 1000, apparent_delta_t(ranges, dt0, lut, ext), style, color=color,
                    label=f"Target offers: ΔT0 = {dt0:.0f} K, {'clear' if ext == 0.1 else 'haze'}")
for netd, color in [(NETD, "tab:red"), (NETD_TEMPORAL, "tab:green")]:
    need = np.where(freq <= 0.5, required_delta_t(np.minimum(freq, 0.5), netd, cam), np.nan)
    ax.semilogy(ranges / 1000, need, color=color, linewidth=2.5,
                label=f"Camera needs (NETD {1000 * netd:.0f} mK)")
ax.axvline(geometric_range("detection", PERSON, cam) / 1000, color="k", linestyle=":",
           label="Pixel limit (Johnson)")
ax.set_xlabel("Distance (km)")
ax.set_ylabel("Temperature difference (K)")
ax.set_title("Person detection: the camera sees the target where 'offers' > 'needs'")
ax.legend(fontsize=8)
ax.grid(which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("results/exp09_range.png", dpi=150)
plt.show()