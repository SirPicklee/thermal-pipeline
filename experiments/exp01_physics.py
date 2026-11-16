"""Experiment 1: Planck curves, LWIR vs MWIR, and contrast loss in the atmosphere."""

import matplotlib.pyplot as plt
import numpy as np

from thermalpipe.physics import (
    LWIR, MWIR, RadianceLUT, apparent_radiance, atmospheric_transmission,
    band_radiance, planck_spectral_radiance,
)

# --- 1) Planck curves ---
wavelengths = np.linspace(1e-6, 20e-6, 2000)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
for temp, label in [(300, "Background 300 K"), (310, "Human 310 K"), (600, "Hot engine 600 K")]:
    ax1.semilogy(wavelengths * 1e6, planck_spectral_radiance(wavelengths, temp) * 1e-6, label=label)
ax1.axvspan(MWIR[0] * 1e6, MWIR[1] * 1e6, alpha=0.15, color="orange", label="MWIR band")
ax1.axvspan(LWIR[0] * 1e6, LWIR[1] * 1e6, alpha=0.15, color="red", label="LWIR band")
ax1.set_xlabel("Wavelength (µm)")
ax1.set_ylabel("Spectral radiance (W / m² / sr / µm)")
ax1.set_ylim(1e-3, 1e4)
ax1.set_title("Planck curves")
ax1.legend()
ax1.grid(which="both", alpha=0.3)

# --- 2) LWIR vs MWIR at room temperature ---
print("Room temperature (300 K) scene:")
for name, band in [("LWIR", LWIR), ("MWIR", MWIR)]:
    radiance = float(band_radiance(300.0, band))
    per_kelvin = float(band_radiance(301.0, band) - band_radiance(299.0, band)) / 2
    print(f"  {name}: radiance {radiance:7.3f} W/m²/sr | "
          f"+1 K changes it by {per_kelvin:.4f} ({100 * per_kelvin / radiance:.2f} %)")

# --- 3) Contrast of a 310 K person vs 300 K background through the air ---
lut = RadianceLUT(LWIR)
one_kelvin = lut.radiance(300.5) - lut.radiance(299.5)       # radiance of 1 K difference
distances = np.linspace(0, 10000, 200)
for ext, label in [(0.1, "Clear air (0.1 /km)"), (0.2, "Average (0.2 /km)"), (0.5, "Haze (0.5 /km)")]:
    tau = atmospheric_transmission(distances, ext)
    contrast = (apparent_radiance(lut.radiance(310.0), lut.radiance(295.0), tau)
                - apparent_radiance(lut.radiance(300.0), lut.radiance(295.0), tau))
    ax2.plot(distances / 1000, contrast / one_kelvin, label=label)
ax2.axhline(0.05, color="r", linestyle="--", label="50 mK (typical NETD)")
ax2.set_xlabel("Distance (km)")
ax2.set_ylabel("Apparent temperature difference (K)")
ax2.set_title("Person vs. background: contrast lost in the atmosphere")
ax2.legend()
ax2.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("results/exp01_physics.png", dpi=150)
plt.show()