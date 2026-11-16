"""Experiment 7: the full pipeline on video - temporal filtering vs. moving targets, and speed."""

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from thermalpipe.config import CameraConfig
from thermalpipe.detector import Microbolometer
from thermalpipe.nuc import calibrate_two_point
from thermalpipe.optics import apply_optics, total_blur_px
from thermalpipe.physics import RadianceLUT
from thermalpipe.pipeline import PipelineConfig, ThermalPipeline
from thermalpipe.scene import flat_field, point_target, render_scene
from thermalpipe.tracker import gaussian_template, locate


def calibrated_detector(cam, lut, seed):
    det = Microbolometer(cam, lut=lut, seed=seed)

    def flats(temperature, n=32, settle=5):
        irradiance = apply_optics(flat_field(cam, lut, temperature), cam)
        frames = [det.read(irradiance) for _ in range(settle + n)]
        return frames[settle:]

    return det, calibrate_two_point(flats(290.0), flats(320.0))


# ---------------------------------------------------------------------------
# Part 1: a faint target moving at different speeds, tracked after the pipeline
# ---------------------------------------------------------------------------
cam = CameraConfig(n_rows=96, n_cols=96)
lut = RadianceLUT(cam.band)
template = gaussian_template(np.hypot(0.8, total_blur_px(cam)))
alphas = [1.0, 0.5, 0.25, 0.1]                 # 1.0 = temporal filter off
speeds = [0.0, 0.1, 0.3, 0.6]                  # target speed [pixels per frame]
results = {a: [] for a in alphas}

for alpha in alphas:
    for speed in speeds:
        errors = []
        for s in range(6):
            det, table = calibrated_detector(cam, lut, seed=200 + s)
            pipe = ThermalPipeline(table, PipelineConfig(temporal_alpha=alpha))
            for k in range(40):
                position = (0.0, -6.0 + speed * k)
                data, _ = pipe.process(det.read(apply_optics(
                    point_target(cam, lut, position, delta_t=0.5, sigma_px=0.8), cam)))
                if k >= 20:                    # measure after the filter has settled
                    r, c = locate(data, template)
                    errors.append(np.hypot(r - position[0], c - position[1]) * cam.ifov * 1e6)
        results[alpha].append(np.sqrt(np.mean(np.square(errors))))

print("Tracker error (µrad RMS) vs. target speed:")
print("alpha  " + "".join(f"{v:>9.1f} px/fr" for v in speeds))
for alpha in alphas:
    label = "off " if alpha == 1.0 else f"{alpha:4.2f}"
    print(f"{label}   " + "".join(f"{e:15.1f}" for e in results[alpha]))

# ---------------------------------------------------------------------------
# Part 2: how fast is the pipeline? (full 320 x 256 frames)
# ---------------------------------------------------------------------------
cam_full = CameraConfig()
lut_full = RadianceLUT(cam_full.band)
det, table = calibrated_detector(cam_full, lut_full, seed=1)
frame = det.read(apply_optics(render_scene(cam_full, lut_full).radiance, cam_full))
print("\nProcessing time per frame (320 x 256, Python):")
for display in ["plateau", "dde"]:
    pipe = ThermalPipeline(table, PipelineConfig(temporal_alpha=0.5, display=display))
    for _ in range(20):
        pipe.process(frame)
    report = pipe.report()
    stages = ", ".join(f"{k} {v:.1f} ms" for k, v in report.items() if k not in ("Total", "FPS"))
    print(f"  {display:8s}: {stages} -> {report['Total']:.1f} ms = {report['FPS']:.0f} FPS")

# ---------------------------------------------------------------------------
# Part 3: video of a running person - temporal filter OFF vs. strong (alpha 0.1)
# ---------------------------------------------------------------------------
pipes = [ThermalPipeline(table, PipelineConfig(temporal_alpha=a)) for a in (1.0, 0.1)]
gif_frames = []
for k in range(60):
    radiance = render_scene(cam_full, lut_full, person_shift_m=-3.0 + 0.12 * k).radiance
    raw = det.read(apply_optics(radiance, cam_full))
    left, right = (p.process(raw)[1] for p in pipes)
    gif_frames.append(Image.fromarray(np.hstack([left, np.zeros((cam_full.n_rows, 6), np.uint8), right])))
gif_frames[0].save("results/exp07_video.gif", save_all=True, append_images=gif_frames[1:],
                   duration=60, loop=0)
print("\nVideo saved: results/exp07_video.gif (left: temporal filter off, right: alpha 0.1)")

plt.figure(figsize=(8, 5))
for alpha in alphas:
    plt.plot(speeds, results[alpha], "o-", label="filter off" if alpha == 1.0 else f"alpha = {alpha}")
plt.xlabel("Target speed (pixels per frame)")
plt.ylabel("Tracker error (µrad RMS)")
plt.title("Temporal filter: great for still targets, bad for fast ones")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/exp07_tradeoff.png", dpi=150)
plt.show()