"""Image quality metrics for thermal imagers."""

import numpy as np


def temporal_noise(frames, valid=None) -> float:
    """Average frame-to-frame standard deviation of a pixel [counts]."""
    per_pixel = np.std(np.asarray(frames, dtype=float), axis=0)
    return float(np.mean(per_pixel if valid is None else per_pixel[valid]))


def spatial_noise(frames, valid=None) -> float:
    """Fixed pattern noise: pixel-to-pixel spread of the time-averaged image [counts]."""
    mean_image = np.mean(np.asarray(frames, dtype=float), axis=0)
    return float(np.std(mean_image if valid is None else mean_image[valid]))


def measured_netd(frames_cold, frames_hot, delta_t, valid=None) -> float:
    """NETD [K] = temporal noise / signal change per kelvin, from two flat-field stacks."""
    cold = np.mean(np.asarray(frames_cold, dtype=float), axis=0)
    hot = np.mean(np.asarray(frames_hot, dtype=float), axis=0)
    response = (hot - cold) / delta_t
    if valid is not None:
        response = response[valid]
    return temporal_noise(frames_cold, valid) / float(np.median(response))

# ---------------------------------------------------------------------------
# Range performance (Johnson criteria + MRTD-like contrast check)
# ---------------------------------------------------------------------------

JOHNSON_CYCLES = {"detection": 1.0, "recognition": 4.0, "identification": 6.4}


def system_mtf(frequency, cam):
    """Lens blur x pixel size: how much contrast survives at a spatial frequency [cycles/pixel]."""
    from thermalpipe.optics import mtf_at
    f = np.asarray(frequency, dtype=float)
    return mtf_at(f, cam) * np.abs(np.sinc(f))


def required_delta_t(frequency, netd, cam, snr_threshold=2.5):
    """Smallest temperature difference an observer can use at a given detail size (MRTD-like)."""
    return snr_threshold * netd / np.maximum(system_mtf(frequency, cam), 1e-9)


def apparent_delta_t(distance_m, delta_t0, lut, extinction_per_km=0.2,
                     background_temp=293.0, air_temp=293.0):
    """Target-background temperature difference after the atmosphere [K]."""
    from thermalpipe.physics import apparent_radiance, atmospheric_transmission
    tau = atmospheric_transmission(distance_m, extinction_per_km)
    l_target = apparent_radiance(lut.radiance(background_temp + delta_t0), lut.radiance(air_temp), tau)
    l_back = apparent_radiance(lut.radiance(background_temp), lut.radiance(air_temp), tau)
    per_kelvin = lut.radiance(background_temp + 0.5) - lut.radiance(background_temp - 0.5)
    return (l_target - l_back) / per_kelvin


def task_range(task, critical_size_m, delta_t0, cam, lut, netd=0.050,
               extinction_per_km=0.2, snr_threshold=2.5, max_range_m=30000.0):
    """Largest distance [m] at which a task (detection, recognition, ...) is still possible.

    Two limits: enough pixels on the target (geometry, Johnson) AND enough contrast left
    after the atmosphere to beat the noise at that detail size.
    """
    ranges = np.linspace(10.0, max_range_m, 6000)
    pixels_on_target = critical_size_m / (ranges * cam.ifov)
    frequency = JOHNSON_CYCLES[task] / pixels_on_target            # cycles per pixel
    resolvable = frequency <= 0.5                                  # Nyquist: 2 pixels per cycle
    visible = apparent_delta_t(ranges, delta_t0, lut, extinction_per_km) >= \
        required_delta_t(np.minimum(frequency, 0.5), netd, cam, snr_threshold)
    ok = resolvable & visible
    if not ok[0]:
        return 0.0
    if ok.all():
        return float(max_range_m)
    return float(ranges[np.argmin(ok)])                            # first distance that fails


def geometric_range(task, critical_size_m, cam):
    """Range limited by pixels only (the classic Johnson estimate)."""
    return critical_size_m / (2.0 * JOHNSON_CYCLES[task] * cam.ifov)