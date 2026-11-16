"""Camera configuration shared by all modules."""

from dataclasses import dataclass

from thermalpipe.physics import LWIR


@dataclass(frozen=True)
class CameraConfig:
    """An uncooled LWIR camera, similar to a thermal weapon sight."""

    n_rows: int = 256                  # image height [pixels]
    n_cols: int = 320                  # image width [pixels]
    pixel_pitch: float = 12e-6         # distance between pixel centres [m]
    focal_length: float = 100e-3       # lens focal length [m]
    f_number: float = 1.2              # focal length / aperture diameter
    optics_transmission: float = 0.9   # fraction of light that passes the lens
    aberration_blur_px: float = 0.3    # extra blur from lens imperfections [pixels, sigma]
    band: tuple = LWIR                 # spectral band [m]

    @property
    def ifov(self) -> float:
        """Angle seen by one pixel [rad]."""
        return self.pixel_pitch / self.focal_length

    @property
    def wavelength_center(self) -> float:
        """Centre wavelength of the band [m]."""
        return 0.5 * (self.band[0] + self.band[1])