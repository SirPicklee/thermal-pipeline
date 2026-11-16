"""The complete processing chain, as it would run inside a thermal camera."""

import time
from dataclasses import dataclass

import numpy as np

from thermalpipe import agc, filters
from thermalpipe.nuc import NUCTable, apply_nuc


@dataclass(frozen=True)
class PipelineConfig:
    replace_bad_pixels: bool = True
    temporal_alpha: float = 1.0          # 1.0 = temporal filter off
    display: str = "plateau"             # "plateau", "dde" or "linear"


class ThermalPipeline:
    """raw 14-bit frame -> (corrected data for the tracker, 8-bit image for the operator)."""

    def __init__(self, table: NUCTable, cfg: PipelineConfig = PipelineConfig()):
        self.table, self.cfg = table, cfg
        self.temporal = filters.TemporalFilter(cfg.temporal_alpha)
        self.timings: dict[str, list[float]] = {}

    def _timed(self, name, function, *args):
        start = time.perf_counter()
        result = function(*args)
        self.timings.setdefault(name, []).append(time.perf_counter() - start)
        return result

    def process(self, raw):
        data = self._timed("NUC", apply_nuc, raw, self.table)
        if self.cfg.replace_bad_pixels:
            data = self._timed("Bad pixels", filters.replace_bad_pixels, data, self.table.bad_pixels)
        if self.cfg.temporal_alpha < 1.0:
            data = self._timed("Temporal filter", self.temporal, data)
        display = self._timed("Display (" + self.cfg.display + ")", self._display, data)
        return data, display

    def _display(self, data):
        if self.cfg.display == "dde":
            return agc.dde(data)
        if self.cfg.display == "linear":
            return agc.linear_percentile(data)
        return agc.plateau_equalization(data)

    def report(self) -> dict[str, float]:
        """Average time per stage [ms] and the resulting frame rate."""
        stages = {name: 1000 * float(np.mean(t)) for name, t in self.timings.items()}
        total = sum(stages.values())
        return {**stages, "Total": total, "FPS": 1000.0 / total if total > 0 else float("inf")}