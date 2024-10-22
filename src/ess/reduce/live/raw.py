# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Raw count processing and visualization for live data display."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import scipp as sc


@dataclass
class DetectorParams:
    detector_number: sc.Variable
    x_pixel_offset: sc.Variable | None = None
    y_pixel_offset: sc.Variable | None = None
    z_pixel_offset: sc.Variable | None = None

    def __post_init__(self):
        self._flat_detector_number = self.detector_number.flatten(to='event_id')
        if not sc.issorted(self._flat_detector_number, dim='event_id'):
            raise ValueError("Detector numbers must be sorted.")
        if self.stop - self.start + 1 != self.size:
            raise ValueError("Detector numbers must be consecutive.")

    @property
    def size(self) -> int:
        return int(self._flat_detector_number.size)

    @property
    def start(self) -> int:
        return int(self._flat_detector_number[0].value)

    @property
    def stop(self) -> int:
        return int(self._flat_detector_number[-1].value)


class Detector:
    def __init__(self, params: DetectorParams):
        self._data = sc.DataArray(
            sc.zeros(sizes=params.detector_number.sizes, unit='counts', dtype='int32'),
            coords={'detector_id': params.detector_number},
        )
        if params.x_pixel_offset is not None:
            self._data.coords['x_pixel_offset'] = params.x_pixel_offset
        if params.y_pixel_offset is not None:
            self._data.coords['y_pixel_offset'] = params.y_pixel_offset
        if params.z_pixel_offset is not None:
            self._data.coords['z_pixel_offset'] = params.z_pixel_offset
        self._start = params.start
        self._size = params.size

    @property
    def data(self) -> sc.DataArray:
        return self._data

    def bincount(self, data: Sequence[int]) -> sc.DataArray:
        offset = np.asarray(data, dtype=np.int32) - self._start
        out = sc.empty_like(self.data)
        out.values = np.bincount(offset, minlength=self._size).reshape(self.data.shape)
        return out

    def add_counts(self, data: Sequence[int]) -> None:
        self._data += self.bincount(data)

    def clear_counts(self) -> None:
        self._data.values *= 0


class RollingDetectorView(Detector):
    def __init__(
        self,
        params: DetectorParams,
        *,
        window: int,
        projection: Callable[[sc.DataArray], sc.DataArray] | None = None,
    ):
        super().__init__(params)
        self._projection = projection
        self._window = window
        self._current = 0
        self._history: sc.DataArray | None = None
        self._cache: sc.DataArray | None = None

        counts = self.bincount([])
        if self._projection is not None:
            counts = self._projection(counts)
        self._history = (
            sc.zeros_like(counts)
            .broadcast(sizes={'window': self._window, **counts.sizes})
            .copy()
        )
        self._cache = self._history.sum('window')

    def get(self, window: int | None = None) -> sc.DataArray:
        if window is not None and not 0 <= window <= self._window:
            raise ValueError("Window size must be less than the history size.")
        if window is None or window == self._window:
            data = self._cache
        else:
            start = self._current - window
            if start >= 0:
                data = self._history['window', start : self._current].sum('window')
            else:
                data = self._history['window', start % self._window :].sum('window')
                data += self._history['window', 0 : self._current].sum('window')
        return data

    def add_counts(self, data: Sequence[int]) -> None:
        counts = self.bincount(data)
        if self._projection is not None:
            counts = self._projection(counts)
        self._cache -= self._history['window', self._current]
        self._history['window', self._current] = counts
        self._cache += self._history['window', self._current]
        self._current = (self._current + 1) % self._window
