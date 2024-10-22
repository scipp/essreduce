# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Raw count processing and visualization for live data display."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import time

import numpy as np
import scipp as sc

from ess.reduce.nexus.types import NeXusTransformation


@dataclass
class DetectorParams:
    detector_number: sc.Variable
    x_pixel_offset: sc.Variable | None = None
    y_pixel_offset: sc.Variable | None = None
    z_pixel_offset: sc.Variable | None = None
    transformation: NeXusTransformation | None = None
    pixel_shape: sc.DataGroup | None = None

    def __post_init__(self):
        self._flat_detector_number = self.detector_number.flatten(to='event_id')
        if not sc.issorted(self._flat_detector_number, dim='event_id'):
            raise ValueError("Detector numbers must be sorted.")
        if self.stop - self.start + 1 != self.size:
            raise ValueError("Detector numbers must be consecutive.")

    @staticmethod
    def from_nexus(nexus_file: str, detector_name: str) -> 'DetectorParams':
        import scippnexus as snx

        from ess.reduce.nexus.types import (
            Filename,
            NeXusComponent,
            NeXusDetectorName,
            NeXusTransformation,
            SampleRun,
        )
        from ess.reduce.nexus.workflow import GenericNeXusWorkflow

        wf = GenericNeXusWorkflow()
        wf[Filename[SampleRun]] = nexus_file
        wf[NeXusDetectorName] = detector_name
        results = wf.compute(
            (
                NeXusTransformation[snx.NXdetector, SampleRun],
                NeXusComponent[snx.NXdetector, SampleRun],
            )
        )
        comp = results[NeXusComponent[snx.NXdetector, SampleRun]]
        pixel_shape = comp.get('pixel_shape')
        pixel_offset = snx.zip_pixel_offsets(comp['data'].coords)
        return DetectorParams(
            # detector_number=comp['data'].coords['detector_number'],
            **comp['data'].coords,
            transformation=results[NeXusTransformation[snx.NXdetector, SampleRun]],
            pixel_shape=pixel_shape,
        )

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
        if all(
            coord in self._data.coords
            for coord in ['x_pixel_offset', 'y_pixel_offset', 'z_pixel_offset']
        ):
            self._data.coords['pixel_offset'] = sc.spatial.as_vectors(
                x=self._data.coords['x_pixel_offset'],
                y=self._data.coords['y_pixel_offset'],
                z=self._data.coords['z_pixel_offset'],
            )
            del self._data.coords['x_pixel_offset']
            del self._data.coords['y_pixel_offset']
            del self._data.coords['z_pixel_offset']
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

    @staticmethod
    def from_nexus(nexus_file: str, detector_name: str) -> 'RollingDetectorView':
        from ess.reduce.nexus.types import (
            Filename,
            NeXusDetectorName,
            SampleRun,
        )
        from ess.reduce.nexus.workflow import GenericNeXusWorkflow

        wf = GenericNeXusWorkflow()
        wf.insert(make_xy_plane_projection)
        wf.insert(make_rolling_detector_view)
        wf[Filename[SampleRun]] = nexus_file
        wf[NeXusDetectorName] = detector_name
        return wf.compute(RollingDetectorView)

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
        start = time()
        counts = self.bincount(data)
        if self._projection is not None:
            counts = self._projection(counts)
        self._cache -= self._history['window', self._current]
        self._history['window', self._current] = counts
        self._cache += counts
        self._current = (self._current + 1) % self._window
        print(f"add_counts {len(data)}: {time() - start:.2f}s")


def project_xy(
    x: sc.Variable, y: sc.Variable, z: sc.Variable
) -> dict[str, sc.Variable]:
    zmin = z.min()
    t = zmin / z
    return {'x': x * t, 'y': y * t, 'z': zmin}


# TODO init chicken-egg problem
# Detector needs DetectorParams
# Projection needs extra info, but some overlap
# What goes where? How to init in what order?
# Just think about DI!
# RollingDetectorView needs
# - DetectorNumber (~NeXusComponent)
# - Does not really needs pixel offsets, but likes to include it in output
#   -> is this just a special kind of projection?
# - Projection
#   - pixel offsets
#   - pixel shape
#   - transformation (to apply to position noise)
#   - position
import scippnexus as snx

from ess.reduce.nexus.types import (
    CalibratedDetector,
    Filename,
    NeXusComponent,
    NeXusDetectorName,
    NeXusTransformation,
    SampleRun,
)
from ess.reduce.nexus.workflow import GenericNeXusWorkflow


class Projection:
    pass


def make_rolling_detector_view(
    detector: CalibratedDetector[SampleRun],  # detector_number
    projection: Projection,
) -> RollingDetectorView:
    params = DetectorParams(detector_number=detector.coords['detector_number'])
    return RollingDetectorView(params=params, window=100, projection=projection)


# Either insert this in workflow, or use wf.bind_and_call
def make_xy_plane_projection(
    component: NeXusComponent[snx.NXdetector, SampleRun],  # pixel shape
    detector: CalibratedDetector[SampleRun],  # position coord
    transformation: NeXusTransformation[snx.NXdetector, SampleRun],
) -> Projection:
    # Basics idea:
    # 1. pixel_shape -> noise_vector (only cylindrical for now?)
    # 2. transform * noise_vector... no! only rot!
    # 3. On call, add randomly selected noise to position
    # 4. Project to plane (TODO only x=y for now)
    return LokiProjection(
        pixel_shape=component['pixel_shape'],
        position=detector.coords['position'],
        transformation=transformation.value,
    )


# Generalize to TubeProjection
class LokiProjection:
    def __init__(
        self,
        pixel_shape: sc.DataGroup,
        position: sc.Variable,
        transformation: sc.Variable,
    ):
        # TODO Use NeXusTransformation, apply to pos and noise once, add in __call__
        # The former is just position of CalibratedDetector
        # TODO Find cylinder axis from NeXusDetector, or hard-code
        rng = np.random.default_rng()
        # 2 mm height, 4 mm radius
        cyl_height = 0.002
        dx = cyl_height / 2
        cyl_radius = 0.004
        dims = ['pixel']
        dims = position.dims
        size = int(1e6)
        dx = sc.array(dims=dims, values=rng.uniform(-dx, dx, size=size), unit='m')
        angle = sc.array(
            dims=dims, values=rng.uniform(0, 2 * np.pi, size=size), unit='rad'
        )
        radius = sc.sqrt(
            sc.array(
                dims=dims, values=rng.uniform(0, cyl_radius**2, size=size), unit='m^2'
            )
        )
        dy = radius * sc.sin(angle)
        dz = radius * sc.cos(angle)
        noise = sc.spatial.as_vectors(x=dx, y=dy, z=dz)
        self._x_edges = sc.linspace('x', -0.4, 0.5, num=151, unit='m')
        self._y_edges = sc.linspace('y', -0.4, 0.4, num=151, unit='m')
        self._position = sc.vector([-0.49902349, 0.43555999, 4.09899989], unit='m')
        self._beam_center = sc.vector([-0.02864121, -0.01850989, 0.0], unit='m')

        # TODO want to rotate the noise, not translate -> use difference
        # noise = transform * noise - transform * origin
        # hmmm, but why not transform the shape? harder to apply noise?
        # or just do position - transform * noise, if we do not care about
        # absolute trans?
        self._position = position
        self._pixel_offset_noise = (
            transformation * noise - transformation * sc.zeros_like(noise[0])
        )

    def __call__(self, da: sc.DataArray) -> sc.DataArray:
        data = da.flatten(to=self._position.dim)
        pos = (
            self._position
            # (self._position - self._beam_center)
            # + data.coords['pixel_offset']
            + self._pixel_offset_noise[: data.size]
        )
        return sc.DataArray(
            data.data, coords=project_xy(pos.fields.x, pos.fields.y, pos.fields.z)
        ).hist(y=self._y_edges, x=self._x_edges)
