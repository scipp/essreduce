# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Raw count processing and visualization for live data display."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import time
from typing import NewType

import numpy as np
import scipp as sc
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
    zero = sc.zeros_like(zmin)
    zplane = z.max() if zmin < zero else zmin
    t = zplane / z
    return {'x': x * t, 'y': y * t, 'z': zplane}


class Projection:
    pass


def make_rolling_detector_view(
    detector: CalibratedDetector[SampleRun],
    projection: Projection,
) -> RollingDetectorView:
    params = DetectorParams(detector_number=detector.coords['detector_number'])
    return RollingDetectorView(params=params, window=100, projection=projection)


def make_xy_plane_projection(
    component: NeXusComponent[snx.NXdetector, SampleRun],
    detector: CalibratedDetector[SampleRun],
    transformation: NeXusTransformation[snx.NXdetector, SampleRun],
) -> Projection:
    return LokiProjection(
        pixel_shape=component['pixel_shape'],
        position=detector.coords['position'],
        transformation=transformation.value,
    )


PixelShape = NewType('PixelShape', sc.DataGroup)
PixelCylinderAxis = NewType('PixelCylinderAxis', sc.Variable)
PixelCylinderRadius = NewType('PixelCylinderRadius', sc.Variable)


def pixel_shape(component: NeXusComponent[snx.NXdetector, SampleRun]) -> PixelShape:
    return PixelShape(component['pixel_shape'])


def pixel_cylinder_axis(
    shape: PixelShape,
    transform: NeXusTransformation[snx.NXdetector, SampleRun],
) -> PixelCylinderAxis:
    vertices = shape['vertices']
    if len(vertices) != 3:
        raise NotImplementedError("Case of multiple cylinders not implemented.")
    # Note that transformation may be affine, so we need to apply it to the vertices
    # *before* subtracting them, to remove the translation part.
    return PixelCylinderAxis(transform * vertices[2] - transform * vertices[0])


def pixel_cylinder_radius(
    shape: PixelShape,
    transform: NeXusTransformation[snx.NXdetector, SampleRun],
) -> PixelCylinderRadius:
    vertices = shape['vertices']
    if len(vertices) != 3:
        raise NotImplementedError("Case of multiple cylinders not implemented.")
    # Note that transformation may be affine, so we need to apply it to the vertices
    # *before* subtracting them, to remove the translation part.
    return PixelCylinderRadius(transform * vertices[1] - transform * vertices[0])


# Generalize to TubeProjection
class LokiProjection:
    def __init__(
        self,
        pixel_shape: sc.DataGroup,
        position: sc.Variable,
        transformation: sc.Variable,
    ):
        rng = np.random.default_rng()
        dims = position.dims
        size = position.size
        # We *assume* that the cylinder is centered on the origin. Real files may not
        # fulfill this. However, the rest of the data reduction currently assumes that
        # the pixel offset corresponds to the pixel center, so if it is not fulfilled
        # there are bigger problems elsewhere anywhere.

        axis = pixel_cylinder_axis(pixel_shape, transformation)
        radius = pixel_cylinder_radius(pixel_shape, transformation)

        # Normalize vectors to get unit directions
        z_hat = axis / sc.norm(axis)  # Unit vector along the cylinder axis
        x_hat = radius / sc.norm(radius)  # Unit vector along the radius direction
        y_hat = sc.cross(z_hat, x_hat)  # Unit vector perpendicular to z_hat and x_hat

        dz = sc.array(dims=dims, values=rng.uniform(-0.5, 0.5, size=size)) * axis
        dphi = sc.array(
            dims=dims, values=rng.uniform(0, 2 * np.pi, size=size), unit='rad'
        )
        dr = sc.sqrt(
            sc.array(
                dims=dims,
                values=rng.uniform(0, sc.norm(radius).value ** 2, size=size),
                unit='m^2',
            )
        )

        dx = dr * sc.cos(dphi) * x_hat
        dy = dr * sc.sin(dphi) * y_hat

        # 2 mm height, 4 mm radius
        # cyl_height = 0.002
        # dx = cyl_height / 2
        # cyl_radius = 0.004
        # dx = sc.array(dims=dims, values=rng.uniform(-dx, dx, size=size), unit='m')
        # angle = sc.array(
        #    dims=dims, values=rng.uniform(0, 2 * np.pi, size=size), unit='rad'
        # )
        # radius = sc.sqrt(
        #    sc.array(
        #        dims=dims, values=rng.uniform(0, cyl_radius**2, size=size), unit='m^2'
        #    )
        # )
        # dy = radius * sc.sin(angle)
        # dz = radius * sc.cos(angle)
        # noise = sc.spatial.as_vectors(x=dx, y=dy, z=dz)
        noise = dx + dy + dz
        # TODO Can we get min and max from extents?
        self._x_edges = sc.linspace('x', -0.4, 0.5, num=151, unit='m')
        self._y_edges = sc.linspace('y', -0.4, 0.4, num=151, unit='m')

        self._position = position
        self._pixel_offset_noise = noise
        # The transformation is in generally an affine transform. We do not want the
        # translation part, so we subtract a transformation applied to the origin.
        # self._pixel_offset_noise = (
        #    transformation * noise - transformation * sc.zeros_like(noise[0])
        # )

        self._split = 0

    def __call__(self, da: sc.DataArray) -> sc.DataArray:
        data = da.flatten(to=self._position.dim)
        noise = sc.concat(
            [
                self._pixel_offset_noise[0 : self._split],
                self._pixel_offset_noise[self._split :],
            ],
            self._position.dim,
        )
        self._split = (self._split + 1) % len(self._position)
        pos = self._position + noise  # + self._pixel_offset_noise[: data.size]
        return sc.DataArray(
            data.data, coords=project_xy(pos.fields.x, pos.fields.y, pos.fields.z)
        ).hist(y=self._y_edges, x=self._x_edges)
