# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Raw count processing and visualization for live data display."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from time import time
from typing import NewType
from math import ceil

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
    def from_nexus(
        nexus_file: str, *, detector_name: str, window: int, xres: int, yres: int
    ) -> 'RollingDetectorView':
        wf = GenericNeXusWorkflow()
        # TODO Could also set workflow parameters.
        wf.insert(make_xy_plane_projection_factory(xres=xres, yres=yres))
        wf.insert(make_rolling_detector_view_factory(window=window))
        wf.insert(pixel_shape)
        wf.insert(pixel_cylinder_axis)
        wf.insert(pixel_cylinder_radius)
        wf.insert(position_noise_for_cylindrical_pixel)
        wf.insert(position_with_noisy_replicas)
        wf[PositionNoiseReplicaCount] = 4
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
    x: sc.Variable, y: sc.Variable, z: sc.Variable, *, zplane: sc.Variable | None = None
) -> dict[str, sc.Variable]:
    zmin = z.min()
    zero = sc.zeros_like(zmin)
    if zplane is None:
        zplane = z.max() if zmin < zero else zmin
    t = zplane / z
    return sc.DataGroup(x=x * t, y=y * t, z=zplane)


def project_onto_cylinder(
    x: sc.Variable, y: sc.Variable, z: sc.Variable, *, radius: sc.Variable | None = None
) -> dict[str, sc.Variable]:
    r_xy = sc.sqrt(x**2 + y**2)
    if radius is None:
        radius = r_xy.min()
    t = radius / r_xy
    return {'x': x * t, 'y': y * t, 'z': z * t}


class Projection:
    pass


PixelShape = NewType('PixelShape', sc.DataGroup)
PixelCylinderAxis = NewType('PixelCylinderAxis', sc.Variable)
PixelCylinderRadius = NewType('PixelCylinderRadius', sc.Variable)
PositionNoiseCylindricalPixel = NewType('PositionNoiseCylindricalPixel', sc.Variable)
PositionNoiseReplicaCount = NewType('PositionNoiseReplicaCount', int)
CalibratedPositionWithNoisyReplicas = NewType(
    'CalibratedPositionWithNoisyReplicas', sc.Variable
)


def make_rolling_detector_view_factory(window: int):
    def make_rolling_detector_view(
        detector: CalibratedDetector[SampleRun],
        projection: Projection,
    ) -> RollingDetectorView:
        params = DetectorParams(detector_number=detector.coords['detector_number'])
        return RollingDetectorView(params=params, window=window, projection=projection)

    return make_rolling_detector_view


def make_xy_plane_projection_factory(xres: int, yres: int):
    def make_xy_plane_projection(
        position: CalibratedPositionWithNoisyReplicas,
    ) -> Projection:
        return XYPlaneProjection(pos=position, xres=xres, yres=yres)

    return make_xy_plane_projection


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
    t = transform.value
    return PixelCylinderAxis(t * vertices[2] - t * vertices[0])


def pixel_cylinder_radius(
    shape: PixelShape,
    transform: NeXusTransformation[snx.NXdetector, SampleRun],
) -> PixelCylinderRadius:
    vertices = shape['vertices']
    if len(vertices) != 3:
        raise NotImplementedError("Case of multiple cylinders not implemented.")
    # Note that transformation may be affine, so we need to apply it to the vertices
    # *before* subtracting them, to remove the translation part.
    t = transform.value
    return PixelCylinderRadius(t * vertices[1] - t * vertices[0])


def position_noise_for_cylindrical_pixel(
    *,
    axis: PixelCylinderAxis,
    radius: PixelCylinderRadius,
) -> PositionNoiseCylindricalPixel:
    # We *assume* that the cylinder is centered on the origin. Real files may not
    # fulfill this. However, the rest of the data reduction currently assumes that
    # the pixel offset corresponds to the pixel center, so if it is not fulfilled
    # there are bigger problems elsewhere anywhere.
    rng = np.random.default_rng()
    dims = ('random_point_in_cylindrical_pixel',)
    size = 117

    z_hat = axis / sc.norm(axis)  # Unit vector along the cylinder axis
    x_hat = radius / sc.norm(radius)  # Unit vector along the radius direction
    y_hat = sc.cross(z_hat, x_hat)  # Unit vector perpendicular to z_hat and x_hat

    dz = sc.array(dims=dims, values=rng.uniform(-0.5, 0.5, size=size)) * axis
    dphi = sc.array(dims=dims, values=rng.uniform(0, 2 * np.pi, size=size), unit='rad')
    r = sc.norm(radius).value
    dr = sc.sqrt(
        sc.array(dims=dims, values=rng.uniform(0, r**2, size=size), unit='m^2')
    )

    dx = dr * sc.cos(dphi) * x_hat
    dy = dr * sc.sin(dphi) * y_hat

    return dx + dy + dz


def position_with_noisy_replicas(
    *,
    detector: CalibratedDetector[SampleRun],
    position_noise: PositionNoiseCylindricalPixel,
    replicas: PositionNoiseReplicaCount,
) -> CalibratedPositionWithNoisyReplicas:
    """
    Create a new position array with noise added to the detector positions.

    The first slice of the new array is the original position data, and the
    remaining slices are the original data with noise added.
    """
    position = detector.coords['position']
    noise_dim = position_noise.dim
    size = position.size * replicas
    # "Paint" the short array of noise on top of the (replicated) position data.
    noise = sc.concat(
        [position_noise] * ceil(size / position_noise.size), dim=noise_dim
    )[:size].fold(dim=noise_dim, sizes={noise_dim: replicas, position.dim: -1})
    return sc.concat([position, noise + position], dim=noise_dim)


class XYPlaneProjection(Projection):
    def __init__(
        self,
        *,
        pos: CalibratedPositionWithNoisyReplicas,
        xres: int = 150,
        yres: int = 150,
    ):
        self._noise_dim = pos.dims[0]
        self._replicas = pos.sizes[self._noise_dim]
        position = pos[self._noise_dim, 0]
        zplane = position.fields.z.min()  # position, not pos, to avoid noise
        self._coords = project_xy(
            pos.fields.x, pos.fields.y, pos.fields.z, zplane=zplane
        )
        x = self._coords['x']
        y = self._coords['y']

        self._current = 0
        delta = sc.scalar(0.01, unit='m')
        self._x_edges = sc.linspace(
            'x', x.min() - delta, x.max() + delta, num=xres + 1, unit='m'
        )
        self._y_edges = sc.linspace(
            'y', y.min() - delta, y.max() + delta, num=yres + 1, unit='m'
        )

    def __call__(self, da: sc.DataArray) -> sc.DataArray:
        data = da.data.flatten(to='detector_number')
        self._current += 1
        coords = self._coords[self._noise_dim, self._current % self._replicas]
        return sc.DataArray(data, coords=coords).hist(y=self._y_edges, x=self._x_edges)
