# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""
Raw count processing and visualization for live data display.

The main feature of this module if :py:class:`RollingDetectorView`, which provides a
rolling view of a detector. This is useful for live data display, where the most recent
counts, i.e., from the last N pulses, are of interest.

As various detector geometries can often not be directly visualized in a useful way,
the module also provides a way to project the detector data onto a different coordinate
systems. A full set of options is not implemented yet. Currently there are three
options:

- `'xy_plane'`: Project the data onto the x-y plane, i.e., perpendicular to the beam.
- `'cylinder_mantle_z'`: Project the data onto the mantle of a cylinder aligned with the
   z-axis.
- A callable, e.g., to select and flatten dimensions of the data.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from math import ceil
from typing import Literal, NewType

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

from . import roi

CalibratedPositionWithNoisyReplicas = NewType(
    'CalibratedPositionWithNoisyReplicas', sc.Variable
)
DetectorViewResolution = NewType('DetectorViewResolution', dict[str, int])
PixelCylinderAxis = NewType('PixelCylinderAxis', sc.Variable)
PixelCylinderRadius = NewType('PixelCylinderRadius', sc.Variable)
PixelShape = NewType('PixelShape', sc.DataGroup)
PositionNoise = NewType('PositionNoise', sc.Variable)
PositionNoiseReplicaCount = NewType('PositionNoiseReplicaCount', int)
PositionNoiseSigma = NewType('PositionNoiseSigma', sc.Variable)
ProjectedCoords = NewType('ProjectedCoords', sc.DataGroup[str, sc.Variable])
RollingDetectorViewWindow = NewType('RollingDetectorViewWindow', int)


class Histogrammer:
    """
    Histogrammer for a detector view.

    The histogrammer is used to project the detector data onto a different coordinate
    system. This class implements the common logic for different projections. For each
    concrete projection a setup mechanism computes relevant coordinates. This is done,
    e.g., in :py:func:`project_xy` and :py:func:`project_onto_cylinder`.
    """

    def __init__(
        self,
        coords: sc.DataGroup[str, sc.Variable],
        edges: sc.DataGroup[str, sc.Variable],
    ):
        self._current = 0
        self._replica_dim = 'replica'
        self._replicas = coords.sizes[self._replica_dim]
        self._coords = coords
        self._edges = edges

    @property
    def replicas(self) -> int:
        return self._replicas

    @staticmethod
    def from_coords(
        coords: ProjectedCoords, resolution: DetectorViewResolution
    ) -> Histogrammer:
        """
        Create a histogrammer from coordinates and resolution.

        Parameters
        ----------
        coords:
            Coordinates to use for histogramming. May contain more than the
            dimensions used for histogramming.
        resolution:
            Resolution to use for histogramming. The keys are the dimensions
            to histogram, and the values are the number of bins to use. The order
            of the dimensions is preserved in the output and thus controls which axis
            is which in a plot.
        """
        edges = sc.DataGroup(
            {
                dim: coords[dim].hist({dim: res}).coords[dim]
                for dim, res in resolution.items()
            }
        )
        return Histogrammer(coords=coords, edges=edges)

    def __call__(self, da: sc.DataArray) -> sc.DataArray:
        self._current += 1
        coords = self._coords[self._replica_dim, self._current % self._replicas]
        return self._hist(da.data, coords=coords)

    def _hist(self, data: sc.Variable, *, coords: sc.DataGroup) -> sc.DataArray:
        # If input is multi-dim we need to flatten since those dims cannot be preserved.
        return sc.DataArray(data, coords=coords).flatten(to='_').hist(self._edges)

    def input_indices(self) -> sc.DataArray:
        """Return an array with input indices corresponding to each histogram bin."""
        dim = 'detector_number'
        # For some projections one of the coords is a scalar, convert to flat table.
        coords = self._coords.broadcast(sizes=self._coords.sizes).flatten(to=dim).copy()
        ndet = sc.index(coords.sizes[dim] // self._replicas)
        da = sc.DataArray(
            sc.arange(dim, coords.sizes[dim], dtype='int64', unit=None) % ndet,
            coords=coords,
        )
        return sc.DataArray(da.bin(self._edges).bins.data, coords=self._edges)

    def apply_full(self, var: sc.Variable) -> sc.DataArray:
        """
        Apply the histogrammer to a variable using all replicas.

        This is used for one-off operations where the full data is needed, e.g., for
        transforming pixel weights. Compare to :py:meth:`__call__`, which applies the
        histogrammer to a single replica for efficiency.
        """
        replicated = sc.concat([var] * self.replicas, dim=self._replica_dim)
        return self._hist(replicated, coords=self._coords) / self.replicas


class Detector:
    def __init__(self, detector_number: sc.Variable):
        self._data = sc.DataArray(
            sc.zeros(sizes=detector_number.sizes, unit='counts', dtype='int32'),
            coords={'detector_number': detector_number},
        )
        self._detector_number = detector_number
        self._flat_detector_number = detector_number.flatten(to='event_id')
        self._start = int(self._flat_detector_number[0].value)
        self._stop = int(self._flat_detector_number[-1].value)
        self._size = int(self._flat_detector_number.size)
        self._sorted = sc.issorted(self._flat_detector_number, dim='event_id')
        self._consecutive = self._stop - self._start + 1 == self._size

    @property
    def detector_number(self) -> sc.Variable:
        return self._detector_number

    @property
    def data(self) -> sc.DataArray:
        return self._data

    def bincount(self, data: Sequence[int]) -> sc.DataArray:
        if not self._sorted:
            raise ValueError("Detector numbers must be sorted to use `bincount`.")
        if not self._consecutive:
            raise ValueError("Detector numbers must be consecutive to use `bincount`.")
        offset = np.asarray(data, dtype=np.int32) - self._start
        # Ignore events with detector numbers outside the range of the detector. This
        # should not happen in valid files but for now it is useful until we are sure
        # we get only valid files from upstream.
        offset = offset[(offset >= 0) & (offset < self._size)]
        out = sc.empty_like(self.data)
        out.values = np.bincount(offset, minlength=self._size).reshape(self.data.shape)
        return out

    def add_counts(self, data: Sequence[int]) -> None:
        self._data += self.bincount(data)

    def clear_counts(self) -> None:
        self._data.values *= 0


class RollingDetectorView(Detector):
    """
    Rolling view of a detector.

    The view keeps a history of counts and allows to access the sum of counts over a
    window of the most recent counts.
    """

    def __init__(
        self,
        *,
        detector_number: sc.Variable,
        window: int,
        projection: Callable[[sc.DataArray], sc.DataArray] | None = None,
    ):
        """
        Create a rolling detector view.

        Parameters
        ----------
        detector_number:
            Detector number for each pixel. This is used for mapping event_id to pixel.
        window:
            Size of the rolling window. This is the maximum window size, smaller windows
            can be accessed using the `get` method.
        projection:
            Optional projection function to apply to the counts before storing them in
            the history. This can be used to project the data onto a different
            coordinate system or to reduce the dimensionality of the data.
        """
        super().__init__(detector_number=detector_number)
        self._projection = projection
        self._window = window
        self._current = 0
        self._history: sc.DataArray
        self._cache: sc.DataArray
        self._cumulative: sc.DataArray
        self.clear_counts()

    @property
    def max_window(self) -> int:
        return self._window

    @property
    def cumulative(self) -> sc.DataArray:
        return self._cumulative

    def clear_counts(self) -> None:
        """
        Clear counts.

        Overrides Detector.clear_counts, to properly clear sliding window history and
        cache.
        """
        counts = sc.zeros_like(self.data)
        if self._projection is not None:
            counts = self._projection(counts)
        self._history = (
            sc.zeros_like(counts)
            .broadcast(sizes={'window': self._window, **counts.sizes})
            .copy()
        )
        self._cache = self._history.sum('window')
        self._cumulative = sc.zeros_like(self._cache)

    def make_roi_filter(self) -> roi.ROIFilter:
        """Return a ROI filter operating via the projection plane of the view."""
        norm = 1.0
        if isinstance(self._projection, Histogrammer):
            indices = self._projection.input_indices()
            norm = self._projection.replicas
        else:
            indices = sc.ones(sizes=self.data.sizes, dtype='int32', unit=None)
            indices = sc.cumsum(indices, mode='exclusive')
            if self._projection is not None:
                indices = self._projection(indices)
        return roi.ROIFilter(indices=indices, norm=norm)

    def transform_weights(
        self,
        weights: sc.Variable | sc.DataArray | None = None,
        *,
        threshold: float = 0.1,
    ) -> sc.DataArray:
        """
        Transform raw pixel weights to the projection plane.

        Parameters
        ----------
        weights:
            Raw pixel weights to transform. If None, default weights of 1 are used.
        threshold:
            Threshold for identifying bins with a low weight. If the weight is below the
            threshold times the median weight, the bin is marked as invalid. This is
            relevant to avoid issues with color scales in plots, where noise in bins
            with low weight may dominate the color scale if auto-scaling is used.
        """
        if weights is None:
            weights = sc.ones(
                sizes=self.detector_number.sizes, dtype='float32', unit=''
            )
        else:
            if weights.sizes != self.detector_number.sizes:
                raise sc.DimensionError(
                    f'Invalid {weights.sizes=} for {self.detector_number.sizes=}.'
                )
            if isinstance(weights, sc.DataArray):
                if (det_num := weights.coords.get('detector_number')) is not None:
                    if not sc.identical(det_num, self.detector_number):
                        raise sc.CoordError("Mismatching detector numbers in weights.")
                weights = weights.data
        if isinstance(self._projection, Histogrammer):
            xs = self._projection.apply_full(weights)  # Use all replicas
        elif self._projection is not None:
            xs = self._projection(weights)
        else:
            xs = weights.copy()
        nonempty = xs.values[xs.values > 0]
        mask = xs.values < threshold * np.median(nonempty)
        xs.values[mask] = np.nan
        return xs if isinstance(xs, sc.DataArray) else sc.DataArray(xs)

    @staticmethod
    def from_detector_and_histogrammer(
        detector: CalibratedDetector[SampleRun],
        window: RollingDetectorViewWindow,
        projection: Histogrammer,
    ) -> RollingDetectorView:
        """Helper for constructing via a Sciline workflow."""
        return RollingDetectorView(
            detector_number=detector.coords['detector_number'],
            window=window,
            projection=projection,
        )

    @staticmethod
    def from_detector_with_projection(
        projection: Callable[[sc.DataArray], sc.DataArray] | None,
    ) -> Callable[
        [CalibratedDetector[SampleRun], RollingDetectorViewWindow], RollingDetectorView
    ]:
        def factory(
            detector: CalibratedDetector[SampleRun],
            window: RollingDetectorViewWindow,
        ) -> RollingDetectorView:
            """Helper for constructing via a Sciline workflow."""
            return RollingDetectorView(
                detector_number=detector.coords['detector_number'],
                window=window,
                projection=projection,
            )

        return factory

    @staticmethod
    def from_nexus(
        nexus_file: str,
        *,
        detector_name: str,
        window: int,
        projection: Literal['xy_plane', 'cylinder_mantle_z']
        | Callable[[sc.DataArray], sc.DataArray]
        | None = None,
        resolution: dict[str, int] | None = None,
        pixel_noise: Literal['cylindrical'] | sc.Variable | None = None,
    ) -> RollingDetectorView:
        """
        Create a rolling detector view from a NeXus file using GenericNeXusWorkflow.

        The configuration parameters are preliminary for testing and are expected to
        change in the future.

        Parameters
        ----------
        nexus_file:
            NeXus file providing detector parameters such as detector_number and pixel
            positions.
        detector_name:
            Name of the detector in the NeXus file.
        window:
            Size of the rolling window.
        projection:
            Projection to use for the detector data. This can be a string selecting a
            predefined projection or a function that takes a DataArray and returns a
            DataArray. The predefined projections are 'xy_plane' and
            'cylinder_mantle_z'.
        resolution:
            Resolution to use for histogramming the detector data. Only required for
            'xy_plane' and 'cylinder_mantle_z' projections.
        pixel_noise:
            Noise to add to the pixel positions. This can be a scalar value to add
            Gaussian noise to the pixel positions or the string 'cylindrical' to add
            noise to the pixel positions of a cylindrical detector. Adding noise can be
            useful to avoid artifacts when projecting the data.
        """
        if pixel_noise is None:
            pixel_noise = sc.scalar(0.0, unit='m')
            noise_replica_count = 0
        else:
            # Unclear what a good number is, could be made configurable.
            noise_replica_count = 4
        wf = GenericNeXusWorkflow(run_types=[SampleRun], monitor_types=[])
        wf[RollingDetectorViewWindow] = window
        if projection == 'cylinder_mantle_z':
            wf.insert(make_cylinder_mantle_coords)
            wf.insert(RollingDetectorView.from_detector_and_histogrammer)
            wf[DetectorViewResolution] = resolution
        elif projection == 'xy_plane':
            wf.insert(make_xy_plane_coords)
            wf.insert(RollingDetectorView.from_detector_and_histogrammer)
            wf[DetectorViewResolution] = resolution
        else:
            wf[NeXusTransformation[snx.NXdetector, SampleRun]] = NeXusTransformation[
                snx.NXdetector, SampleRun
            ](sc.scalar(1))
            wf.insert(
                RollingDetectorView.from_detector_with_projection(projection=projection)
            )
        if isinstance(pixel_noise, sc.Variable):
            wf.insert(gaussian_position_noise)
            wf[PositionNoiseSigma] = pixel_noise
        elif pixel_noise == 'cylindrical':
            wf.insert(pixel_shape)
            wf.insert(pixel_cylinder_axis)
            wf.insert(pixel_cylinder_radius)
            wf.insert(position_noise_for_cylindrical_pixel)
        else:
            raise ValueError(f"Invalid {pixel_noise=}.")
        wf.insert(position_with_noisy_replicas)
        wf.insert(Histogrammer.from_coords)
        wf[PositionNoiseReplicaCount] = noise_replica_count
        wf[Filename[SampleRun]] = nexus_file
        wf[NeXusDetectorName] = detector_name
        return wf.compute(RollingDetectorView)

    def get(self, *, window: int | None = None) -> sc.DataArray:
        """
        Get the sum of counts over a window of the most recent counts.

        Parameters
        ----------
        window:
            Size of the window to use. If None, the full history is used.

        Returns
        -------
        :
            Sum of counts over the window.
        """
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

    def add_events(self, data: sc.DataArray) -> None:
        """
        Add counts in the form of events grouped by pixel ID.

        Parameters
        ----------
        data:
            Events grouped by pixel ID, given by binned data.
        """
        counts = data.bins.size().to(dtype='int32', copy=False)
        counts.unit = 'counts'
        self._add_counts(counts)

    def add_counts(self, data: Sequence[int]) -> None:
        """
        Add counts in the form of a sequence of pixel IDs.

        Parameters
        ----------
        data:
            List of pixel IDs.
        """
        counts = self.bincount(data)
        self._add_counts(counts)

    def _add_counts(self, counts: sc.Variable) -> None:
        if self._projection is not None:
            counts = self._projection(counts)
        self._cache -= self._history['window', self._current]
        self._history['window', self._current] = counts
        self._cache += counts
        self._cumulative += counts
        self._current = (self._current + 1) % self._window


def project_xy(
    position: sc.Variable, *, zplane: sc.Variable | None = None
) -> dict[str, sc.Variable]:
    """
    Project positions onto the x-y plane, i.e., perpendicular to the beam (z axis).

    This is useful, e.g., for SANS instruments.
    """
    z = position.fields.z
    zmin = z.min()
    zero = sc.zeros_like(zmin)
    if zplane is None:
        zplane = z.max() if zmin < zero else zmin
    t = zplane / z
    return sc.DataGroup(x=position.fields.x * t, y=position.fields.y * t, z=zplane)


def project_onto_cylinder_z(
    position: sc.Variable, *, radius: sc.Variable | None = None
) -> dict[str, sc.Variable]:
    """
    Project positions onto the mantle of a cylinder aligned with the z axis.

    This is useful for cylindrical detectors, provided they are aligned along the beam.
    """
    x = position.fields.x
    y = position.fields.y
    r_xy = sc.sqrt(x**2 + y**2)
    if radius is None:
        radius = r_xy.min()
    t = radius / r_xy
    phi = sc.atan2(y=y, x=x).to(unit='deg')
    arc_length = radius * (phi * sc.scalar(np.pi / 180.0, unit='1/deg'))
    return sc.DataGroup(
        phi=phi, r=radius, z=position.fields.z * t, arc_length=arc_length
    )


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


# Arbitrary small but not tiny size. More is likely not necessary, but having it too
# small might lead to visible patterns after projection. It is therefore also chosen
# to be a prime number, but it likely does not matter.
_noise_size = 107


def position_noise_for_cylindrical_pixel(
    *, axis: PixelCylinderAxis, radius: PixelCylinderRadius
) -> PositionNoise:
    # We *assume* that the cylinder is centered on the origin. Real files may not
    # fulfill this. However, the rest of the data reduction currently assumes that
    # the pixel offset corresponds to the pixel center, so if it is not fulfilled
    # there are bigger problems elsewhere anywhere.
    rng = np.random.default_rng()
    dims = ('position',)
    size = _noise_size
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

    return PositionNoise(dx + dy + dz)


def gaussian_position_noise(sigma: PositionNoiseSigma) -> PositionNoise:
    sigma = sigma.to(unit='m', copy=False)
    size = _noise_size
    position = sc.empty(sizes={'position': size}, unit='m', dtype=sc.DType.vector3)
    position.values = np.random.default_rng(seed=1234).normal(
        0, sigma.value, size=(size, 3)
    )
    return PositionNoise(position)


def position_with_noisy_replicas(
    *,
    detector: CalibratedDetector[SampleRun],
    position_noise: PositionNoise,
    replicas: PositionNoiseReplicaCount,
) -> CalibratedPositionWithNoisyReplicas:
    """
    Create a new position array with noise added to the detector positions.

    The first slice of the new array is the original position data, and the
    remaining slices are the original data with noise added.
    """
    position = detector.coords['position'].to(unit='m')
    if replicas == 0:
        return sc.concat([position], dim='replica')
    noise_dim = position_noise.dim
    size = position.size * replicas
    # "Paint" the short array of noise on top of the (replicated) position data.
    noise = (
        sc.broadcast(
            position_noise,
            sizes={'dummy': ceil(size / position_noise.size), **position_noise.sizes},
        )
        .flatten(to=noise_dim)[:size]
        .fold(dim=noise_dim, sizes={'replica': replicas, **position.sizes})
    )
    return sc.concat([position, noise + position], dim='replica')


def make_xy_plane_coords(
    position: CalibratedPositionWithNoisyReplicas,
) -> ProjectedCoords:
    # The first slice is the original data, so we use it to determine the z plane.
    # This avoids noise in the z plane which could later cause trouble when
    # combining the data.
    zplane = project_xy(position['replica', 0])['z']
    return project_xy(position, zplane=zplane)


def make_cylinder_mantle_coords(
    position: CalibratedPositionWithNoisyReplicas,
) -> ProjectedCoords:
    radius = project_onto_cylinder_z(position['replica', 0])['r']
    return project_onto_cylinder_z(position, radius=radius)
