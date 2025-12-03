# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Utilities for region of interest (ROI) selection."""

from __future__ import annotations

from collections.abc import Sequence
from typing import TypeVar

import numpy as np
import scipp as sc
from matplotlib.path import Path

# Type for polygon vertices: either scipp Variable (coord-based) or sequence of
# floats (index-based)
PolygonVertices = sc.Variable | Sequence[float]


def select_indices_in_intervals(
    intervals: sc.DataGroup[tuple[int, int] | tuple[sc.Variable, sc.Variable]],
    indices: sc.Variable | sc.DataArray,
) -> sc.Variable:
    """
    Return subset of indices that fall within the intervals.

    Parameters
    ----------
    intervals:
        DataGroup with dimension names as keys and tuples of low and high values. This
        can be used to define a band or a rectangle to selected. When low and high are
        scipp.Variable, the selection is done using label-based indexing. In this case
        `indices` must be a DataArray with corresponding coordinates.
    indices:
        Variable or DataArray with indices to select from. If binned data the selected
        indices will be returned concatenated into a dense array.
    """
    out_dim = 'index'
    for dim, bounds in intervals.items():
        low, high = sorted(bounds)
        indices = indices[dim, low:high]
    indices = indices if isinstance(indices, sc.Variable) else indices.data
    indices = indices.flatten(to=out_dim)
    if indices.bins is None:
        return indices
    indices = indices.bins.concat().value
    return indices.rename_dims({indices.dim: out_dim})


def _get_polygon_axis_data(
    axis_name: str,
    vertices: PolygonVertices,
    indices: sc.DataArray,
) -> tuple[np.ndarray, sc.Variable]:
    """
    Get polygon vertices and corresponding coordinates for one axis.

    Parameters
    ----------
    axis_name:
        Name of the axis (dimension or coordinate name).
    vertices:
        Polygon vertices for this axis - either a scipp Variable (coord-based)
        or a sequence of floats (index-based).
    indices:
        DataArray with indices to select from.

    Returns
    -------
    :
        Tuple of (polygon_vertices_array, coordinate_values).
    """
    if isinstance(vertices, sc.Variable):
        # Coord-based: use named coordinate from indices
        if axis_name not in indices.coords:
            raise KeyError(
                f"Coordinate '{axis_name}' not found in indices. "
                f"Available coordinates: {list(indices.coords.keys())}"
            )
        coords = indices.coords[axis_name]
        if indices.coords.is_edges(axis_name):
            coords = sc.midpoints(coords, dim=axis_name)
        # Validate units match
        if vertices.unit != coords.unit:
            raise sc.UnitError(
                f"Unit mismatch for '{axis_name}': "
                f"polygon has unit '{vertices.unit}' "
                f"but coordinates have unit '{coords.unit}'"
            )
        return vertices.values, coords
    else:
        # Index-based: use dimension indices as coordinates
        if axis_name not in indices.sizes:
            raise KeyError(
                f"Dimension '{axis_name}' not found in indices. "
                f"Available dimensions: {list(indices.dims)}"
            )
        # Create coordinate from dimension indices
        coords = sc.arange(axis_name, indices.sizes[axis_name], dtype='float64')
        return np.asarray(vertices, dtype='float64'), coords


def select_indices_in_polygon(
    polygon: dict[str, PolygonVertices],
    indices: sc.DataArray,
) -> sc.Variable:
    """
    Return subset of indices that fall within the polygon.

    Parameters
    ----------
    polygon:
        Polygon vertices as a dict mapping axis names to 1-D arrays of vertex
        positions. Must contain exactly two entries. Each entry can be either:

        - A scipp Variable for coordinate-based selection (uses named coordinates
          from the indices DataArray, with unit validation)
        - A sequence of floats for index-based selection (uses dimension indices
          as coordinates, no unit handling)

        The two axes can independently use either mode.
    indices:
        DataArray with indices to select from. For coordinate-based axes, must have
        coordinates matching the axis names. For index-based axes, must have
        dimensions matching the axis names.

    Returns
    -------
    :
        Variable with selected indices.
    """
    out_dim = 'index'

    if len(polygon) != 2:
        raise ValueError(
            f"Polygon must have exactly two coordinate arrays, got {len(polygon)}"
        )

    # Get the two axis names from the polygon dict
    axis_a, axis_b = polygon.keys()

    # Get polygon vertices and coordinates for each axis
    poly_a, a_coords = _get_polygon_axis_data(axis_a, polygon[axis_a], indices)
    poly_b, b_coords = _get_polygon_axis_data(axis_b, polygon[axis_b], indices)

    # Extract polygon vertices as 2D array
    vertices_2d = np.column_stack([poly_a, poly_b])

    # Broadcast coordinates to match indices shape and flatten
    a_flat = sc.broadcast(a_coords, sizes=indices.sizes).values.flatten()
    b_flat = sc.broadcast(b_coords, sizes=indices.sizes).values.flatten()
    points = np.column_stack([a_flat, b_flat])

    # Use matplotlib Path for point-in-polygon test
    polygon_path = Path(vertices_2d)
    mask = polygon_path.contains_points(points)

    # Get indices that are inside the polygon
    all_indices = indices.data.flatten(to=out_dim)

    # Apply mask first, then concat if binned (mask is per-bin, not per-index)
    sc_mask = sc.array(dims=[out_dim], values=mask)
    selected = all_indices[sc_mask]

    if selected.bins is not None:
        selected = selected.bins.concat().value
        selected = selected.rename_dims({selected.dim: out_dim})

    return sc.array(dims=[out_dim], values=selected.values, dtype='int32', unit=None)


T = TypeVar('T', sc.DataArray, sc.Variable)


def apply_selection(
    data: T,
    *,
    selection: sc.Variable,
    norm: float = 1.0,
    spatial_dims: tuple[str, ...] | None = None,
) -> tuple[T, sc.Variable]:
    """
    Apply selection to data.

    Parameters
    ----------
    data:
        Data to filter.
    selection:
        Variable with indices to select.
    norm:
        Normalization factor to apply to the selected data. This is used for cases where
        indices may be selected multiple times.
    spatial_dims:
        Dimensions to flatten into 'detector_number'. If None, all dims are flattened.
        For dense data like (time, x, y), pass ('x', 'y') to preserve time.

    Returns
    -------
    :
        Filtered data and scale factor.
    """
    indices, counts = np.unique(selection.values, return_counts=True)
    if spatial_dims is None:
        dims_to_flatten = data.dims
    else:
        dims_to_flatten = tuple(d for d in data.dims if d in spatial_dims)
    if len(dims_to_flatten) > 0 and data.dims != ('detector_number',):
        data = data.flatten(dims=dims_to_flatten, to='detector_number')
    scale = sc.array(dims=['detector_number'], values=counts) / norm
    return data['detector_number', indices], scale


class ROIFilter:
    """Filter for selecting a region of interest (ROI)."""

    def __init__(
        self,
        indices: sc.Variable | sc.DataArray,
        norm: float = 1.0,
        spatial_dims: tuple[str, ...] | None = None,
    ) -> None:
        """
        Create a new ROI filter.

        Parameters
        ----------
        indices:
            Variable with indices to filter. The indices facilitate selecting a 2-D
            ROI in a projection of a 3-D dataset. Typically the indices are given by a
            2-D array. Each element in the array may correspond to a single index (when
            there is no projection) or a list of indices that were projected into an
            output pixel.
        spatial_dims:
            Dimensions of the detector that should be flattened when applying the
            filter. If None, defaults to indices.dims. For projections where indices
            represent a subset of the detector, this should be set to the full
            detector dimensions.
        """
        self._indices = indices
        self._selection = sc.array(dims=['index'], values=[])
        self._norm = norm
        self._spatial_dims = spatial_dims if spatial_dims is not None else indices.dims

    def set_roi_from_intervals(self, intervals: sc.DataGroup) -> None:
        """Set the ROI from (typically 1 or 2) intervals."""
        self._selection = select_indices_in_intervals(intervals, self._indices)

    def set_roi_from_polygon(self, polygon: dict[str, PolygonVertices]) -> None:
        """
        Set the ROI from polygon vertices.

        Parameters
        ----------
        polygon:
            Polygon vertices as a dict mapping axis names to 1-D arrays of vertex
            positions. Must contain exactly two entries. Each entry can be either:

            - A scipp Variable for coordinate-based selection (uses named coordinates
              from the indices DataArray, with unit validation)
            - A sequence of floats for index-based selection (uses dimension indices
              as coordinates, no unit handling)

            The two axes can independently use either mode.
        """
        if not isinstance(self._indices, sc.DataArray):
            raise TypeError("Polygon ROI requires indices to be a DataArray")
        self._selection = select_indices_in_polygon(
            polygon=polygon,
            indices=self._indices,
        )

    @property
    def spatial_dims(self) -> tuple[str, ...]:
        """Dimensions that define the spatial extent of the ROI."""
        return self._spatial_dims

    def apply(self, data: T) -> tuple[T, sc.Variable]:
        """
        Apply the ROI filter to data.

        The returned scale factor can be used to handle filtering via a projection, to
        take into account that fractions of source data point contribute to a data point
        in the projection.

        Parameters
        ----------
        data:
            Data to filter.

        Returns
        -------
        :
            Filtered data and scale factor.
        """
        return apply_selection(
            data,
            selection=self._selection,
            norm=self._norm,
            spatial_dims=self.spatial_dims,
        )
