# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Utilities for region of interest (ROI) selection."""

from __future__ import annotations

from typing import TypeVar

import numpy as np
import scipp as sc


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


T = TypeVar('T', sc.DataArray, sc.Variable)


def apply_selection(
    data: T, *, selection: sc.Variable, norm: float = 1.0
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

    Returns
    -------
    :
        Filtered data and scale factor.
    """
    indices, counts = np.unique(selection.values, return_counts=True)
    if data.ndim != 1:
        data = data.flatten(to='detector_number')
    scale = sc.array(dims=[data.dim], values=counts) / norm
    return data[indices], scale


class ROIFilter:
    """Filter for selecting a region of interest (ROI)."""

    def __init__(self, indices: sc.Variable | sc.DataArray, norm: float = 1.0) -> None:
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
        """
        self._indices = indices
        self._selection = sc.array(dims=['index'], values=[])
        self._norm = norm

    def set_roi_from_intervals(self, intervals: sc.DataGroup) -> None:
        """Set the ROI from (typically 1 or 2) intervals."""
        self._selection = select_indices_in_intervals(intervals, self._indices)

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
        return apply_selection(data, selection=self._selection, norm=self._norm)
