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
    for dim, (low, high) in intervals.items():
        indices = indices[dim, low:high]
    indices = indices.flatten(to=out_dim)
    if indices.bins is None:
        return indices
    indices = indices.bins.concat().value
    return indices.rename_dims({indices.dim: out_dim})


T = TypeVar('T', sc.DataArray, sc.Variable)


def apply_selection(data: T, *, selection: sc.Variable) -> T:
    """
    Apply selection to data.
    """
    indices, counts = np.unique(selection.values, return_counts=True)
    if data.ndim != 1:
        data = data.flatten(to='detector_number')
    scale = sc.array(dims=[data.dim], values=counts, dtype=data.dtype)
    return data[indices] * scale


class ROIFilter:
    def __init__(self, indices: sc.Variable):
        self._indices = indices
        self._selection = sc.array(dims=['index'], values=[])

    def set_roi_from_intervals(self, intervals: sc.DataGroup) -> None:
        self._selection = select_indices_in_intervals(intervals, self._indices)

    def apply(self, data: T) -> T:
        return apply_selection(data, selection=self._selection)
