# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)


import numpy as np
import scipp as sc


def find_strictly_increasing_sections(var: sc.Variable) -> list[slice]:
    """
    Find strictly increasing sections in a coordinate dimension (minimum length 2).

    Parameters
    ----------
    var:
        The variable to analyze, which should be one-dimensional.

    Returns
    -------
    sections:
        Slice objects that can be used extract strictly increasing sections.
    """
    values = var.values
    finite = np.isfinite(values)
    increasing = (np.sign(np.diff(values)) > 0) & finite[:-1] & finite[1:]
    # 1 marks the start of an increasing section, -1 marks the end
    transitions = np.diff(np.concatenate(([False], increasing, [False])).astype(int))
    section_starts = np.where(transitions == 1)[0]
    section_ends = np.where(transitions == -1)[0] + np.array(1)
    return [
        slice(start, end)
        for start, end in zip(section_starts, section_ends, strict=True)
        if end - start >= 2  # Ensure section has at least 2 points
    ]


def get_min_max(
    var: sc.Variable, *, dim: str, slices: list[slice]
) -> tuple[sc.Variable, sc.Variable]:
    if not slices:
        raise ValueError("No strictly increasing sections found.")
    combined = sc.concat([var[dim, slice] for slice in slices], dim)
    return combined.min(), combined.max()


def make_regular_grid(
    var: sc.Variable, *, dim: str, slices: list[slice]
) -> sc.Variable:
    """
    Create a regular grid variable based on the min and max of the slices.

    The grid is constructed such that it includes the minimum and maximum values
    of the strictly increasing sections, with a step size equal to the difference
    between the first two values of the section with the minimum start value (which is
    not necessarily the first section).
    """
    min_val, max_val = get_min_max(var, dim=dim, slices=slices)
    first: sc.Variable | None = None
    for s in slices:
        first = var[dim, s]
        if sc.identical(first[0], min_val):
            break
    if first is None:
        # This should not happen if slices are correctly identified and passed from
        # find_strictly_increasing_sections.
        raise ValueError("Section is not strictly increasing.")
    step = first[1] - first[0]
    return sc.arange(
        dim=dim,
        start=min_val.value,
        stop=max_val.value + step.value,  # Ensure the last bin edge is included
        step=step.value,
        unit=step.unit,
        dtype=step.dtype,
    )


def rebin_strictly_increasing(da: sc.DataArray, dim: str) -> sc.DataArray:
    """
    Find strictly monotonic sections in a coordinate dimension and rebin the data array
    into a regular grid based on these sections.
    """
    # Ensure the dimension is named like the coordinate.
    da = da.rename_dims({da.coords[dim].dim: dim})
    slices = find_strictly_increasing_sections(da.coords[dim])
    if len(slices) == 1:
        return da[dim, slices[0]]
    if not slices:
        raise ValueError("No strictly increasing sections found.")
    if da.coords[dim].dtype not in (sc.DType.float64, sc.DType.float32):
        # rebin does not like integer coords.
        da = da.assign_coords({dim: da.coords[dim].to(dtype='float64')})
    # Slices refer to the indices in the coord, which are bin edges. For slicing data
    # we need to stop at the last index minus one.
    sections = [da[dim, section.start : section.stop - 1] for section in slices]
    edges = make_regular_grid(da.coords[dim], dim=dim, slices=slices)
    return sc.reduce([sc.rebin(section, {dim: edges}) for section in sections]).sum()
