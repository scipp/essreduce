# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)


import numpy as np
import scipp as sc


def rebin_strictly_increasing(da: sc.DataArray, dim: str) -> sc.DataArray:
    """
    Find strictly monotonic sections in a coordinate dimension and rebin the data array
    into a regular grid based on these sections.
    """
    pass


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
