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

    # Find indices where values are strictly increasing
    sections = []
    start_idx = 0
    in_section = False

    diffs = np.diff(values)
    signs = np.sign(diffs)
    is_increasing = signs > 0 & np.isfinite(values[:-1]) & np.isfinite(values[1:])

    for i in range(1, len(values)):
        increasing = is_increasing[i - 1]

        if increasing and not in_section:
            # Start of a new section
            start_idx = i - 1
            in_section = True
        elif not increasing and in_section:
            # End of a section
            if i - start_idx >= 2:  # Ensure section has at least 2 points
                sections.append(slice(start_idx, i))
            in_section = False

    # Don't forget to handle the case where the sequence is increasing at the end
    if in_section and len(values) - start_idx >= 2:
        sections.append(slice(start_idx, len(values)))

    return sections
