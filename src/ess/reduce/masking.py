# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import uuid
from typing import Optional

import numpy as np
import scipp as sc


def mask_range(
    da: sc.DataArray, mask: Optional[sc.DataArray], name: Optional[str] = None
) -> sc.DataArray:
    """
    Mask a range on a data array.
    The provided edges are used to define the ranges to be masked.

    Parameters
    ----------
    da:
        The data array to be masked.
    mask:
        A data array defining the mask to be applied. Only one-dimensional masks are
        supported. The data array should contain a bin-edge coordinate which represents
        the edges of the ranges to be masked. The values of the data array represent the
        mask values (``True`` or ``False``) inside each range defined by the coordinate.
    name:
        The name of the mask to be applied. If not provided, a random name will be used.

    Returns
    -------
    :
        A copy of the input data array with the mask applied.
    """
    if mask is None:
        return da
    if name is None:
        name = uuid.uuid4().hex
    if name in da.masks:
        raise ValueError(
            f'Mask {name} already exists in data array and would be overwritten.'
        )
    dim = mask.dim
    edges = mask.coords[dim]
    if not mask.coords.is_edges(dim):
        raise sc.DimensionError(
            f'Coordinate {dim} must be bin-edges to mask a range, found midpoints.'
        )
    if (dim in da.coords) and (da.coords[dim].ndim > 1):
        raise sc.DimensionError(
            'Cannot mask range on data with multi-dimensional coordinate. '
            f'Found dimensions {da.coords[dim].dims} for coordinate {dim}.'
        )

    coord = (
        da.bins.constituents['data'].coords[dim]
        if da.bins is not None
        else da.coords[dim]
    )
    edges = edges.to(unit=coord.unit)
    lu = sc.DataArray(data=mask.data, coords={dim: edges})
    if da.bins is not None:
        if dim not in da.coords:
            underlying = da.bins.coords[dim]
            new_bins = np.union1d(
                edges.values,
                np.array(
                    [
                        underlying.min().value,
                        np.nextafter(underlying.max().value, np.inf),
                    ]
                ),
            )
        else:
            new_bins = np.union1d(edges.values, da.coords[dim].values)
        new_bins = sc.array(dims=[dim], values=new_bins, unit=edges.unit)
        out = da.bin({dim: new_bins})
        out.masks[name] = sc.lookup(lu, dim)[sc.midpoints(new_bins, dim=dim)]
    else:
        out = da.copy(deep=False)
        mask_values = sc.lookup(lu, dim)[da.coords[dim]]
        if da.coords.is_edges(dim):
            out.masks[name] = mask_values[dim, 1:] | mask_values[dim, :-1]
        else:
            out.masks[name] = mask_values
    return out
