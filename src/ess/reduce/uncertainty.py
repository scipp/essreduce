# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Tools for handling statistical uncertainties.

This module provides tools for handling statistical uncertainties in the context of
data reduction. Handling variances during broadcast operations is not handled
correctly by Scipp because correlations are not tracked.
See https://doi.org/10.3233/JNR-220049 for context.

This module provides three ways of handling variances during broadcast operations,
defined by :py:class:`UncertaintyBroadcastMode`.
The recommended use of this module is via the :py:func:`broadcast_uncertainties`
helper function.
"""

from enum import Enum, auto
from typing import TypeVar, overload

import numpy as np
import scipp as sc
from scipp.core.concepts import irreducible_mask

T = TypeVar("T", bound=sc.Variable | sc.DataArray)


class UncertaintyBroadcastMode(Enum):
    """Mode for broadcasting uncertainties.

    See https://doi.org/10.3233/JNR-220049 for context.
    """

    drop = auto()
    """Drop variances if the data is broadcast."""
    upper_bound = auto()
    """Compute an upper bound for the variances."""
    fail = auto()
    """Do not broadcast, simply return the input data."""


@overload
def broadcast_with_upper_bound_variances(
    data: sc.Variable, /, *, prototype: sc.DataArray | sc.Variable
) -> sc.Variable:
    pass


@overload
def broadcast_with_upper_bound_variances(
    data: sc.DataArray, /, *, prototype: sc.DataArray | sc.Variable
) -> sc.DataArray:
    pass


def broadcast_with_upper_bound_variances(
    data: sc.Variable | sc.DataArray, /, *, prototype: sc.DataArray | sc.Variable
) -> sc.Variable | sc.DataArray:
    """
    Compute an upper bound for the variances of the broadcasted data.

    The variances of the broadcasted data are computed by scaling the variances of the
    input data by the volume of the new subspace. The volume of the new subspace is
    computed as the product of the sizes of the new dimensions. In the case of an
    event-data prototype the events are counted.

    Parameters
    ----------
    data:
        The data to broadcast.
    prototype:
        Defines the new sizes (dims and shape). If present, masks are used to exclude
        masked values from the variance computation.

    Returns
    -------
    :
        The data with the variances scaled by the volume of the new subspace.
    """
    if _no_variance_broadcast(data, prototype=prototype):
        return data
    for dim in prototype.dims:
        coord1 = None if isinstance(data, sc.Variable) else data.coords.get(dim)
        coord2 = (
            None if isinstance(prototype, sc.Variable) else prototype.coords.get(dim)
        )
        if coord1 is None or coord2 is None:
            if dim in data.dims:
                if data.sizes[dim] != prototype.sizes[dim]:
                    raise ValueError("Mismatching binning not supported in broadcast.")
            continue
        elif sc.identical(coord1, coord2):
            continue
        raise ValueError("Mismatching binning not supported in broadcast.")
    sizes = prototype.sizes
    mask = sc.scalar(False)
    if isinstance(prototype, sc.DataArray):
        if (irred := irreducible_mask(prototype, dim=sizes)) is not None:
            for dim in data.dims:
                if dim in irred.dims:
                    irred = irred.all(dim)
            mask = irred
    data = data.copy()
    sizes = {**sizes, **data.sizes}
    if prototype.bins is None:
        size = (~mask).sum().to(dtype='int64', copy=False)
        for dim, dim_size in sizes.items():
            if dim not in data.dims and dim not in mask.dims:
                size *= sc.index(dim_size)
    else:
        size = prototype.bins.size().sum(set(prototype.dims) - set(data.dims))
    scale = size.broadcast(sizes=sizes).to(dtype='float64')
    if not sc.identical(mask, sc.scalar(False)):
        # The masked values are not counted in the variance, so we set them to infinity.
        scale.values[mask.broadcast(sizes=sizes).values] = np.inf
    data = data.broadcast(sizes=sizes).copy()
    data.variances *= scale.values
    if prototype.bins is not None:
        # Note that we are not using event masks in the upper-bound computation. Less
        # than optimal, but simpler.
        if isinstance(data, sc.Variable):
            data = sc.bins_like(prototype, data)
        else:
            data.data = sc.bins_like(prototype, data.data)
    return data


@overload
def drop_variances_if_broadcast(
    data: sc.Variable, /, *, prototype: sc.DataArray | sc.Variable
) -> sc.Variable:
    pass


@overload
def drop_variances_if_broadcast(
    data: sc.DataArray, /, *, prototype: sc.DataArray | sc.Variable
) -> sc.DataArray:
    pass


def drop_variances_if_broadcast(
    data: sc.Variable | sc.DataArray, /, *, prototype: sc.DataArray | sc.Variable
) -> sc.Variable | sc.DataArray:
    """
    Drop variances if the data is broadcasted.

    Parameters
    ----------
    data:
        The data to broadcast.
    prototype:
        Defines the new sizes (dims and shape).

    Returns
    -------
    :
        The data without variances if the data is broadcasted.
    """
    if _no_variance_broadcast(data, prototype=prototype):
        return data
    return sc.values(data)


def _no_variance_broadcast(
    data: sc.Variable | sc.DataArray, /, *, prototype: sc.Variable | sc.DataArray
) -> bool:
    if data.bins is not None:
        raise ValueError("Cannot broadcast binned data.")
    if data.variances is None:
        return True
    if prototype.bins is not None:
        return False
    sizes = prototype.sizes
    return all(data.sizes.get(dim) == size for dim, size in sizes.items())


def _fail(
    data: sc.Variable | sc.DataArray, /, *, prototype: sc.Variable | sc.DataArray
) -> sc.Variable | sc.DataArray:
    # If there are variances, a subsequent broadcasting operation using Scipp will fail.
    # Do nothing here.
    return data


broadcasters = {
    UncertaintyBroadcastMode.drop: drop_variances_if_broadcast,
    UncertaintyBroadcastMode.upper_bound: broadcast_with_upper_bound_variances,
    UncertaintyBroadcastMode.fail: _fail,
}


def broadcast_uncertainties(
    data: sc.Variable | sc.DataArray,
    /,
    *,
    prototype: sc.DataArray | sc.Variable,
    mode: UncertaintyBroadcastMode,
) -> sc.Variable | sc.DataArray:
    """Broadcast uncertainties using the specified mode.

    Since Scipp raises an error when broadcasting data with variances, this function
    provides an explicit way to handle variances during broadcast operations.

    Parameters
    ----------
    data:
        Data with uncertainties to broadcast.
    prototype:
        Prototype defining the new sizes (dims and shape, or binned data sizes).
    mode:
        Selected broadcast mode.

    Returns
    -------
    :
        Data with broadcast uncertainties.
    """
    return broadcasters[mode](data, prototype=prototype)
