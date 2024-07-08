# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""Tools for handling statistical uncertainties.

This module provides tools for handling statistical uncertainties in the context of
data reduction. Handling variances during broadcast operations is not handled
correctly by Scipp because correlations are not tracked.
See https://doi.org/10.3233/JNR-220049 for context.

This module provides two ways of handling variances during broadcast operations:
- `drop_variances_if_broadcast`: Drop variances if the data is broadcasted.
- `broadcast_with_upper_bound_variances`: Compute an upper bound for the variances.
"""

from enum import Enum
from typing import TypeVar, Union, overload

import numpy as np
import scipp as sc
from scipp.core.concepts import irreducible_mask

T = TypeVar("T", bound=Union[sc.Variable, sc.DataArray])


UncertaintyBroadcastMode = Enum(
    'UncertaintyBroadcastMode', ['drop', 'upper_bound', 'fail']
)
"""
Mode for broadcasting uncertainties.

- `drop`: Drop variances if the data is broadcasted.
- `upper_bound`: Compute an upper bound for the variances.
- `fail`: Do not broadcast, simply return the input data.

See https://doi.org/10.3233/JNR-220049 for context.
"""


@overload
def broadcast_with_upper_bound_variances(
    data: sc.Variable, prototype: sc.DataArray | sc.Variable
) -> sc.Variable:
    pass


@overload
def broadcast_with_upper_bound_variances(
    data: sc.DataArray, prototype: sc.DataArray | sc.Variable
) -> sc.DataArray:
    pass


def broadcast_with_upper_bound_variances(
    data: Union[sc.Variable, sc.DataArray], prototype: sc.DataArray | sc.Variable
) -> Union[sc.Variable, sc.DataArray]:
    """
    Compute an upper bound for the variances of the broadcasted data.

    The variances of the broadcasted data are computed by scaling the variances of the
    input data by the volume of the new subspace. The volume of the new subspace is
    computed as the product of the sizes of the new dimensions.

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
    if _no_variance_broadcast(data, prototype):
        return data
    sizes = prototype.sizes
    mask = sc.scalar(False)
    if isinstance(prototype, sc.DataArray):
        if (irred := irreducible_mask(prototype, dim=sizes)) is not None:
            for dim in data.dims:
                if dim in irred.dims:
                    irred = irred.all(dim)
            mask = irred
    size = (~mask).sum().value
    for dim, dim_size in sizes.items():
        if dim not in data.dims and dim not in mask.dims:
            size *= dim_size
    data = data.copy()
    data.variances *= size
    sizes = {**sizes, **data.sizes}
    data = data.broadcast(sizes=sizes).copy()
    if mask is not None:
        # The masked values are not counted in the variance, so we set them to infinity.
        data.variances[mask.broadcast(sizes=sizes).values] = np.inf
    return data


@overload
def drop_variances_if_broadcast(
    data: sc.Variable, prototype: sc.DataArray | sc.Variable
) -> sc.Variable:
    pass


@overload
def drop_variances_if_broadcast(
    data: sc.DataArray, prototype: sc.DataArray | sc.Variable
) -> sc.DataArray:
    pass


def drop_variances_if_broadcast(
    data: Union[sc.Variable, sc.DataArray], prototype: sc.DataArray | sc.Variable
) -> Union[sc.Variable, sc.DataArray]:
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
    if _no_variance_broadcast(data, prototype):
        return data
    return sc.values(data)


def _no_variance_broadcast(
    data: sc.Variable | sc.DataArray, prototype: sc.Variable | sc.DataArray
) -> bool:
    if data.bins is not None:
        raise ValueError("Cannot broadcast binned data.")
    if data.variances is None:
        return True
    if prototype.bins is not None:
        return False
    sizes = prototype.sizes
    return all(data.sizes.get(dim) == size for dim, size in sizes.items())


# TODO: For now, we only have broadcasters for dense data. Event-data broadcasters will
# be added at a later stage, as we currently only have one which is valid for SANS.

broadcasters = {
    UncertaintyBroadcastMode.drop: drop_variances_if_broadcast,
    UncertaintyBroadcastMode.upper_bound: broadcast_with_upper_bound_variances,
    UncertaintyBroadcastMode.fail: lambda x, _: x,
}
