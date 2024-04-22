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
from typing import Dict, TypeVar, Union, overload

import scipp as sc

T = TypeVar("T", bound=Union[sc.Variable, sc.DataArray])


UncertaintyBroadcastMode = Enum(
    'UncertaintyBroadcastMode', ['drop', 'upper_bound', 'fail']
)
"""
Mode for broadcasting uncertainties.

See https://doi.org/10.3233/JNR-220049 for context.
"""


@overload
def broadcast_with_upper_bound_variances(
    data: sc.Variable, sizes: Dict[str, int]
) -> sc.Variable:
    pass


@overload
def broadcast_with_upper_bound_variances(
    data: sc.DataArray, sizes: Dict[str, int]
) -> sc.DataArray:
    pass


def broadcast_with_upper_bound_variances(
    data: Union[sc.Variable, sc.DataArray], sizes: Dict[str, int]
) -> Union[sc.Variable, sc.DataArray]:
    if _no_variance_broadcast(data, sizes):
        return data
    size = 1
    for dim, dim_size in sizes.items():
        if dim not in data.dims:
            size *= dim_size
    data = data.copy()
    data.variances *= size
    return data.broadcast(sizes={**sizes, **data.sizes}).copy()


@overload
def drop_variances_if_broadcast(
    data: sc.Variable, sizes: Dict[str, int]
) -> sc.Variable:
    pass


@overload
def drop_variances_if_broadcast(
    data: sc.DataArray, sizes: Dict[str, int]
) -> sc.DataArray:
    pass


def drop_variances_if_broadcast(
    data: Union[sc.Variable, sc.DataArray], sizes: Dict[str, int]
) -> Union[sc.Variable, sc.DataArray]:
    if _no_variance_broadcast(data, sizes):
        return data
    return sc.values(data)


def _no_variance_broadcast(
    data: Union[sc.Variable, sc.DataArray], sizes: Dict[str, int]
) -> bool:
    return (data.variances is None) or all(
        data.sizes.get(dim) == size for dim, size in sizes.items()
    )


# TODO: For now, we only have broadcasters for dense data. Event-data broadcasters will
# be added at a later stage, as we currently only have one which is valid for SANS.

broadcasters = {
    UncertaintyBroadcastMode.drop: drop_variances_if_broadcast,
    UncertaintyBroadcastMode.upper_bound: broadcast_with_upper_bound_variances,
    UncertaintyBroadcastMode.fail: lambda x, sizes: x,
}
