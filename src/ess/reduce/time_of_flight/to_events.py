# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from functools import reduce

import numpy as np
import scipp as sc


def to_events(
    da: sc.DataArray, event_dim: str, events_per_bin: int = 500
) -> sc.DataArray:
    """
    Convert a histogrammed data array to an event list.
    The generated events have a uniform distribution within each bin.
    Each dimension with a bin-edge coordinate is converted to an event coordinate.
    The contract is that if we re-histogram the event list with the same bin edges,
    we should get the original counts back.
    Masks on non-bin-edge dimensions are preserved.
    If there are masks on bin-edge dimensions, the masked values are zeroed out in the
    original data before the conversion to events.

    Parameters
    ----------
    da:
        DataArray to convert to events.
    event_dim:
        Name of the new event dimension.
    events_per_bin:
        Number of events to generate per bin.
    """
    if da.bins is not None:
        raise ValueError("Cannot convert a binned DataArray to events.")
    rng = np.random.default_rng()
    event_coords = {}
    edge_dims = []
    midp_dims = set(da.dims)
    midp_coord_names = []
    # Separate bin-edge and midpoints coords
    for name in da.coords:
        dims = da.coords[name].dims
        is_edges = False if not dims else da.coords.is_edges(name)
        if is_edges:
            if name in dims:
                edge_dims.append(name)
                midp_dims -= {name}
        else:
            midp_coord_names.append(name)

    edge_sizes = {dim: da.sizes[da.coords[dim].dim] for dim in edge_dims}
    for dim in edge_dims:
        coord = da.coords[dim]
        left = sc.broadcast(coord[dim, :-1], sizes=edge_sizes).values
        right = sc.broadcast(coord[dim, 1:], sizes=edge_sizes).values

        # The numpy.random.uniform function below does not support NaNs, so we need to
        # replace them with zeros, and then replace them back after the random numbers
        # have been generated.
        nans = np.isnan(left) | np.isnan(right)
        left = np.where(nans, 0.0, left)
        right = np.where(nans, 0.0, right)
        # Ensure left <= right
        left, right = np.minimum(left, right), np.maximum(left, right)

        # In each bin, we generate a number of events with a uniform distribution.
        events = rng.uniform(
            left, right, size=(events_per_bin, *list(edge_sizes.values()))
        )
        events[..., nans] = np.nan
        event_coords[dim] = sc.array(
            dims=[event_dim, *edge_dims], values=events, unit=coord.unit
        )

    # Find and apply masks that are on a bin-edge dimension
    event_masks = {}
    other_masks = {}
    edge_dims_set = set(edge_dims)
    for key, mask in da.masks.items():
        if set(mask.dims) & edge_dims_set:
            event_masks[key] = mask
        else:
            other_masks[key] = mask

    data = da.data
    if event_masks:
        inv_mask = (~reduce(lambda a, b: a | b, event_masks.values())).to(dtype=int)
        inv_mask.unit = ""
        data = data * inv_mask

    # Create the data counts, which are the original counts divided by the number of
    # events per bin
    sizes = {event_dim: events_per_bin} | da.sizes
    val = sc.broadcast(sc.values(data) / float(events_per_bin), sizes=sizes)
    kwargs = {"dims": sizes.keys(), "values": val.values, "unit": data.unit}
    if data.variances is not None:
        # Note here that all the events are correlated.
        # If we later histogram the events with different edges than the original
        # histogram, then neighboring bins will be correlated, and the error obtained
        # will be too small. It is however not clear what can be done to improve this.
        kwargs["variances"] = sc.broadcast(
            sc.variances(data) / float(events_per_bin), sizes=sizes
        ).values
    new_data = sc.array(**kwargs)

    new = sc.DataArray(data=new_data, coords=event_coords)
    new = new.transpose((*midp_dims, *edge_dims, event_dim)).flatten(
        dims=[*edge_dims, event_dim], to=event_dim
    )
    return new.assign_coords(
        {dim: da.coords[dim].copy() for dim in midp_coord_names}
    ).assign_masks({key: mask.copy() for key, mask in other_masks.items()})
