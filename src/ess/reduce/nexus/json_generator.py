# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Generators for "JSON" data from a NeXus file, for the purpose of testing."""

from collections.abc import Generator
from typing import Any

import scipp as sc


def _as_str_attr(value: Any, name: str) -> dict:
    val = str(value)
    return {"string_size": len(val), "type": "string", "name": name, "values": val}


def _variable_to_json(var: sc.Variable, name: str):
    attrs = []
    if var.dtype == sc.DType.datetime64:
        offset = var.min()
        attrs.append(_as_str_attr(offset.value, "offset"))
        var = var - offset
    if var.unit is not None:
        attrs.append(_as_str_attr(var.unit, "units"))
    return {
        "module": "dataset",
        "config": {
            "name": name,
            "values": var.values,
            "size": list(var.shape),
            "type": str(var.dtype),
        },
        "attributes": attrs,
    }


_event_index_0 = sc.array(dims=('dummy',), values=[0], unit=None)


def _event_data_pulse_to_json(pulse: sc.DataArray) -> dict:
    content = pulse.value.coords
    event_time_zero = sc.concat([pulse.coords['event_time_zero']], 'dummy')
    event_time_offset = content['event_time_offset']
    event_id = content.get('event_id')
    # I think we always have a pixel_id in the flatbuffer, so monitors just get ones?
    if event_id is None:
        event_id = sc.ones(sizes=event_time_offset.sizes, dtype='int32', unit=None)
    children = [
        _variable_to_json(event_time_zero, name='event_time_zero'),
        _variable_to_json(event_time_offset, name='event_time_offset'),
        _variable_to_json(event_id, name='event_id'),
        _variable_to_json(_event_index_0, name='event_index'),
    ]
    group = {
        "type": "group",
        "name": "events_0",
        "children": children,
        "attributes": [_as_str_attr("NXevent_data", name="NX_class")],
    }
    return group


def event_data_generator(data: sc.DataArray) -> Generator[dict, None, None]:
    """
    Generate JSON data for event data from a NeXus file.

    Parameters
    ----------
    data:
        A data array with event data, equivalent to what ScippNexus would load from an
        NXevent_data group in a NeXus file.

    Yields
    ------
    :
        A dict of data for a single event data pulse that can be wrapped in a
        :py:class:`ess.reduce.nexus.json_nexus.JSONGroup`.
    """
    for pulse in data:
        yield _event_data_pulse_to_json(pulse)
