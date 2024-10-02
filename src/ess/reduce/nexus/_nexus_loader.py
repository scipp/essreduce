# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""NeXus loaders."""

from collections.abc import Mapping
from contextlib import AbstractContextManager, nullcontext
from dataclasses import dataclass
from math import prod
from typing import cast

import scipp as sc
import scippnexus as snx

from ..logging import get_logger
from .types import FilePath, NeXusEntryName, NeXusFile, NeXusGroup, NeXusLocationSpec


class NoNewDefinitionsType: ...


NoNewDefinitions = NoNewDefinitionsType()


def load_component(
    location: NeXusLocationSpec,
    *,
    nx_class: type[snx.NXobject],
    definitions: Mapping | None | NoNewDefinitionsType = NoNewDefinitions,
) -> sc.DataGroup:
    file_path = location.filename
    selection = location.selection
    entry_name = location.entry_name
    group_name = location.component_name
    with _open_nexus_file(file_path, definitions=definitions) as f:
        entry = _unique_child_group(f, snx.NXentry, entry_name)
        if nx_class is snx.NXsample:
            instrument = entry
        else:
            instrument = _unique_child_group(entry, snx.NXinstrument, None)
        component = _unique_child_group(instrument, nx_class, group_name)
        loaded = cast(sc.DataGroup, component[selection])
        loaded['nexus_component_name'] = component.name.split('/')[-1]
    return loaded


def compute_component_position(dg: sc.DataGroup) -> sc.DataGroup:
    # In some downstream packages we use some of the Nexus components which attempt
    # to compute positions without having actual Nexus data defining depends_on chains.
    # We assume positions have been set in the non-Nexus input somehow and return early.
    if 'depends_on' not in dg:
        return dg
    transform_out_name = 'transform'
    if transform_out_name in dg:
        raise RuntimeError(
            f"Loaded data contains an item '{transform_out_name}' but we want to "
            "store the combined NeXus transformations under that name."
        )
    position_out_name = 'position'
    if position_out_name in dg:
        raise RuntimeError(
            f"Loaded data contains an item '{position_out_name}' but we want to "
            "store the computed positions under that name."
        )
    return snx.compute_positions(
        dg, store_position=position_out_name, store_transform=transform_out_name
    )


def _open_nexus_file(
    file_path: FilePath | NeXusFile | NeXusGroup,
    definitions: Mapping | None | NoNewDefinitionsType = NoNewDefinitions,
) -> AbstractContextManager:
    if isinstance(file_path, getattr(NeXusGroup, '__supertype__', type(None))):
        if (
            definitions is not NoNewDefinitions
            and definitions != file_path._definitions
        ):
            raise ValueError(
                "Cannot apply new definitions to open nexus file or nexus group."
            )
        return nullcontext(file_path)
    if definitions is NoNewDefinitions:
        return snx.File(file_path)
    return snx.File(file_path, definitions=definitions)


def _unique_child_group(
    group: snx.Group, nx_class: type[snx.NXobject], name: str | None
) -> snx.Group:
    if name is not None:
        child = group[name]
        if isinstance(child, snx.Field):
            raise ValueError(
                f"Expected a NeXus group as item '{name}' but got a field."
            )
        if child.nx_class != nx_class:
            raise ValueError(
                f"The NeXus group '{name}' was expected to be a "
                f'{nx_class} but is a {child.nx_class}.'
            )
        return child

    children = group[nx_class]
    if len(children) != 1:
        raise ValueError(
            f"Expected exactly one {nx_class.__name__} group '{group.name}', "
            f"got {len(children)}"
        )
    return next(iter(children.values()))  # type: ignore[return-value]


def _contains_nx_class(group: snx.Group, nx_class: type[snx.NXobject]) -> bool:
    # See https://github.com/scipp/scippnexus/issues/241
    try:
        return bool(group[nx_class])
    except KeyError:
        # This does not happen with the current implementation in ScippNexus.
        # The fallback is here to future-proof this function.
        return False


def extract_signal_data_array(dg: sc.DataGroup) -> sc.DataArray:
    event_data_arrays = {
        key: value
        for key, value in dg.items()
        if isinstance(value, sc.DataArray) and value.bins is not None
    }
    # Transformations and thus the position can be time dependent, in which case they
    # are stored as DataArray instead of Variable. We do not want to select these.
    histogram_data_arrays = {
        key: value
        for key, value in dg.items()
        if isinstance(value, sc.DataArray)
        and value.bins is None
        and key not in ('position', 'transform')
    }
    if (array := _select_unique_array(event_data_arrays, 'event')) is not None:
        if histogram_data_arrays:
            get_logger().info(
                "Selecting event data '%s' in favor of histogram data {%s}",
                next(iter(event_data_arrays.keys())),
                ', '.join(f"'{k}'" for k in histogram_data_arrays),
            )
        return array

    if (array := _select_unique_array(histogram_data_arrays, 'histogram')) is not None:
        return array

    raise ValueError(
        "Raw data loaded from NeXus does not contain events or a histogram. "
        "Expected to find a data array, "
        f"but the data only contains {set(dg.keys())}"
    )


def _select_unique_array(
    arrays: dict[str, sc.DataArray], mapping_name: str
) -> sc.DataArray | None:
    if not arrays:
        return None
    if len(arrays) > 1:
        raise ValueError(
            f"Raw data loaded from NeXus contains more than one {mapping_name} "
            "data array. Cannot uniquely identify the data to extract. "
            f"Got {mapping_name} items {set(arrays.keys())}"
        )
    return next(iter(arrays.values()))


def _to_snx_selection(selection, *, for_events: bool) -> snx.typing.ScippIndex:
    if selection == slice(None, None):
        return ()
    if isinstance(selection, slice):
        if for_events:
            return {'event_time_zero': selection}
        return {'time': selection}
    return selection


def load_data(
    file_path: FilePath | NeXusFile | NeXusGroup,
    selection: snx.typing.ScippIndex | slice = (),
    *,
    entry_name: NeXusEntryName | None = None,
    component_name: str,
    definitions: Mapping | NoNewDefinitionsType = NoNewDefinitions,
) -> sc.DataArray:
    """Load data of a detector or monitor from a NeXus file.

    Loads either event data from an ``NXevent_data`` group or histogram
    data from an ``NXdata`` group depending on which ``group`` contains.
    Event data is grouped by ``'event_time_zero'`` as in the NeXus file.
    Histogram data is returned as encoded in the file.

    Parameters
    ----------
    file_path:
        Indicates where to load data from.
        One of:

        - Path to a NeXus file on disk.
        - File handle or buffer for reading binary data.
        - A ScippNexus group of the root of a NeXus file.
    selection:
        Select which aprt of the data to load.
        By default, load all data.
        Supports anything that ScippNexus supports.
    component_name:
        Name of the NXdetector or NXmonitor containing the NXevent_data to load.
        Must be a group in an instrument group in the entry (see below).
    entry_name:
        Name of the entry that contains the detector.
        If ``None``, the entry will be located based
        on its NeXus class, but there cannot be more than 1.
    definitions:
        Definitions used by scippnexus loader, see :py:`scippnexus.File`
        for documentation.

    Returns
    -------
    :
        Data array with events or a histogram.
    """
    with _open_nexus_file(file_path, definitions=definitions) as f:
        entry = _unique_child_group(f, snx.NXentry, entry_name)
        instrument = _unique_child_group(entry, snx.NXinstrument, None)
        component = instrument[component_name]
        if _contains_nx_class(component, snx.NXevent_data):
            data = _unique_child_group(component, snx.NXevent_data, None)
            sel = _to_snx_selection(selection, for_events=True)
        elif _contains_nx_class(component, snx.NXdata):
            data = _unique_child_group(component, snx.NXdata, None)
            sel = _to_snx_selection(selection, for_events=False)
        else:
            raise ValueError(
                f"NeXus group '{component.name}' contains neither "
                "NXevent_data nor NXdata."
            )

        return data[sel]


def group_event_data(
    *, event_data: sc.DataArray, detector_number: sc.Variable
) -> sc.DataArray:
    """Group event data by detector number.

    The detector_number variable also defines the output shape and dimension names.

    Parameters
    ----------
    event_data:
        Data array with events to group, as returned from :py:func:`load_event_data`.
    detector_number:
        Variable with detector numbers matching the `event_id` field of the event data.

    Returns
    -------
    :
        Data array with events grouped by detector number.
    """
    event_id = detector_number.flatten(to='event_id').copy()
    constituents = event_data.bins.constituents
    begin = constituents['begin']
    end = constituents['end']
    data = constituents['data'].copy(deep=False)
    if 'event_time_zero' in event_data.coords:
        data.coords['event_time_zero'] = sc.bins_like(
            event_data, fill_value=event_data.coords['event_time_zero']
        ).bins.constituents['data']
    # After loading raw NXevent_data it is guaranteed that the event table
    # is contiguous and that there is no masking. We can therefore use the
    # more efficient approach of binning from scratch instead of erasing the
    # 'event_time_zero' binning defined by NXevent_data. This sanity check should
    # therefore always pass unless some unusual modifications were performed.
    if (
        event_data.masks
        or begin[0] != sc.index(0)
        or end[-1] != sc.index(data.sizes[data.dim])
        or (begin[1:] != end[:-1]).any()
    ):
        raise ValueError("Grouping only implemented for contiguous data with no masks.")
    out = data.group(event_id).fold(dim='event_id', sizes=detector_number.sizes)
    out.coords['detector_number'] = out.coords.pop('event_id')
    return out


def _format_time(time: sc.Variable | None) -> str:
    if time is None:
        return 'None'
    return f"{time:c}" if time.dtype != 'datetime64' else str(time.value)


@dataclass
class NeXusDetectorInfo:
    name: str
    start_time: sc.Variable | None
    end_time: sc.Variable | None
    n_pulse: int | None
    n_pixel: int | None

    def __repr__(self) -> str:
        return (
            f"{self.name}: n_pulse={self.n_pulse}, n_pixel={self.n_pixel}, "
            f"start_time={_format_time(self.start_time)}, "
            f"end_time={_format_time(self.end_time)}"
        )


@dataclass
class NeXusMonitorInfo:
    name: str
    start_time: sc.Variable | None
    end_time: sc.Variable | None
    n_pulse: int | None

    def __repr__(self) -> str:
        return (
            f"{self.name}: n_pulse={self.n_pulse}, "
            f"start_time={_format_time(self.start_time)}, "
            f"end_time={_format_time(self.end_time)}"
        )


@dataclass
class NeXusFileInfo:
    detectors: dict[str, NeXusDetectorInfo]
    monitors: dict[str, NeXusMonitorInfo]

    @property
    def start_time(self) -> sc.Variable | None:
        times = [
            comp.start_time
            for comp in (*self.detectors.values(), *self.monitors.values())
            if comp.start_time is not None
        ]
        return sc.reduce(times).min() if times else None

    @property
    def end_time(self) -> sc.Variable | None:
        times = [
            comp.end_time
            for comp in (*self.detectors.values(), *self.monitors.values())
            if comp.end_time is not None
        ]
        return sc.reduce(times).max() if times else None

    def __repr__(self) -> str:
        s = "NeXusFileInfo(\n"
        s += "  Detectors:\n"
        s += "\n".join(f"    {det}" for det in self.detectors.values())
        s += "\n  Monitors:\n"
        s += "\n".join(f"    {mon}" for mon in self.monitors.values())
        s += ')'
        return s


def _parse_name(group: snx.Group) -> str:
    return group.name.split('/')[-1]


def _parse_pixel_count(group: snx.Group) -> int | None:
    if (detector_number := group.get('detector_number')) is not None:
        return prod(detector_number.shape)


def _parse_pulse_count(group: snx.Group) -> int | None:
    try:
        events = _unique_child_group(group, snx.NXevent_data, None)
    except ValueError:
        return None
    try:
        return events['event_index'].shape[0]
    except KeyError:
        return None


def _get_start_time(group: snx.Group) -> sc.Variable | None:
    try:
        events = _unique_child_group(group, snx.NXevent_data, None)
    except ValueError:
        return None
    try:
        return events['event_time_zero'][0]
    except KeyError:
        return None


def _get_end_time(group: snx.Group) -> sc.Variable | None:
    try:
        events = _unique_child_group(group, snx.NXevent_data, None)
    except ValueError:
        return None
    try:
        return events['event_time_zero'][-1]
    except KeyError:
        return None


def _parse_detector(group: snx.Group) -> NeXusDetectorInfo:
    return NeXusDetectorInfo(
        name=_parse_name(group),
        start_time=_get_start_time(group),
        end_time=_get_end_time(group),
        n_pulse=_parse_pulse_count(group),
        n_pixel=_parse_pixel_count(group),
    )


def _parse_monitor(group: snx.Group) -> NeXusMonitorInfo:
    return NeXusMonitorInfo(
        name=_parse_name(group),
        start_time=_get_start_time(group),
        end_time=_get_end_time(group),
        n_pulse=_parse_pulse_count(group),
    )


def read_nexus_file_info(file_path: FilePath | NeXusFile | NeXusGroup) -> NeXusFileInfo:
    """Opens and inspects a NeXus file, returning a summary of its contents."""
    with _open_nexus_file(file_path) as f:
        entry = _unique_child_group(f, snx.NXentry, None)
        instrument = _unique_child_group(entry, snx.NXinstrument, None)
        detectors = {}
        monitors = {}
        for name, obj in instrument.items():
            if not isinstance(obj, snx.Group):
                continue
            if obj.nx_class == snx.NXdetector:
                detectors[name] = _parse_detector(obj)
            elif obj.nx_class == snx.NXmonitor:
                monitors[name] = _parse_monitor(obj)

        return NeXusFileInfo(detectors=detectors, monitors=monitors)
