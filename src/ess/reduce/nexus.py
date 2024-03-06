# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""NeXus utilities.

This module defines functions and domain types that can be used
to build Sciline pipelines for simple workflows.
If multiple different kind sof files (e.g., sample and background runs)
are needed, custom types and providers need to be defined to wrap
the basic ones here.
"""

from contextlib import nullcontext
from pathlib import Path
from typing import BinaryIO, ContextManager, NewType, Optional, Type, Union, cast

import scipp as sc
import scippnexus as snx

FilePath = NewType('FilePath', Path)
"""Full path to a NeXus file on disk."""
NeXusFile = NewType('NeXusFile', BinaryIO)
"""An open NeXus file.

Can be any file handle for reading binary data.

Note that this cannot be used as a parameter in Sciline as there are no
concrete implementations of ``BinaryIO``.
The type alias is provided for callers of load functions outside of pipelines.
"""
NeXusGroup = NewType('NeXusGroup', snx.Group)
"""A ScippNexus group in an open file."""

DetectorName = NewType('DetectorName', str)
"""Name of a detector (bank) in a NeXus file."""
EntryName = NewType('EntryName', str)
"""Name of an entry in a NeXus file."""
MonitorName = NewType('MonitorName', str)
"""Name of a monitor in a NeXus file."""
SourceName = NewType('SourceName', str)
"""Name of a source in a NeXus file."""

RawDetector = NewType('RawDetector', sc.DataGroup)
"""Full raw data from a NeXus detector."""
RawDetectorData = NewType('RawDetectorData', sc.DataArray)
"""Data extracted from a RawDetector."""
RawMonitor = NewType('RawMonitor', sc.DataGroup)
"""Full raw data from a NeXus monitor."""
RawMonitorData = NewType('RawMonitorData', sc.DataArray)
"""Data extracted from a RawMonitor."""
RawSample = NewType('RawSample', sc.DataGroup)
"""Raw data from a NeXus sample."""
RawSource = NewType('RawSource', sc.DataGroup)
"""Raw data from a NeXus source."""


def load_detector(
    file_path: Union[FilePath, NeXusFile, NeXusGroup],
    *,
    detector_name: DetectorName,
    entry_name: Optional[EntryName] = None,
) -> RawDetector:
    """Load a single detector (bank) from a NeXus file.

    The detector positions are computed automatically for NeXus transformations.

    Parameters
    ----------
    file_path:
        Indicates where to load data from.
        One of:

        - Path to a NeXus file on disk.
        - File handle or buffer for reading binary data.
        - A ScippNexus group of the root of a NeXus file.
    detector_name:
        Name of the detector (bank) to load.
        Must be a group in an instrument group in the entry (see below).
    entry_name:
        Name of the entry that contains the detector.
        If ``None``, the entry will be located based
        on its NeXus class, but there cannot be more than 1.

    Returns
    -------
    :
        A data group containing the detector events or histogram
        and any auxiliary data stored in the same NeXus group.
    """
    return RawDetector(
        _load_group_with_positions(
            file_path,
            group_name=detector_name,
            nx_class=snx.NXdetector,
            entry_name=entry_name,
        )
    )


def load_monitor(
    file_path: Union[FilePath, NeXusFile, NeXusGroup],
    *,
    monitor_name: MonitorName,
    entry_name: Optional[EntryName] = None,
) -> RawMonitor:
    """Load a single monitor from a NeXus file.

    The monitor position is computed automatically for NeXus transformations.

    Parameters
    ----------
    file_path:
        Indicates where to load data from.
        One of:

        - Path to a NeXus file on disk.
        - File handle or buffer for reading binary data.
        - A ScippNexus group of the root of a NeXus file.
    monitor_name:
        Name of the monitor to load.
        Must be a group in an instrument group in the entry (see below).
    entry_name:
        Name of the entry that contains the monitor.
        If ``None``, the entry will be located based
        on its NeXus class, but there cannot be more than 1.

    Returns
    -------
    :
        A data group containing the monitor events or histogram
        and any auxiliary data stored in the same NeXus group.
    """
    return RawMonitor(
        _load_group_with_positions(
            file_path,
            group_name=monitor_name,
            nx_class=snx.NXmonitor,
            entry_name=entry_name,
        )
    )


def load_source(
    file_path: Union[FilePath, NeXusFile, NeXusGroup],
    *,
    source_name: Optional[SourceName] = None,
    entry_name: Optional[EntryName] = None,
) -> RawSource:
    """Load a source from a NeXus file.

    The source position is computed automatically for NeXus transformations.

    Parameters
    ----------
    file_path:
        Indicates where to load data from.
        One of:

        - Path to a NeXus file on disk.
        - File handle or buffer for reading binary data.
        - A ScippNexus group of the root of a NeXus file.
    source_name:
        Name of the source to load.
        Must be a group in an instrument group in the entry (see below).
        If ``None``, the source will be located based
        on its NeXus class.
    entry_name:
        Name of the instrument that contains the source.
        If ``None``, the entry will be located based
        on its NeXus class, but there cannot be more than 1.

    Returns
    -------
    :
        A data group containing all data stored in
        the source NeXus group.
    """
    return RawSource(
        _load_group_with_positions(
            file_path,
            group_name=source_name,
            nx_class=snx.NXsource,
            entry_name=entry_name,
        )
    )


def load_sample(
    file_path: Union[FilePath, NeXusFile, NeXusGroup],
    entry_name: Optional[EntryName] = None,
) -> RawSample:
    """Load a sample from a NeXus file.

    The sample is located based on its NeXus class.
    There can be only one sample in a NeXus file or
    in the entry indicated by ``entry_name``.

    Parameters
    ----------
    file_path:
        Indicates where to load data from.
        One of:

        - Path to a NeXus file on disk.
        - File handle or buffer for reading binary data.
        - A ScippNexus group of the root of a NeXus file.
    entry_name:
        Name of the instrument that contains the source.
        If ``None``, the entry will be located based
        on its NeXus class, but there cannot be more than 1.

    Returns
    -------
    :
        A data group containing all data stored in
        the sample NeXus group.
    """
    with _open_nexus_file(file_path) as f:
        entry = _unique_child_group(f, snx.NXentry, entry_name)
        loaded = cast(sc.DataGroup, _unique_child_group(entry, snx.NXsample, None)[()])
    return RawSample(loaded)


def _load_group_with_positions(
    file_path: Union[FilePath, NeXusFile, NeXusGroup],
    *,
    group_name: Optional[str],
    nx_class: Type[snx.NXobject],
    entry_name: Optional[EntryName] = None,
) -> sc.DataGroup:
    with _open_nexus_file(file_path) as f:
        entry = _unique_child_group(f, snx.NXentry, entry_name)
        instrument = _unique_child_group(entry, snx.NXinstrument, None)
        loaded = cast(
            sc.DataGroup, _unique_child_group(instrument, nx_class, group_name)[()]
        )
        loaded = snx.compute_positions(loaded)
        return loaded


def _open_nexus_file(
    file_path: Union[FilePath, NeXusFile, NeXusGroup]
) -> ContextManager:
    if isinstance(file_path, getattr(NeXusGroup, '__supertype__', type(None))):
        return nullcontext(file_path)
    return snx.File(file_path)


def _unique_child_group(
    group: snx.Group, nx_class: Type[snx.NXobject], name: Optional[str]
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
        raise ValueError(f'Expected exactly one {nx_class} group, got {len(children)}')
    return next(iter(children.values()))  # type: ignore[return-value]


def extract_detector_data(detector: RawDetector) -> RawDetectorData:
    """Get and return the events or histogram from a detector loaded from NeXus.

    This function looks for a data array in the detector group and returns that.

    Parameters
    ----------
    detector:
        A detector loaded from NeXus.

    Returns
    -------
    :
        A data array containing the events or histogram.

    Raises
    ------
    ValueError
        If there is more than one data array.

    See also
    --------
    load_detector:
        Load a detector from a NeXus file in a format compatible with
        ``extract_detector_data``.
    """
    return RawDetectorData(_extract_events_or_histogram(detector))


def extract_monitor_data(monitor: RawMonitor) -> RawMonitorData:
    """Get and return the events or histogram from a monitor loaded from NeXus.

    This function looks for a data array in the monitor group and returns that.

    Parameters
    ----------
    monitor:
        A monitor loaded from NeXus.

    Returns
    -------
    :
        A data array containing the events or histogram.

    Raises
    ------
    ValueError
        If there is more than one data array.

    See also
    --------
    load_monitor:
        Load a monitor from a NeXus file in a format compatible with
        ``extract_monitor_data``.
    """
    return RawMonitorData(_extract_events_or_histogram(monitor))


def _extract_events_or_histogram(dg: sc.DataGroup) -> sc.DataArray:
    data_arrays = {
        key: value for key, value in dg.items() if isinstance(value, sc.DataArray)
    }
    if len(data_arrays) == 0:
        raise ValueError(
            "Raw data loaded from NeXus does not contain events or a histogram. "
            "Expected to find a data array, "
            f"but the data only contains {set(dg.keys())}"
        )
    if len(data_arrays) > 1:
        raise ValueError(
            "Raw data loaded from NeXus contains more than one data array. "
            "Cannot uniquely identify the event or histogram data. "
            f"Got items {set(dg.keys())}"
        )
    return next(iter(data_arrays.values()))
