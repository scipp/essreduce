# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

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

InstrumentName = NewType('InstrumentName', str)
"""Name of an instrument in a NeXus file."""
DetectorName = NewType('DetectorName', str)
"""Name of a detector (bank) in a NeXus file."""
MonitorName = NewType('MonitorName', str)
"""Name of a monitor in a NeXus file."""

RawDetector = NewType('RawDetector', sc.DataGroup)
"""Raw data from a NeXus detector."""
RawMonitor = NewType('RawMonitor', sc.DataGroup)
"""Raw data from a NeXus monitor."""


def load_detector(
    file_path: Union[FilePath, NeXusFile, NeXusGroup],
    *,
    detector_name: DetectorName,
    instrument_name: Optional[InstrumentName] = None,
) -> RawDetector:
    """
    TODO handling of names, including event name
    """
    return RawDetector(
        _load_data(
            file_path,
            detector_name=detector_name,
            detector_class=snx.NXdetector,
            instrument_name=instrument_name,
        )
    )


def load_monitor(
    file_path: Union[FilePath, NeXusFile, NeXusGroup],
    *,
    monitor_name: MonitorName,
    instrument_name: Optional[InstrumentName] = None,
) -> RawMonitor:
    return RawMonitor(
        _load_data(
            file_path,
            detector_name=monitor_name,
            detector_class=snx.NXmonitor,
            instrument_name=instrument_name,
        )
    )


def _load_data(
    file_path: Union[FilePath, NeXusFile, NeXusGroup],
    *,
    detector_name: Union[DetectorName, MonitorName],
    detector_class: Type[snx.NXobject],
    instrument_name: Optional[InstrumentName] = None,
) -> sc.DataGroup:
    with _open_nexus_file(file_path) as f:
        entry = f['entry']
        instrument = _unique_child_group(entry, snx.NXinstrument, instrument_name)
        detector = _unique_child_group(instrument, detector_class, detector_name)
        loaded = cast(sc.DataGroup, detector[()])
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
