"""NeXus domain types for use with Sciline."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Generic, NewType, TypeVar

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

NeXusDetectorName = NewType('NeXusDetectorName', str)
"""Name of a detector (bank) in a NeXus file."""
NeXusEntryName = NewType('NeXusEntryName', str)
"""Name of an entry in a NeXus file."""
NeXusMonitorName = NewType('NeXusMonitorName', str)
"""Name of a monitor in a NeXus file."""
NeXusSourceName = NewType('NeXusSourceName', str)
"""Name of a source in a NeXus file."""

RawDetectorData = NewType('RawDetectorData', sc.DataArray)
"""Data extracted from a RawDetector."""
RawMonitorData = NewType('RawMonitorData', sc.DataArray)
"""Data extracted from a RawMonitor."""

NeXusDetector = NewType('NeXusDetector', sc.DataGroup)
"""Full raw data from a NeXus detector."""
NeXusMonitor = NewType('NeXusMonitor', sc.DataGroup)
"""Full raw data from a NeXus monitor."""
NeXusSample = NewType('NeXusSample', sc.DataGroup)
"""Raw data from a NeXus sample."""
NeXusSource = NewType('NeXusSource', sc.DataGroup)
"""Raw data from a NeXus source."""
NeXusDetectorEventData = NewType('NeXusDetectorEventData', sc.DataArray)
"""Data array loaded from a NeXus NXevent_data group within an NXdetector."""
NeXusMonitorEventData = NewType('NeXusMonitorEventData', sc.DataArray)
"""Data array loaded from a NeXus NXevent_data group within an NXmonitor."""

SourcePosition = NewType('SourcePosition', sc.Variable | None)
"""Position of the neutron source."""

SamplePosition = NewType('SamplePosition', sc.Variable | None)
"""Position of the sample."""

DetectorPositionOffset = NewType('DetectorPositionOffset', sc.Variable | None)
"""Offset of the detector position, SUBTRACTED from base position."""

MonitorPositionOffset = NewType('MonitorPositionOffset', sc.Variable | None)


DetectorBankSizes = NewType("DetectorBankSizes", dict[str, dict[str, int | Any]])

CalibratedDetector = NewType('CalibratedDetector', sc.DataArray)
CalibratedMonitor = NewType('CalibratedMonitor', sc.DataArray)

DetectorData = NewType('DetectorData', sc.DataArray)
MonitorData = NewType('MonitorData', sc.DataArray)

PulseSelection = NewType('PulseSelection', snx.typing.ScippIndex)


class NoNewDefinitionsType: ...


NoNewDefinitions = NoNewDefinitionsType()


Component = TypeVar('Component', bound=snx.NXobject)


NeXusFileSpec = FilePath | NeXusFile | NeXusGroup


@dataclass
class NeXusLocationSpec(Generic[Component]):
    """
    NeXus filename and optional parameters to identify (parts of) a component to load.
    """

    filename: NeXusFileSpec
    entry_name: NeXusEntryName | None = None
    component_name: str | None = None
    selection: snx.typing.ScippIndex = ()
