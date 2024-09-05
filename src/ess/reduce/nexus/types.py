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
AnyNeXusMonitorName = NewType('AnyNeXusMonitorName', str)
"""Name of a monitor in a NeXus file."""
NeXusSourceName = NewType('NeXusSourceName', str)
"""Name of a source in a NeXus file."""

RawDetectorData = NewType('RawDetectorData', sc.DataArray)
"""Data extracted from a RawDetector."""
RawMonitorData = NewType('RawMonitorData', sc.DataArray)
"""Data extracted from a RawMonitor."""

AnyRunNeXusDetector = NewType('AnyRunNeXusDetector', sc.DataGroup)
"""Full raw data from a NeXus detector."""
AnyRunAnyNeXusMonitor = NewType('AnyRunAnyNeXusMonitor', sc.DataGroup)
"""Full raw data from a NeXus monitor."""
AnyRunNeXusSample = NewType('AnyRunNeXusSample', sc.DataGroup)
"""Raw data from a NeXus sample."""
AnyRunNeXusSource = NewType('AnyRunNeXusSource', sc.DataGroup)
"""Raw data from a NeXus source."""
AnyRunNeXusDetectorEventData = NewType('AnyRunNeXusDetectorEventData', sc.DataArray)
"""Data array loaded from a NeXus NXevent_data group within an NXdetector."""
AnyRunAnyNeXusMonitorEventData = NewType('AnyRunAnyNeXusMonitorEventData', sc.DataArray)
"""Data array loaded from a NeXus NXevent_data group within an NXmonitor."""

AnyRunSourcePosition = NewType('AnyRunSourcePosition', sc.Variable)
"""Position of the neutron source."""

AnyRunSamplePosition = NewType('AnyRunSamplePosition', sc.Variable)
"""Position of the sample."""

AnyRunDetectorPositionOffset = NewType('AnyRunDetectorPositionOffset', sc.Variable)
"""Offset for the detector position, added to base position."""

AnyRunAnyMonitorPositionOffset = NewType('AnyRunAnyMonitorPositionOffset', sc.Variable)
"""Offset for the monitor position, added to base position."""


DetectorBankSizes = NewType("DetectorBankSizes", dict[str, dict[str, int | Any]])

AnyRunCalibratedDetector = NewType('AnyRunCalibratedDetector', sc.DataArray)
AnyRunAnyCalibratedMonitor = NewType('AnyRunAnyCalibratedMonitor', sc.DataArray)

AnyRunDetectorData = NewType('AnyRunDetectorData', sc.DataArray)
AnyRunAnyMonitorData = NewType('AnyRunAnyMonitorData', sc.DataArray)

AnyRunPulseSelection = NewType('AnyRunPulseSelection', slice)
"""Range of neutron pulses to load from NXevent_data groups."""

GravityVector = NewType('GravityVector', sc.Variable)

Component = TypeVar('Component', bound=snx.NXobject)

AnyRunFilename = FilePath | NeXusFile | NeXusGroup


@dataclass
class NeXusLocationSpec(Generic[Component]):
    """
    NeXus filename and optional parameters to identify (parts of) a component to load.
    """

    filename: AnyRunFilename
    entry_name: NeXusEntryName | None = None
    component_name: str | None = None
    selection: snx.typing.ScippIndex = ()


@dataclass
class NeXusEventDataLocationSpec(NeXusLocationSpec[Component]):
    """NeXus filename and parameters to identify (parts of) events to load."""
