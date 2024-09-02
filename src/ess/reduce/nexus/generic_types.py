"""Domain types for use with Sciline, parametrized by run- and monitor-type."""

from dataclasses import dataclass
from pathlib import Path
from typing import Generic, NewType, TypeVar

import sciline
import scipp as sc
import scippnexus as snx

from .types import Component, FilePath, NeXusFile, NeXusGroup, NeXusLocationSpec

# 1  TypeVars used to parametrize the generic parts of the workflow

# 1.1  Run types
BackgroundRun = NewType('BackgroundRun', int)
"""Background run such as a run with only a solvent which the sample is placed in."""
EmptyBeamRun = NewType('EmptyBeamRun', int)
"""
Run with empty sample holder, sometimes called 'direct run'.

It is used for reading the data from the transmission monitor.
"""
SampleRun = NewType('SampleRun', int)
"""Sample run."""
VanadiumRun = NewType('VanadiumRun', int)
"""Vanadium run."""

ScatteringRunType = TypeVar(
    'ScatteringRunType',
    BackgroundRun,
    SampleRun,
    VanadiumRun,
)


class TransmissionRun(Generic[ScatteringRunType]):
    """
    Mapping between ScatteringRunType and transmission run.

    In the case where no transmission run is provided, the transmission run should be
    the same as the measurement (sample or background) run.
    """


RunType = TypeVar(
    'RunType',
    BackgroundRun,
    EmptyBeamRun,
    SampleRun,
    # Note that mypy does not seem to like this nesting, may need to find a workaround
    TransmissionRun[SampleRun],
    TransmissionRun[BackgroundRun],
    VanadiumRun,
)
"""TypeVar used for specifying BackgroundRun, EmptyBeamRun or SampleRun"""

# 1.2  Monitor types
Monitor1 = NewType('Monitor1', int)
"""Identifier for an arbitrary monitor"""
Monitor2 = NewType('Monitor2', int)
"""Identifier for an arbitrary monitor"""
Monitor3 = NewType('Monitor3', int)
"""Identifier for an arbitrary monitor"""
Monitor4 = NewType('Monitor4', int)
"""Identifier for an arbitrary monitor"""
Monitor5 = NewType('Monitor5', int)
"""Identifier for an arbitrary monitor"""
Incident = NewType('Incident', int)
"""Incident monitor"""
Transmission = NewType('Transmission', int)
"""Transmission monitor"""
MonitorType = TypeVar(
    'MonitorType',
    Monitor1,
    Monitor2,
    Monitor3,
    Monitor4,
    Monitor5,
    Incident,
    Transmission,
)
"""TypeVar used for specifying the monitor type such as Incident or Transmission"""


class NeXusMonitorName(sciline.Scope[MonitorType, str], str):
    """Name of a monitor in a NeXus file."""


class NeXusDetector(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Full raw data from a NeXus detector."""


class NeXusMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataGroup], sc.DataGroup
):
    """Full raw data from a NeXus monitor."""


class NeXusSample(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Raw data from a NeXus sample."""


class NeXusSource(sciline.Scope[RunType, sc.DataGroup], sc.DataGroup):
    """Raw data from a NeXus source."""


class NeXusDetectorEventData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Data array loaded from a NeXus NXevent_data group within an NXdetector."""


class NeXusMonitorEventData(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Data array loaded from a NeXus NXevent_data group within an NXmonitor."""


class SourcePosition(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Position of the neutron source."""


class SamplePosition(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Position of the sample."""


class DetectorPositionOffset(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Offset for the detector position, added to base position."""


class MonitorPositionOffset(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.Variable], sc.Variable
):
    """Offset for the monitor position, added to base position."""


class CalibratedDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Calibrated data from a detector."""


class CalibratedMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Calibrated data from a monitor."""


class DetectorData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Calibrated detector merged with neutron event data."""


class MonitorData(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Calibrated monitor merged with neutron event data."""


class Filename(sciline.Scope[RunType, Path], Path): ...


@dataclass
class PulseSelection(Generic[RunType]):
    """Range of neutron pulses to load from NXevent_data groups."""

    value: slice


@dataclass
class NeXusFileSpec(Generic[RunType]):
    value: FilePath | NeXusFile | NeXusGroup


@dataclass
class NeXusComponentLocationSpec(
    NeXusLocationSpec[Component], Generic[Component, RunType]
):
    """
    NeXus filename and optional parameters to identify (parts of) a detector to load.
    """


@dataclass
class NeXusMonitorLocationSpec(
    NeXusLocationSpec[snx.NXmonitor], Generic[RunType, MonitorType]
):
    """
    NeXus filename and optional parameters to identify (parts of) a monitor to load.
    """


@dataclass
class NeXusDetectorEventLocationSpec(
    NeXusLocationSpec[snx.NXevent_data], Generic[RunType]
):
    """NeXus filename and parameters to identify (parts of) detector events to load."""


@dataclass
class NeXusMonitorEventLocationSpec(
    NeXusLocationSpec[snx.NXevent_data], Generic[RunType, MonitorType]
):
    """NeXus filename and parameters to identify (parts of) monitor events to load."""
