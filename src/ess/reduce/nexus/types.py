"""Domain types for use with Sciline, parametrized by run- and monitor-type."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Generic, NewType, TypeVar

import sciline
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

NeXusEntryName = NewType('NeXusEntryName', str)
"""Name of an entry in a NeXus file."""
NeXusSourceName = NewType('NeXusSourceName', str)
"""Name of a source in a NeXus file."""

DetectorBankSizes = NewType("DetectorBankSizes", dict[str, dict[str, int | Any]])

GravityVector = NewType('GravityVector', sc.Variable)

PreopenNeXusFile = NewType('PreopenNeXusFile', bool)
"""Whether to preopen NeXus files before passing them to the rest of the workflow."""


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

Component = TypeVar(
    'Component',
    snx.NXdetector,
    snx.NXsample,
    snx.NXsource,
    Monitor1,
    Monitor2,
    Monitor3,
    Monitor4,
    Monitor5,
    Incident,
    Transmission,
)
UniqueComponentType = TypeVar('UniqueComponentType', snx.NXsample, snx.NXsource)
"""Components that can be identified by their type as there will only be one."""


class NeXusComponentName(sciline.Scope[Component, str], str):
    """Name of a monitor or detector component in a NeXus file."""


class NeXusClassName(sciline.Scope[Component, str], str):
    """NX_class of a component in a NeXus file."""


NeXusDetectorName = NeXusComponentName[snx.NXdetector]
"""Name of a detector (bank) in a NeXus file."""


class NeXusComponent(
    sciline.ScopeTwoParams[Component, RunType, sc.DataGroup], sc.DataGroup
):
    """Raw data from a NeXus component."""


class NeXusData(sciline.ScopeTwoParams[Component, RunType, sc.DataArray], sc.DataArray):
    """
    Data array loaded from an NXevent_data or NXdata group.

    This must be contained in an NXmonitor or NXdetector group.
    """


class ComponentPosition(
    sciline.ScopeTwoParams[Component, RunType, sc.Variable], sc.Variable
):
    """Position of a component such as source, sample, monitor, or detector."""


class DetectorPositionOffset(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Offset for the detector position, added to base position."""


class MonitorPositionOffset(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.Variable], sc.Variable
):
    """Offset for the monitor position, added to base position."""


class CalibratedDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Calibrated data from a detector."""


class CalibratedBeamline(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Calibrated beamline with detector and other components."""


class CalibratedMonitor(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Calibrated data from a monitor."""


class DetectorData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Calibrated detector merged with neutron event or histogram data."""


class MonitorData(
    sciline.ScopeTwoParams[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Calibrated monitor merged with neutron event or histogram data."""


class Filename(sciline.Scope[RunType, Path], Path): ...


@dataclass
class PulseSelection(Generic[RunType]):
    """Range of neutron pulses to load from NXevent_data or NXdata groups."""

    value: snx.typing.ScippIndex | slice


@dataclass
class NeXusFileSpec(Generic[RunType]):
    value: FilePath | NeXusFile | NeXusGroup


@dataclass
class NeXusLocationSpec:
    """
    NeXus filename and optional parameters to identify (parts of) a component to load.
    """

    filename: FilePath | NeXusFile | NeXusGroup
    entry_name: NeXusEntryName | None = None
    component_name: str | None = None
    selection: snx.typing.ScippIndex | slice = ()


@dataclass
class NeXusComponentLocationSpec(NeXusLocationSpec, Generic[Component, RunType]):
    """
    NeXus filename and optional parameters to identify (parts of) a component to load.
    """


@dataclass
class NeXusDataLocationSpec(NeXusLocationSpec, Generic[Component, RunType]):
    """NeXus filename and parameters to identify (parts of) detector data to load."""


T = TypeVar('T', bound='NeXusTransformationChain')


@dataclass
class NeXusTransformationChain(snx.TransformationChain, Generic[Component, RunType]):
    @classmethod
    def from_base(cls: type[T], base: snx.TransformationChain) -> T:
        return cls(
            parent=base.parent,
            value=base.value,
            transformations=base.transformations,
        )

    def compute_position(self) -> sc.Variable | sc.DataArray:
        return self.compute() * sc.vector([0, 0, 0], unit='m')


@dataclass
class NeXusTransformation(Generic[Component, RunType]):
    value: sc.Variable

    @staticmethod
    def from_chain(
        chain: NeXusTransformationChain[Component, RunType],
        # TODO can add filter options here
    ) -> 'NeXusTransformation[Component, RunType]':
        transform = chain.compute()
        if transform.ndim == 0:
            return NeXusTransformation(value=transform)
        raise ValueError(f"Expected scalar transformation, got {transform}")
