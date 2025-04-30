"""Domain types for use with Sciline, parametrized by run- and monitor-type."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO, Generic, NewType, TypeVar

import sciline
import scipp as sc
import scippnexus as snx
from scippneutron import metadata as scn_meta

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
"""TypeVar for specifying what run some data belongs to.

Possible values:

- :class:`BackgroundRun`
- :class:`EmptyBeamRun`
- :class:`SampleRun`
- :class:`TransmissionRun`
- :class:`VanadiumRun`
"""


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
Monitor6 = NewType('Monitor6', int)
"""Identifier for an arbitrary monitor"""
IncidentMonitor = NewType('IncidentMonitor', int)
"""Incident monitor"""
TransmissionMonitor = NewType('TransmissionMonitor', int)
"""Transmission monitor"""
FrameMonitor0 = NewType('FrameMonitor', int)
"""Frame monitor number 0"""
FrameMonitor1 = NewType('FrameMonitor', int)
"""Frame monitor number 1"""
FrameMonitor2 = NewType('FrameMonitor', int)
"""Frame monitor number 2"""
FrameMonitor3 = NewType('FrameMonitor', int)
"""Frame monitor number 3"""
CaveMonitor = NewType('CaveMonitor', int)
"""A monitor located in the instrument cave"""
MonitorType = TypeVar(
    'MonitorType',
    Monitor1,
    Monitor2,
    Monitor3,
    Monitor4,
    Monitor5,
    Monitor6,
    IncidentMonitor,
    TransmissionMonitor,
    FrameMonitor0,
    FrameMonitor1,
    FrameMonitor2,
    FrameMonitor3,
    CaveMonitor,
)
"""TypeVar for specifying what monitor some data belongs to.

Possible values:

- :class:`Monitor1`
- :class:`Monitor2`
- :class:`Monitor3`
- :class:`Monitor4`
- :class:`Monitor5`
- :class:`Monitor6`
- :class:`IncidentMonitor`
- :class:`TransmissionMonitor`
- :class:`FrameMonitor0`
- :class:`FrameMonitor1`
- :class:`FrameMonitor2`
- :class:`FrameMonitor3`
- :class:`CaveMonitor`
"""


Component = TypeVar(
    'Component',
    snx.NXdetector,
    snx.NXsample,
    snx.NXsource,
    snx.NXdisk_chopper,
    snx.NXcrystal,
    Monitor1,
    Monitor2,
    Monitor3,
    Monitor4,
    Monitor5,
    Monitor6,
    IncidentMonitor,
    TransmissionMonitor,
    FrameMonitor0,
    FrameMonitor1,
    FrameMonitor2,
    FrameMonitor3,
    CaveMonitor,
)
UniqueComponent = TypeVar('UniqueComponent', snx.NXsample, snx.NXsource)
"""Components that can be identified by their type as there will only be one."""

Beamline = scn_meta.Beamline
"""Beamline metadata."""
Measurement = scn_meta.Measurement
"""measurement metadata."""
Source = scn_meta.Source
"""Neutron source metadata."""


class NeXusName(sciline.Scope[Component, str], str):
    """Name of a component in a NeXus file."""


class NeXusClass(sciline.Scope[Component, type], type):
    """NX_class of a component in a NeXus file."""


NeXusDetectorName = NeXusName[snx.NXdetector]
"""Name of a detector (bank) in a NeXus file."""


class NeXusComponent(sciline.Scope[Component, RunType, sc.DataGroup], sc.DataGroup):
    """Raw data from a NeXus component."""


class AllNeXusComponents(sciline.Scope[Component, RunType, sc.DataGroup], sc.DataGroup):
    """Raw data from all NeXus components of one class."""


class NeXusData(sciline.Scope[Component, RunType, sc.DataArray], sc.DataArray):
    """
    Data array loaded from an NXevent_data or NXdata group.

    This must be contained in an NXmonitor or NXdetector group.
    """


class Position(sciline.Scope[Component, RunType, sc.Variable], sc.Variable):
    """Position of a component such as source, sample, monitor, or detector."""


class DetectorPositionOffset(sciline.Scope[RunType, sc.Variable], sc.Variable):
    """Offset for the detector position, added to base position."""


class MonitorPositionOffset(
    sciline.Scope[RunType, MonitorType, sc.Variable], sc.Variable
):
    """Offset for the monitor position, added to base position."""


class CalibratedDetector(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Calibrated data from a detector."""


class CalibratedBeamline(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Calibrated beamline with detector and other components."""


class CalibratedMonitor(
    sciline.Scope[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """Calibrated data from a monitor."""


class DetectorData(sciline.Scope[RunType, sc.DataArray], sc.DataArray):
    """Calibrated detector merged with neutron event or histogram data."""


class MonitorData(sciline.Scope[RunType, MonitorType, sc.DataArray], sc.DataArray):
    """Calibrated monitor merged with neutron event or histogram data."""


class Filename(sciline.Scope[RunType, Path], Path): ...


@dataclass
class TimeInterval(Generic[RunType]):
    """Range of neutron pulses to load from NXevent_data or NXdata groups."""

    value: slice


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
class NeXusAllLocationSpec:
    """
    NeXus parameters to identify all components of a class to load.
    """

    filename: FilePath | NeXusFile | NeXusGroup
    entry_name: NeXusEntryName | None = None
    selection: snx.typing.ScippIndex | slice = ()


@dataclass
class NeXusAllComponentLocationSpec(NeXusAllLocationSpec, Generic[Component, RunType]):
    """
    NeXus parameters to identify all components of a class to load.
    """


@dataclass
class NeXusDataLocationSpec(NeXusLocationSpec, Generic[Component, RunType]):
    """NeXus filename and parameters to identify (parts of) detector data to load."""


class NeXusTransformationChain(
    sciline.Scope[Component, RunType, snx.TransformationChain],
    snx.TransformationChain,
): ...


@dataclass
class NeXusTransformation(Generic[Component, RunType]):
    value: sc.Variable

    @staticmethod
    def from_chain(
        chain: NeXusTransformationChain[Component, RunType],
    ) -> 'NeXusTransformation[Component, RunType]':
        """
        Convert a transformation chain to a single transformation.

        As transformation chains may be time-dependent, this method will need to select
        a specific time point to convert to a single transformation. This may include
        averaging as well as threshold checks. This is not implemented yet and we
        therefore currently raise an error if the transformation chain does not compute
        to a scalar.
        """
        if chain.transformations.sizes != {}:
            raise ValueError(f"Expected scalar transformation, got {chain}")
        transform = chain.compute()
        return NeXusTransformation(value=transform)


class Choppers(
    sciline.Scope[RunType, sc.DataGroup[sc.DataGroup[Any]]],
    sc.DataGroup[sc.DataGroup[Any]],
):
    """All choppers in a NeXus file."""


class Analyzers(
    sciline.Scope[RunType, sc.DataGroup[sc.DataGroup[Any]]],
    sc.DataGroup[sc.DataGroup[Any]],
):
    """All analyzers in a NeXus file."""
