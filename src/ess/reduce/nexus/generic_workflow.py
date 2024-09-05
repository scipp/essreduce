# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""
Generic workflow and components for interacting with NeXus files.

In contrast to the :py:mod:`ess.reduce.nexus.workflow` module, this module provides
generic components that can be used to build a workflow with a generic run type and
monitor type. This can be used by instrument-specific workflows as a common base
implementation of interaction with ESS NeXus files, without the need of writing
instrument-specific wrappers or implementations.
"""

import sciline
import scippnexus as snx

from . import generic_types as gt
from . import workflow
from .generic_types import MonitorType, PulseSelection, RunType
from .types import (
    DetectorBankSizes,
    GravityVector,
    NeXusDetectorName,
)


def file_path_to_file_spec(filename: gt.Filename[RunType]) -> gt.NeXusFileSpec[RunType]:
    return gt.NeXusFileSpec[RunType](filename)


def no_monitor_position_offset() -> gt.MonitorPositionOffset[RunType, MonitorType]:
    return gt.MonitorPositionOffset[RunType, MonitorType](workflow.no_offset)


def no_detector_position_offset() -> gt.DetectorPositionOffset[RunType]:
    return gt.DetectorPositionOffset[RunType](workflow.no_offset)


def all_pulses() -> PulseSelection[RunType]:
    """Select all neutron pulses in the event data."""
    return PulseSelection[RunType](slice(None, None))


def unique_sample_spec(
    filename: gt.NeXusFileSpec[RunType],
) -> gt.NeXusComponentLocationSpec[snx.NXsample, RunType]:
    return gt.NeXusComponentLocationSpec[snx.NXsample, RunType](filename=filename.value)


def unique_source_spec(
    filename: gt.NeXusFileSpec[RunType],
) -> gt.NeXusComponentLocationSpec[snx.NXsource, RunType]:
    return gt.NeXusComponentLocationSpec[snx.NXsource, RunType](filename=filename.value)


def monitor_by_name(
    filename: gt.NeXusFileSpec[RunType], name: gt.NeXusMonitorName[MonitorType]
) -> gt.NeXusMonitorLocationSpec[RunType, MonitorType]:
    return gt.NeXusMonitorLocationSpec[RunType, MonitorType](
        filename=filename.value, component_name=name
    )


def monitor_events_by_name(
    filename: gt.NeXusFileSpec[RunType],
    name: gt.NeXusMonitorName[MonitorType],
    selection: PulseSelection[RunType],
) -> gt.NeXusMonitorEventLocationSpec[RunType, MonitorType]:
    return gt.NeXusMonitorEventLocationSpec[RunType, MonitorType](
        filename=filename.value,
        component_name=name,
        selection={'event_time_zero': selection.value},
    )


def detector_by_name(
    filename: gt.NeXusFileSpec[RunType], name: NeXusDetectorName
) -> gt.NeXusComponentLocationSpec[snx.NXdetector, RunType]:
    return gt.NeXusComponentLocationSpec[snx.NXdetector, RunType](
        filename=filename.value, component_name=name
    )


def detector_events_by_name(
    filename: gt.NeXusFileSpec[RunType],
    name: NeXusDetectorName,
    selection: PulseSelection[RunType],
) -> gt.NeXusDetectorEventLocationSpec[RunType]:
    return gt.NeXusDetectorEventLocationSpec[RunType](
        filename=filename.value,
        component_name=name,
        selection={'event_time_zero': selection.value},
    )


def load_nexus_sample(
    location: gt.NeXusComponentLocationSpec[snx.NXsample, RunType],
) -> gt.NeXusSample[RunType]:
    return gt.NeXusSample[RunType](workflow.load_nexus_sample(location))


def load_nexus_source(
    location: gt.NeXusComponentLocationSpec[snx.NXsource, RunType],
) -> gt.NeXusSource[RunType]:
    return gt.NeXusSource[RunType](workflow.load_nexus_source(location))


def load_nexus_detector(
    location: gt.NeXusComponentLocationSpec[snx.NXdetector, RunType],
) -> gt.NeXusDetector[RunType]:
    return gt.NeXusDetector[RunType](workflow.load_nexus_detector(location))


def load_nexus_monitor(
    location: gt.NeXusMonitorLocationSpec[RunType, MonitorType],
) -> gt.NeXusMonitor[RunType, MonitorType]:
    return gt.NeXusMonitor[RunType, MonitorType](workflow.load_nexus_monitor(location))


def load_nexus_detector_event_data(
    location: gt.NeXusDetectorEventLocationSpec[RunType],
) -> gt.NeXusDetectorEventData[RunType]:
    return gt.NeXusDetectorEventData[RunType](
        workflow.load_nexus_detector_event_data(location)
    )


def load_nexus_monitor_event_data(
    location: gt.NeXusMonitorEventLocationSpec[RunType, MonitorType],
) -> gt.NeXusMonitorEventData[RunType, MonitorType]:
    return gt.NeXusMonitorEventData[RunType, MonitorType](
        workflow.load_nexus_monitor_event_data(location)
    )


def get_source_position(source: gt.NeXusSource[RunType]) -> gt.SourcePosition[RunType]:
    return gt.SourcePosition[RunType](workflow.get_source_position(source))


def get_sample_position(sample: gt.NeXusSample[RunType]) -> gt.SamplePosition[RunType]:
    return gt.SamplePosition[RunType](workflow.get_sample_position(sample))


def get_calibrated_detector(
    detector: gt.NeXusDetector[RunType],
    *,
    offset: gt.DetectorPositionOffset[RunType],
    source_position: gt.SourcePosition[RunType],
    sample_position: gt.SamplePosition[RunType],
    gravity: GravityVector,
    bank_sizes: DetectorBankSizes,
) -> gt.CalibratedDetector[RunType]:
    return gt.CalibratedDetector[RunType](
        workflow.get_calibrated_detector(
            detector,
            offset=offset,
            source_position=source_position,
            sample_position=sample_position,
            gravity=gravity,
            bank_sizes=bank_sizes,
        )
    )


def assemble_detector_data(
    detector: gt.CalibratedDetector[RunType],
    event_data: gt.NeXusDetectorEventData[RunType],
) -> gt.DetectorData[RunType]:
    return gt.DetectorData[RunType](
        workflow.assemble_detector_data(detector, event_data)
    )


def get_calibrated_monitor(
    monitor: gt.NeXusMonitor[RunType, MonitorType],
    offset: gt.MonitorPositionOffset[RunType, MonitorType],
    source_position: gt.SourcePosition[RunType],
) -> gt.CalibratedMonitor[RunType, MonitorType]:
    return gt.CalibratedMonitor[RunType, MonitorType](
        workflow.get_calibrated_monitor(monitor, offset, source_position)
    )


def assemble_monitor_data(
    monitor: gt.CalibratedMonitor[RunType, MonitorType],
    event_data: gt.NeXusMonitorEventData[RunType, MonitorType],
) -> gt.MonitorData[RunType, MonitorType]:
    return gt.MonitorData[RunType, MonitorType](
        workflow.assemble_monitor_data(monitor, event_data)
    )


# Use the same docstrings. The functions defined here are just wrappers with modified
# annotations adding generic in run and/or monitor around the functions in workflow.py.

load_nexus_sample.__doc__ = workflow.load_nexus_sample.__doc__
load_nexus_source.__doc__ = workflow.load_nexus_source.__doc__
load_nexus_detector.__doc__ = workflow.load_nexus_detector.__doc__
load_nexus_monitor.__doc__ = workflow.load_nexus_monitor.__doc__
load_nexus_detector_event_data.__doc__ = workflow.load_nexus_detector_event_data.__doc__
load_nexus_monitor_event_data.__doc__ = workflow.load_nexus_monitor_event_data.__doc__
get_source_position.__doc__ = workflow.get_source_position.__doc__
get_sample_position.__doc__ = workflow.get_sample_position.__doc__
get_calibrated_detector.__doc__ = workflow.get_calibrated_detector.__doc__
assemble_detector_data.__doc__ = workflow.assemble_detector_data.__doc__
get_calibrated_monitor.__doc__ = workflow.get_calibrated_monitor.__doc__
assemble_monitor_data.__doc__ = workflow.assemble_monitor_data.__doc__


_common_providers = (workflow.gravity_vector_neg_y, file_path_to_file_spec, all_pulses)

_monitor_providers = (
    no_monitor_position_offset,
    unique_source_spec,
    monitor_by_name,
    monitor_events_by_name,
    load_nexus_monitor,
    load_nexus_monitor_event_data,
    load_nexus_source,
    get_source_position,
    get_calibrated_monitor,
    assemble_monitor_data,
)

_detector_providers = (
    no_detector_position_offset,
    unique_source_spec,
    unique_sample_spec,
    detector_by_name,
    detector_events_by_name,
    load_nexus_detector,
    load_nexus_detector_event_data,
    load_nexus_source,
    load_nexus_sample,
    get_source_position,
    get_sample_position,
    get_calibrated_detector,
    assemble_detector_data,
)


def LoadMonitorWorkflow() -> sciline.Pipeline:
    """Generic workflow for loading monitor data from a NeXus file."""
    wf = sciline.Pipeline((*_common_providers, *_monitor_providers))
    return wf


def LoadDetectorWorkflow() -> sciline.Pipeline:
    """Generic workflow for loading detector data from a NeXus file."""
    wf = sciline.Pipeline((*_common_providers, *_detector_providers))
    wf[DetectorBankSizes] = DetectorBankSizes({})
    return wf


def GenericNeXusWorkflow() -> sciline.Pipeline:
    """Generic workflow for loading detector and monitor data from a NeXus file."""
    wf = sciline.Pipeline(
        (*_common_providers, *_monitor_providers, *_detector_providers)
    )
    wf[DetectorBankSizes] = DetectorBankSizes({})
    return wf
