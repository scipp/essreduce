# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""Workflow and workflow components for interacting with NeXus files."""

from dataclasses import replace
from typing import Any

import sciline
import scipp as sc
import scippnexus as snx
from scipp.constants import g

from . import _nexus_loader as nexus
from .types import (
    AnyNeXusMonitorName,
    AnyRunAnyCalibratedMonitor,
    AnyRunAnyMonitorData,
    AnyRunAnyMonitorPositionOffset,
    AnyRunAnyNeXusMonitor,
    AnyRunAnyNeXusMonitorEventData,
    AnyRunCalibratedDetector,
    AnyRunDetectorData,
    AnyRunDetectorPositionOffset,
    AnyRunFilename,
    AnyRunNeXusDetector,
    AnyRunNeXusDetectorEventData,
    AnyRunNeXusSample,
    AnyRunNeXusSource,
    AnyRunPulseSelection,
    AnyRunSamplePosition,
    AnyRunSourcePosition,
    DetectorBankSizes,
    GravityVector,
    NeXusDetectorName,
    NeXusEventDataLocationSpec,
    NeXusLocationSpec,
)

origin = sc.vector([0, 0, 0], unit="m")
"""The origin, used as default sample position."""
no_offset = sc.vector([0, 0, 0], unit="m")
"""Offset that does not change the position."""


def gravity_vector_neg_y() -> GravityVector:
    """
    Gravity vector for default instrument coordinate system where y is up.
    """
    return GravityVector(sc.vector(value=[0, -1, 0]) * g)


def unique_sample_spec(filename: AnyRunFilename) -> NeXusLocationSpec[snx.NXsample]:
    """
    Create a location spec for a unique sample group in a NeXus file.

    Parameters
    ----------
    filename:
        NeXus file to use for the location spec.
    """
    return NeXusLocationSpec[snx.NXsample](filename=filename)


def unique_source_spec(filename: AnyRunFilename) -> NeXusLocationSpec[snx.NXsource]:
    """
    Create a location spec for a unique source group in a NeXus file.

    Parameters
    ----------
    filename:
        NeXus file to use for the location spec.
    """
    return NeXusLocationSpec[snx.NXsource](filename=filename)


def monitor_by_name(
    filename: AnyRunFilename, name: AnyNeXusMonitorName
) -> NeXusLocationSpec[snx.NXmonitor]:
    """
    Create a location spec for a monitor group in a NeXus file.

    Parameters
    ----------
    filename:
        NeXus file to use for the location spec.
    name:
        Name of the monitor group.
    """
    return NeXusLocationSpec[snx.NXmonitor](filename=filename, component_name=name)


def monitor_events_by_name(
    filename: AnyRunFilename, name: AnyNeXusMonitorName, selection: AnyRunPulseSelection
) -> NeXusEventDataLocationSpec[snx.NXmonitor]:
    """
    Create a location spec for monitor event data in a NeXus file.

    Parameters
    ----------
    filename:
        NeXus file to use for the location spec.
    name:
        Name of the monitor group.
    selection:
        Selection (start and stop as a Python slice object) for the monitor event data.
    """
    return NeXusEventDataLocationSpec[snx.NXmonitor](
        filename=filename, component_name=name, selection={'event_time_zero': selection}
    )


def detector_by_name(
    filename: AnyRunFilename, name: NeXusDetectorName
) -> NeXusLocationSpec[snx.NXdetector]:
    """
    Create a location spec for a detector group in a NeXus file.

    Parameters
    ----------
    filename:
        NeXus file to use for the location spec.
    name:
        Name of the detector group.
    """
    return NeXusLocationSpec[snx.NXdetector](filename=filename, component_name=name)


def detector_events_by_name(
    filename: AnyRunFilename, name: NeXusDetectorName, selection: AnyRunPulseSelection
) -> NeXusEventDataLocationSpec[snx.NXdetector]:
    """
    Create a location spec for detector event data in a NeXus file.

    Parameters
    ----------
    filename:
        NeXus file to use for the location spec.
    name:
        Name of the detector group.
    selection:
        Selection (start and stop as a Python slice object) for the detector event data.
    """
    return NeXusEventDataLocationSpec[snx.NXdetector](
        filename=filename, component_name=name, selection={'event_time_zero': selection}
    )


def load_nexus_sample(location: NeXusLocationSpec[snx.NXsample]) -> AnyRunNeXusSample:
    """
    Load a NeXus sample group from a file.

    If there is no sample group in the file, an empty group is returned. This should
    not happen, but handling it gracefully makes testing and working with
    pre-production files easier. There should be little harm in returning an empty
    group. Subsequent extract of the sample position will then default to the origin.

    Parameters
    ----------
    location:
        Location spec for the sample group.
    """
    try:
        dg = nexus.load_component(location, nx_class=snx.NXsample)
    except ValueError:
        dg = sc.DataGroup()
    return AnyRunNeXusSample(dg)


def load_nexus_source(location: NeXusLocationSpec[snx.NXsource]) -> AnyRunNeXusSource:
    """
    Load a NeXus source group from a file.

    Parameters
    ----------
    location:
        Location spec for the source group.
    """
    return AnyRunNeXusSource(nexus.load_component(location, nx_class=snx.NXsource))


def load_nexus_detector(
    location: NeXusLocationSpec[snx.NXdetector],
) -> AnyRunNeXusDetector:
    """
    Load detector from NeXus, but with event data replaced by placeholders.

    As the event data can be large and is not needed at this stage, it is replaced by
    a placeholder. A placeholder is used to allow for returning a scipp.DataArray, which
    is what most downstream code will expect.
    Currently the placeholder is the detector number, but this may change in the future.

    The returned object is a scipp.DataGroup, as it may contain additional information
    about the detector that cannot be represented as a single scipp.DataArray. Most
    downstream code will only be interested in the contained scipp.DataArray so this
    needs to be extracted. However, other processing steps may require the additional
    information, so it is kept in the DataGroup.

    Loading thus proceeds in three steps:

    1. This function loads the detector, but replaces the event data with placeholders.
    2. :py:func:`get_calibrated_detector` drops the additional information, returning
       only the contained scipp.DataArray, reshaped to the logical detector shape.
       This will generally contain coordinates as well as pixel masks.
    3. :py:func:`assemble_detector_data` replaces placeholder data values with the
       event data, and adds source and sample positions.

    Parameters
    ----------
    location:
        Location spec for the detector group.
    """
    definitions = snx.base_definitions()
    definitions["NXdetector"] = _StrippedDetector
    # The selection is only used for selecting a range of event data.
    location = replace(location, selection=())

    return AnyRunNeXusDetector(
        nexus.load_component(location, nx_class=snx.NXdetector, definitions=definitions)
    )


def load_nexus_monitor(
    location: NeXusLocationSpec[snx.NXmonitor],
) -> AnyRunAnyNeXusMonitor:
    """
    Load monitor from NeXus, but with event data replaced by placeholders.

    As the event data can be large and is not needed at this stage, it is replaced by
    a placeholder. A placeholder is used to allow for returning a scipp.DataArray, which
    is what most downstream code will expect.
    Currently the placeholder is a size-0 array, but this may change in the future.

    The returned object is a scipp.DataGroup, as it may contain additional information
    about the monitor that cannot be represented as a single scipp.DataArray. Most
    downstream code will only be interested in the contained scipp.DataArray so this
    needs to be extracted. However, other processing steps may require the additional
    information, so it is kept in the DataGroup.

    Loading thus proceeds in three steps:

    1. This function loads the monitor, but replaces the event data with placeholders.
    2. :py:func:`get_calbirated_monitor` drops the additional information, returning
       only the contained scipp.DataArray.
       This will generally contain coordinates as well as pixel masks.
    3. :py:func:`assemble_monitor_data` replaces placeholder data values with the
       event data, and adds source and sample positions.

    Parameters
    ----------
    location:
        Location spec for the monitor group.
    """
    definitions = snx.base_definitions()
    definitions["NXmonitor"] = _StrippedMonitor
    return AnyRunAnyNeXusMonitor(
        nexus.load_component(location, nx_class=snx.NXmonitor, definitions=definitions)
    )


def load_nexus_detector_event_data(
    location: NeXusEventDataLocationSpec[snx.NXdetector],
) -> AnyRunNeXusDetectorEventData:
    """
    Load event data from a NeXus detector group.

    Parameters
    ----------
    location:
        Location spec for the detector group.
    """
    return AnyRunNeXusDetectorEventData(
        nexus.load_event_data(
            file_path=location.filename,
            entry_name=location.entry_name,
            selection=location.selection,
            component_name=location.component_name,
        )
    )


def load_nexus_monitor_event_data(
    location: NeXusEventDataLocationSpec[snx.NXmonitor],
) -> AnyRunAnyNeXusMonitorEventData:
    """
    Load event data from a NeXus monitor group.

    Parameters
    ----------
    location:
        Location spec for the monitor group.
    """
    return AnyRunAnyNeXusMonitorEventData(
        nexus.load_event_data(
            file_path=location.filename,
            entry_name=location.entry_name,
            selection=location.selection,
            component_name=location.component_name,
        )
    )


def get_source_position(source: AnyRunNeXusSource) -> AnyRunSourcePosition:
    """
    Extract the source position from a NeXus source group.

    Parameters
    ----------
    source:
        NeXus source group.
    """
    return AnyRunSourcePosition(source["position"])


def get_sample_position(sample: AnyRunNeXusSample) -> AnyRunSamplePosition:
    """
    Extract the sample position from a NeXus sample group.

    Defaults to the origin if the sample group does not contain a position field.

    Parameters
    ----------
    sample:
        NeXus sample group.
    """
    return AnyRunSamplePosition(sample.get("position", origin))


def get_calibrated_detector(
    detector: AnyRunNeXusDetector,
    *,
    offset: AnyRunDetectorPositionOffset,
    source_position: AnyRunSourcePosition,
    sample_position: AnyRunSamplePosition,
    gravity: GravityVector,
    bank_sizes: DetectorBankSizes,
) -> AnyRunCalibratedDetector:
    """
    Extract the data array corresponding to a detector's signal field.

    The returned data array includes coords and masks pertaining directly to the
    signal values array, but not additional information about the detector. The
    data array is reshaped to the logical detector shape, which by folding the data
    array along the detector_number dimension.

    Parameters
    ----------
    detector:
        NeXus detector group.
    offset:
        Offset to add to the detector position.
    source_position:
        Position of the neutron source.
    sample_position:
        Position of the sample.
    gravity:
        Gravity vector.
    bank_sizes:
        Dictionary of detector bank sizes.
    """
    da = nexus.extract_detector_data(detector)
    if (
        sizes := (bank_sizes or {}).get(detector.get('nexus_component_name'))
    ) is not None:
        da = da.fold(dim="detector_number", sizes=sizes)
    # Note: We apply offset as early as possible, i.e., right in this function
    # the detector array from the raw loader NeXus group, to prevent a source of bugs.
    return AnyRunCalibratedDetector(
        da.assign_coords(
            position=da.coords['position'] + offset.to(unit=da.coords['position'].unit),
            source_position=source_position,
            sample_position=sample_position,
            gravity=gravity,
        )
    )


def assemble_detector_data(
    detector: AnyRunCalibratedDetector, event_data: AnyRunNeXusDetectorEventData
) -> AnyRunDetectorData:
    """
    Assemble a detector data array with event data and source- and sample-position.

    Also adds variances to the event data if they are missing.

    Parameters
    ----------
    detector:
        Calibrated detector data array.
    event_data:
        Event data array.
    """
    grouped = nexus.group_event_data(
        event_data=event_data, detector_number=detector.coords['detector_number']
    )
    return AnyRunDetectorData(
        _add_variances(grouped)
        .assign_coords(detector.coords)
        .assign_masks(detector.masks)
    )


def get_calibrated_monitor(
    monitor: AnyRunAnyNeXusMonitor,
    offset: AnyRunAnyMonitorPositionOffset,
    source_position: AnyRunSourcePosition,
) -> AnyRunAnyCalibratedMonitor:
    """
    Extract the data array corresponding to a monitor's signal field.

    The returned data array includes coords pertaining directly to the
    signal values array, but not additional information about the monitor.

    Parameters
    ----------
    monitor:
        NeXus monitor group.
    offset:
        Offset to add to the monitor position.
    source_position:
        Position of the neutron source.
    """
    return AnyRunAnyCalibratedMonitor(
        nexus.extract_monitor_data(monitor).assign_coords(
            position=monitor['position'] + offset.to(unit=monitor['position'].unit),
            source_position=source_position,
        )
    )


def assemble_monitor_data(
    monitor: AnyRunAnyCalibratedMonitor,
    event_data: AnyRunAnyNeXusMonitorEventData,
) -> AnyRunAnyMonitorData:
    """
    Assemble a monitor data array with event data.

    Also adds variances to the event data if they are missing.

    Parameters
    ----------
    monitor:
        Calibrated monitor data array.
    event_data:
        Event data array.
    """
    da = event_data.assign_coords(monitor.coords).assign_masks(monitor.masks)
    return AnyRunAnyMonitorData(_add_variances(da))


def _drop(
    children: dict[str, snx.Field | snx.Group], classes: tuple[snx.NXobject, ...]
) -> dict[str, snx.Field | snx.Group]:
    return {
        name: child
        for name, child in children.items()
        if not (isinstance(child, snx.Group) and (child.nx_class in classes))
    }


class _StrippedDetector(snx.NXdetector):
    """Detector definition without large geometry or event data for ScippNexus.

    Drops NXoff_geometry and NXevent_data groups, data is replaced by detector_number.
    """

    def __init__(
        self, attrs: dict[str, Any], children: dict[str, snx.Field | snx.Group]
    ):
        children = _drop(children, (snx.NXoff_geometry, snx.NXevent_data))
        children['data'] = children['detector_number']
        super().__init__(attrs=attrs, children=children)


class _DummyField:
    """Dummy field that can replace snx.Field in NXmonitor."""

    def __init__(self):
        self.attrs = {}
        self.sizes = {'event_time_zero': 0}
        self.dims = ('event_time_zero',)
        self.shape = (0,)

    def __getitem__(self, key: Any) -> sc.Variable:
        return sc.empty(dims=self.dims, shape=self.shape, unit=None)


class _StrippedMonitor(snx.NXmonitor):
    """Monitor definition without event data for ScippNexus.

    Drops NXevent_data group, data is replaced by a dummy field.
    """

    def __init__(
        self, attrs: dict[str, Any], children: dict[str, snx.Field | snx.Group]
    ):
        children = _drop(children, (snx.NXevent_data,))
        children['data'] = _DummyField()
        super().__init__(attrs=attrs, children=children)


def _add_variances(da: sc.DataArray) -> sc.DataArray:
    out = da.copy(deep=False)
    if out.bins is not None:
        content = out.bins.constituents['data']
        if content.variances is None:
            content.variances = content.values
    return out


def LoadMonitorWorkflow() -> sciline.Pipeline:
    """Workflow for loading monitor data from a NeXus file."""
    wf = sciline.Pipeline(
        (
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
    )
    wf[AnyRunPulseSelection] = AnyRunPulseSelection(slice(None, None))
    wf[AnyRunAnyMonitorPositionOffset] = AnyRunAnyMonitorPositionOffset(no_offset)
    return wf


def LoadDetectorWorkflow() -> sciline.Pipeline:
    """Workflow for loading detector data from a NeXus file."""
    wf = sciline.Pipeline(
        (
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
    )
    wf[AnyRunPulseSelection] = AnyRunPulseSelection(slice(None, None))
    wf[DetectorBankSizes] = DetectorBankSizes({})
    wf[AnyRunDetectorPositionOffset] = AnyRunDetectorPositionOffset(no_offset)
    return wf


def LoadNeXusWorkflow(filename: AnyRunFilename) -> sciline.Pipeline:
    """
    Workflow for loading detector and monitor data from a NeXus file.

    On creation (as opposed to when the workflow is computed), the workflow will open
    the file and determine the available detectors and monitors. This information is
    used to create a workflow that loads all available detectors and monitors. This
    can be useful when working with unknown files.

    Parameters
    ----------
    filename:
        NeXus file to load.
    """
    import pandas as pd

    wf = sciline.Pipeline()
    wf[AnyRunDetectorData] = LoadDetectorWorkflow()
    wf[AnyRunAnyMonitorData] = LoadMonitorWorkflow()
    wf[AnyRunFilename] = filename
    wf.insert(nexus.read_nexus_file_info)
    wf[nexus.NeXusFileInfo] = info = wf.compute(nexus.NeXusFileInfo)
    # Note: There is a good reason against auto-mapping here:
    # It will make extending the workflow much harder, as mapping should generally be
    # done after the workflow is complete. We may consider splitting this.
    # Note: What should we do when detector or monitor is incomplete? Parts of the
    # workflow may make sense, but as a whole it will not run.
    dets = [name for name, det in info.detectors.items() if det.n_pixel is not None]
    det_df = pd.DataFrame({NeXusDetectorName: dets}, index=dets).rename_axis('detector')
    mons = list(info.monitors)
    mon_df = pd.DataFrame({AnyNeXusMonitorName: mons}, index=mons).rename_axis(
        'monitor'
    )
    return wf.map(det_df).map(mon_df)


def with_chunks(wf: sciline.Pipeline, chunk_length: sc.Variable) -> sciline.Pipeline:
    """
    Experimental helper to transform a workflow for chunked event-data processing.

    Parameters
    ----------
    wf:
        Workflow to transform.
    chunk_length:
        Length of chunks as a time difference, i.e., with a time unit.
    """
    import pandas as pd

    info = wf.compute(nexus.NeXusFileInfo)
    # start_time and end_time are taken from event_time_zero, so we always want to
    # include the end
    bounds = sc.arange(
        'chunk', info.start_time, info.end_time + chunk_length, chunk_length
    )
    if bounds.sizes['chunk'] < 2:
        slices = [slice(None)]
    else:
        slices = [slice(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]
    # Be sure to not drop anything, use open range
    slices[0] = slice(None, slices[0].stop)
    slices[-1] = slice(slices[-1].start, None)
    return wf.map(pd.DataFrame({AnyRunPulseSelection: slices}).rename_axis('chunk'))
