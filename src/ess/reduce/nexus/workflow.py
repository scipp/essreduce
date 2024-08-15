# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""Workflow and workflow components for interacting with NeXus files."""

from dataclasses import replace
from typing import Any

import pandas as pd
import sciline
import scipp as sc
import scippnexus as snx
from scipp.constants import g

from . import _nexus_loader as nexus
from .types import (
    CalibratedDetector,
    CalibratedMonitor,
    DetectorBankSizes,
    DetectorData,
    DetectorPositionOffset,
    GravityVector,
    MonitorData,
    MonitorPositionOffset,
    NeXusDetector,
    NeXusDetectorEventData,
    NeXusDetectorName,
    NeXusFileSpec,
    NeXusLocationSpec,
    NeXusMonitor,
    NeXusMonitorEventData,
    NeXusMonitorName,
    NeXusSample,
    NeXusSource,
    PulseSelection,
    SamplePosition,
    SourcePosition,
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


def unique_sample_spec(filename: NeXusFileSpec) -> NeXusLocationSpec[snx.NXsample]:
    return NeXusLocationSpec[snx.NXsample](filename=filename)


def unique_source_spec(filename: NeXusFileSpec) -> NeXusLocationSpec[snx.NXsource]:
    return NeXusLocationSpec[snx.NXsource](filename=filename)


def monitor_by_name(
    filename: NeXusFileSpec, name: NeXusMonitorName, selection: PulseSelection
) -> NeXusLocationSpec[snx.NXmonitor]:
    return NeXusLocationSpec[snx.NXmonitor](
        filename=filename, component_name=name, selection={'event_time_zero': selection}
    )


def detector_by_name(
    filename: NeXusFileSpec, name: NeXusDetectorName, selection: PulseSelection
) -> NeXusLocationSpec[snx.NXdetector]:
    return NeXusLocationSpec[snx.NXdetector](
        filename=filename, component_name=name, selection={'event_time_zero': selection}
    )


def load_nexus_sample(location: NeXusLocationSpec[snx.NXsample]) -> NeXusSample:
    """
    Load a NeXus sample group from a file.

    If there is no sample group in the file, an empty group is returned. This should
    not happen, but handling it gracefully makes testing and working with
    pre-production files easier. There should be little harm in returning an empty
    group. Subsequent extract of the sample position will then default to the origin.
    """
    try:
        dg = nexus.load_component(location, nx_class=snx.NXsample)
    except ValueError:
        dg = sc.DataGroup()
    return NeXusSample(dg)


def load_nexus_source(location: NeXusLocationSpec[snx.NXsource]) -> NeXusSource:
    return NeXusSource(nexus.load_component(location, nx_class=snx.NXsource))


def load_nexus_detector(location: NeXusLocationSpec[snx.NXdetector]) -> NeXusDetector:
    """
    Load detector from NeXus, but with event data replaced by placeholders.

    Currently the placeholder is the detector number, but this may change in the future.

    The returned object is a scipp.DataGroup, as it may contain additional information
    about the detector that cannot be represented as a single scipp.DataArray. Most
    downstream code will only be interested in the contained scipp.DataArray so this
    needs to be extracted. However, other processing steps may require the additional
    information, so it is kept in the DataGroup.

    Loading thus proceeds in three steps:

    1. This function loads the detector, but replaces the event data with placeholders.
    2. :py:func:`get_detector_array` drops the additional information, returning only
       the contained scipp.DataArray, reshaped to the logical detector shape.
       This will generally contain coordinates as well as pixel masks.
    3. :py:func:`assemble_detector_data` replaces placeholder data values with the
       event data, and adds source and sample positions.
    """
    definitions = snx.base_definitions()
    definitions["NXdetector"] = _StrippedDetector
    # The selection is only used for selecting a range of event data.
    location = replace(location, selection=())

    return NeXusDetector(
        nexus.load_component(location, nx_class=snx.NXdetector, definitions=definitions)
    )


def load_nexus_monitor(location: NeXusLocationSpec[snx.NXmonitor]) -> NeXusMonitor:
    """
    Load monitor from NeXus, but with event data replaced by placeholders.

    Currently the placeholder is a size-0 array, but this may change in the future.

    The returned object is a scipp.DataGroup, as it may contain additional information
    about the monitor that cannot be represented as a single scipp.DataArray. Most
    downstream code will only be interested in the contained scipp.DataArray so this
    needs to be extracted. However, other processing steps may require the additional
    information, so it is kept in the DataGroup.

    Loading thus proceeds in three steps:

    1. This function loads the monitor, but replaces the event data with placeholders.
    2. :py:func:`get_monitor_array` drops the additional information, returning only
         the contained scipp.DataArray.
         This will generally contain coordinates as well as pixel masks.
    3. :py:func:`assemble_monitor_data` replaces placeholder data values with the
         event data, and adds source and sample positions.
    """
    definitions = snx.base_definitions()
    definitions["NXmonitor"] = _StrippedMonitor
    return NeXusMonitor(
        nexus.load_component(location, nx_class=snx.NXmonitor, definitions=definitions)
    )


def load_nexus_detector_event_data(
    location: NeXusLocationSpec[snx.NXdetector],
) -> NeXusDetectorEventData:
    return NeXusDetectorEventData(
        nexus.load_event_data(
            file_path=location.filename,
            entry_name=location.entry_name,
            selection=location.selection,
            component_name=location.component_name,
        )
    )


def load_nexus_monitor_event_data(
    location: NeXusLocationSpec[snx.NXmonitor],
) -> NeXusMonitorEventData:
    return NeXusMonitorEventData(
        nexus.load_event_data(
            file_path=location.filename,
            entry_name=location.entry_name,
            selection=location.selection,
            component_name=location.component_name,
        )
    )


def get_source_position(source: NeXusSource) -> SourcePosition:
    return SourcePosition(source["position"])


def get_sample_position(sample: NeXusSample) -> SamplePosition:
    return SamplePosition(sample.get("position", origin))


def get_calibrated_detector(
    detector: NeXusDetector,
    *,
    offset: DetectorPositionOffset,
    source_position: SourcePosition,
    sample_position: SamplePosition,
    gravity: GravityVector,
    bank_sizes: DetectorBankSizes,
) -> CalibratedDetector:
    """
    Extract the data array corresponding to a detector's signal field.

    The returned data array includes coords and masks pertaining directly to the
    signal values array, but not additional information about the detector. The
    data array is reshaped to the logical detector shape, which by folding the data
    array along the detector_number dimension.
    """
    da = nexus.extract_detector_data(detector)
    if (sizes := (bank_sizes or {}).get(detector['nexus_component_name'])) is not None:
        da = da.fold(dim="detector_number", sizes=sizes)
    # Note: We apply offset as early as possible, i.e., right in this function
    # the detector array from the raw loader NeXus group, to prevent a source of bugs.
    position = detector['position']
    return CalibratedDetector(
        da.assign_coords(
            position=position if offset is no_offset else position - offset,
            source_position=source_position,
            sample_position=sample_position,
            gravity=gravity,
        )
    )


def assemble_detector_data(
    detector: CalibratedDetector, event_data: NeXusDetectorEventData
) -> DetectorData:
    """
    Assemble a detector data array with event data and source- and sample-position.

    Also adds variances to the event data if they are missing.
    """
    grouped = nexus.group_event_data(
        event_data=event_data, detector_number=detector.coords['detector_number']
    )
    return DetectorData(
        _add_variances(grouped)
        .assign_coords(detector.coords)
        .assign_masks(detector.masks)
    )


def get_calibrated_monitor(
    monitor: NeXusMonitor,
    offset: MonitorPositionOffset,
    source_position: SourcePosition,
) -> CalibratedMonitor:
    """
    Extract the data array corresponding to a monitor's signal field.

    The returned data array includes coords pertaining directly to the
    signal values array, but not additional information about the monitor.
    """
    return CalibratedMonitor(
        nexus.extract_monitor_data(monitor).assign_coords(
            position=monitor['position'] - offset,
            source_position=source_position,
        )
    )


def assemble_monitor_data(
    monitor: CalibratedMonitor, event_data: NeXusMonitorEventData
) -> MonitorData:
    """
    Assemble a monitor data array with event data.

    Also adds variances to the event data if they are missing.
    """
    da = event_data.assign_coords(monitor.coords).assign_masks(monitor.masks)
    return MonitorData(_add_variances(da))


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
    wf = sciline.Pipeline(
        (
            unique_source_spec,
            monitor_by_name,
            load_nexus_monitor,
            load_nexus_monitor_event_data,
            load_nexus_source,
            get_source_position,
            get_calibrated_monitor,
            assemble_monitor_data,
        )
    )
    wf[PulseSelection] = PulseSelection(())
    wf[MonitorPositionOffset] = MonitorPositionOffset(no_offset)
    return wf


def LoadDetectorWorkflow() -> sciline.Pipeline:
    wf = sciline.Pipeline(
        (
            unique_source_spec,
            unique_sample_spec,
            detector_by_name,
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
    wf[PulseSelection] = PulseSelection(())
    wf[DetectorBankSizes] = DetectorBankSizes({})
    wf[DetectorPositionOffset] = DetectorPositionOffset(no_offset)
    return wf


def LoadNeXusWorkflow(filename: NeXusFileSpec) -> sciline.Pipeline:
    """Opens and inspects a NeXus file, returning a summary of its contents."""
    wf = sciline.Pipeline()
    wf[DetectorData] = LoadDetectorWorkflow()
    wf[MonitorData] = LoadMonitorWorkflow()
    wf[NeXusFileSpec] = filename
    wf.insert(nexus.read_nexus_file_info)
    wf[nexus.NeXusFileInfo] = info = wf.compute(nexus.NeXusFileInfo)
    # TODO There is a good reason against auto-mapping here:
    # It will make extending the workflow much harder, as mapping should generally be
    # done after the workflow is complete.
    # TODO What should we do when detector or monitor is incomplete? Parts of the
    # workflow may make sense, but as a whole it will not run.
    dets = [name for name, det in info.detectors.items() if det.n_pixel is not None]
    det_df = pd.DataFrame({NeXusDetectorName: dets}, index=dets).rename_axis('detector')
    mons = list(info.monitors)
    mon_df = pd.DataFrame({NeXusMonitorName: mons}, index=mons).rename_axis('monitor')
    return wf.map(det_df).map(mon_df)


def with_chunks(wf: sciline.Pipeline, chunk_length: sc.Variable) -> sciline.Pipeline:
    info = wf.compute(nexus.NeXusFileInfo)
    # start_time and end_time are taken from event_time_zero, so we always want to
    # include the end
    # TODO dtype and units
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
    wf = wf.map(pd.DataFrame({PulseSelection: slices}).rename_axis('chunk'))
    return wf
