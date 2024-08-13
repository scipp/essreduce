# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""Workflow and workflow components for interacting with NeXus files."""

from typing import Any

import sciline
import scipp as sc
import scippnexus as snx

from . import _nexus_loader as nexus
from .types import (
    CalibratedDetector,
    CalibratedMonitor,
    DetectorBankSizes,
    DetectorData,
    DetectorName,
    DetectorPositionOffset,
    MonitorData,
    MonitorName,
    MonitorPositionOffset,
    NeXusDetector,
    NeXusEventData,
    NeXusFileSpec,
    NeXusLocationSpec,
    NeXusMonitor,
    NeXusSample,
    NeXusSource,
    SamplePosition,
    SourcePosition,
)


def find_unique_sample(filename: NeXusFileSpec) -> NeXusLocationSpec[snx.NXsample]:
    return NeXusLocationSpec[snx.NXsample](filename=filename)


def find_unique_source(filename: NeXusFileSpec) -> NeXusLocationSpec[snx.NXsource]:
    return NeXusLocationSpec[snx.NXsource](filename=filename)


def find_monitor(
    filename: NeXusFileSpec, name: MonitorName
) -> NeXusLocationSpec[snx.NXmonitor]:
    return NeXusLocationSpec[snx.NXmonitor](filename=filename, component_name=name)


def find_detector(
    filename: NeXusFileSpec, name: DetectorName
) -> NeXusLocationSpec[snx.NXdetector]:
    return NeXusLocationSpec[snx.NXdetector](filename=filename, component_name=name)


# TODO how to distinguish det and mon event data? do we have to?


def load_nexus_sample(location: NeXusLocationSpec[snx.NXsample]) -> NeXusSample:
    # TODO Should we return default sample position of no sample?
    # sc.DataGroup({'position': sc.vector(value=[0, 0, 0], unit='m')})
    return NeXusSample(nexus.load_component(location, nx_class=snx.NXsample))


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


def load_nexus_event_data(
    location: NeXusLocationSpec[snx.NXevent_data],
) -> NeXusEventData:
    return NeXusEventData(
        nexus.load_event_data(
            file_path=location.filename,
            entry_name=location.entry_name,
            selection=location.selection,
            component_name=location.component,
        )
    )


def get_source_position(source: NeXusSource) -> SourcePosition:
    return SourcePosition(source["position"])


def get_sample_position(sample: NeXusSample) -> SamplePosition:
    return SamplePosition(sample["position"])


def get_calibrated_detector(
    detector: NeXusDetector,
    offset: DetectorPositionOffset,
    # TODO Want to be able to get det if no sample or source or no offset!
    source_position: SourcePosition,
    sample_position: SamplePosition,
    bank_sizes: DetectorBankSizes | None = None,
) -> CalibratedDetector:
    """
    Extract the data array corresponding to a detector's signal field.

    The returned data array includes coords and masks pertaining directly to the
    signal values array, but not additional information about the detector. The
    data array is reshaped to the logical detector shape, which by folding the data
    array along the detector_number dimension.
    """
    # Note: We apply offset as early as possible to prevent a source of bugs.
    # TODO gravity=gravity_vector(),
    da = nexus.extract_detector_data(detector)
    if (sizes := (bank_sizes or {}).get(detector['nexus_component_name'])) is not None:
        da = da.fold(dim="detector_number", sizes=sizes)
    position = detector['position']
    return CalibratedDetector(
        da.assign_coords(
            position=position if offset is None else position - offset,
            source_position=source_position,
            sample_position=sample_position,
        )
    )


def assemble_detector_data(
    detector: CalibratedDetector, event_data: NeXusEventData
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
    position = monitor['position']
    return CalibratedMonitor(
        nexus.extract_monitor_data(monitor).assign_coords(
            position=position if offset is None else position - offset,
            source_position=source_position,
        )
    )


def assemble_monitor_data(
    monitor: CalibratedMonitor, event_data: NeXusEventData
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
    return sciline.Pipeline(
        (
            load_nexus_monitor,
            load_nexus_event_data,
            load_nexus_source,
            get_source_position,
            get_calibrated_monitor,
            assemble_monitor_data,
        )
    )


def LoadDetectorWorkflow() -> sciline.Pipeline:
    return sciline.Pipeline(
        (
            load_nexus_detector,
            load_nexus_event_data,
            load_nexus_source,
            load_nexus_sample,
            get_source_position,
            get_sample_position,
            get_calibrated_detector,
            assemble_detector_data,
        )
    )
