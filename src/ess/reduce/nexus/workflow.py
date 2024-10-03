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
    CalibratedBeamline,
    CalibratedDetector,
    CalibratedMonitor,
    ComponentPosition,
    ComponentType,
    DetectorBankSizes,
    DetectorData,
    DetectorPositionOffset,
    Filename,
    GravityVector,
    MonitorData,
    MonitorPositionOffset,
    MonitorType,
    NeXusComponent,
    NeXusComponentLocationSpec,
    NeXusComponentName,
    NeXusData,
    NeXusDataLocationSpec,
    NeXusFileSpec,
    NeXusTransformation,
    NeXusTransformationChain,
    PreopenNeXusFile,
    PulseSelection,
    RunType,
)

origin = sc.vector([0, 0, 0], unit="m")
"""The origin, used as default sample position."""
no_offset = sc.vector([0, 0, 0], unit="m")
"""Offset that does not change the position."""


def file_path_to_file_spec(
    filename: Filename[RunType], preopen: PreopenNeXusFile
) -> NeXusFileSpec[RunType]:
    return NeXusFileSpec[RunType](
        snx.File(filename, definitions=definitions) if preopen else filename
    )


def no_monitor_position_offset() -> MonitorPositionOffset[RunType, MonitorType]:
    return MonitorPositionOffset[RunType, MonitorType](no_offset)


def no_detector_position_offset() -> DetectorPositionOffset[RunType]:
    return DetectorPositionOffset[RunType](no_offset)


def all_pulses() -> PulseSelection[RunType]:
    """Select all neutron pulses in the data."""
    return PulseSelection[RunType](slice(None, None))


def gravity_vector_neg_y() -> GravityVector:
    """
    Gravity vector for default instrument coordinate system where y is up.
    """
    return GravityVector(sc.vector(value=[0, -1, 0]) * g)


def unique_component_spec(
    filename: NeXusFileSpec[RunType],
) -> NeXusComponentLocationSpec[ComponentType, RunType]:
    """
    Create a location spec for a unique component group in a NeXus file.

    Parameters
    ----------
    filename:
        NeXus file to use for the location spec.
    """
    return NeXusComponentLocationSpec[ComponentType, RunType](filename=filename.value)


def monitor_by_name(
    filename: NeXusFileSpec[RunType], name: NeXusComponentName[MonitorType]
) -> NeXusComponentLocationSpec[MonitorType, RunType]:
    """
    Create a location spec for a monitor group in a NeXus file.

    Parameters
    ----------
    filename:
        NeXus file to use for the location spec.
    name:
        Name of the monitor group.
    """
    return NeXusComponentLocationSpec[MonitorType, RunType](
        filename=filename.value, component_name=name
    )


def detector_by_name(
    filename: NeXusFileSpec[RunType], name: NeXusComponentName[snx.NXdetector]
) -> NeXusComponentLocationSpec[snx.NXdetector, RunType]:
    """
    Create a location spec for a detector group in a NeXus file.

    Parameters
    ----------
    filename:
        NeXus file to use for the location spec.
    name:
        Name of the detector group.
    """
    return NeXusComponentLocationSpec[snx.NXdetector, RunType](
        filename=filename.value, component_name=name
    )


def data_by_name(
    filename: NeXusFileSpec[RunType],
    name: NeXusComponentName[ComponentType],
    selection: PulseSelection[RunType],
) -> NeXusDataLocationSpec[ComponentType, RunType]:
    """
    Create a location spec for monitor or detector data in a NeXus file.

    Parameters
    ----------
    filename:
        NeXus file to use for the location spec.
    name:
        Name of the monitor or detector group.
    selection:
        Time range (start and stop as a Python slice object).
    """
    return NeXusDataLocationSpec[ComponentType, RunType](
        filename=filename.value, component_name=name, selection=selection.value
    )


def load_nexus_sample(
    location: NeXusComponentLocationSpec[snx.NXsample, RunType],
) -> NeXusComponent[snx.NXsample, RunType]:
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
        dg = sc.DataGroup(depends_on=snx.TransformationChain(parent='', value='.'))
    return NeXusComponent[snx.NXsample, RunType](dg)


def load_nexus_source(
    location: NeXusComponentLocationSpec[snx.NXsource, RunType],
) -> NeXusComponent[snx.NXsource, RunType]:
    """
    Load a NeXus source group from a file.

    Parameters
    ----------
    location:
        Location spec for the source group.
    """
    return NeXusComponent[snx.NXsource, RunType](
        nexus.load_component(location, nx_class=snx.NXsource)
    )


def load_nexus_detector(
    location: NeXusComponentLocationSpec[snx.NXdetector, RunType],
) -> NeXusComponent[snx.NXdetector, RunType]:
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
    # The selection is only used for selecting a range of event data.
    location = replace(location, selection=())

    return NeXusComponent[snx.NXdetector, RunType](
        nexus.load_component(location, nx_class=snx.NXdetector, definitions=definitions)
    )


def load_nexus_monitor(
    location: NeXusComponentLocationSpec[MonitorType, RunType],
) -> NeXusComponent[MonitorType, RunType]:
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
    2. :py:func:`get_calibrated_monitor` drops the additional information, returning
       only the contained scipp.DataArray.
       This will generally contain coordinates as well as pixel masks.
    3. :py:func:`assemble_monitor_data` replaces placeholder data values with the
       event data, and adds source and sample positions.

    Parameters
    ----------
    location:
        Location spec for the monitor group.
    """
    return NeXusComponent[MonitorType, RunType](
        nexus.load_component(location, nx_class=snx.NXmonitor, definitions=definitions)
    )


def load_nexus_data(
    location: NeXusDataLocationSpec[ComponentType, RunType],
) -> NeXusData[ComponentType, RunType]:
    """
    Load event or histogram data from a NeXus detector group.

    Parameters
    ----------
    location:
        Location spec for the detector group.
    """
    return NeXusData[ComponentType, RunType](
        nexus.load_data(
            file_path=location.filename,
            entry_name=location.entry_name,
            selection=location.selection,
            component_name=location.component_name,
        )
    )


def get_transformation_chain(
    detector: NeXusComponent[ComponentType, RunType],
) -> NeXusTransformationChain[ComponentType, RunType]:
    """
    Extract the transformation chain from a NeXus detector group.

    Parameters
    ----------
    detector:
        NeXus detector group.
    """
    chain = detector['depends_on']
    return NeXusTransformationChain[ComponentType, RunType].from_base(chain)


def to_transformation(
    chain: NeXusTransformationChain[ComponentType, RunType],
) -> NeXusTransformation[ComponentType, RunType]:
    """Convert transformation chain into a single transformation matrix."""
    return NeXusTransformation[ComponentType, RunType].from_chain(chain)


def compute_position(
    transformation: NeXusTransformation[ComponentType, RunType],
) -> ComponentPosition[ComponentType, RunType]:
    """Compute the position of a component from a transformation matrix."""
    return ComponentPosition[ComponentType, RunType](transformation.value * origin)


def get_calibrated_detector(
    detector: NeXusComponent[snx.NXdetector, RunType],
    *,
    transform: NeXusTransformation[snx.NXdetector, RunType],
    offset: DetectorPositionOffset[RunType],
    bank_sizes: DetectorBankSizes,
) -> CalibratedDetector[RunType]:
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
    da = nexus.extract_signal_data_array(detector)
    if (
        sizes := (bank_sizes or {}).get(detector.get('nexus_component_name'))
    ) is not None:
        da = da.fold(dim="detector_number", sizes=sizes)
    # Note: We apply offset as early as possible, i.e., right in this function
    # the detector array from the raw loader NeXus group, to prevent a source of bugs.
    base_pos = snx.zip_pixel_offsets(da.coords)
    position = transform.value * base_pos
    return CalibratedDetector[RunType](
        da.assign_coords(
            position=position + offset.to(unit=position.unit),
        )
    )


def assemble_beamline(
    detector: CalibratedDetector[RunType],
    source_position: ComponentPosition[snx.NXsource, RunType],
    sample_position: ComponentPosition[snx.NXsample, RunType],
    gravity: GravityVector,
) -> CalibratedBeamline[RunType]:
    return CalibratedBeamline[RunType](
        detector.assign_coords(
            source_position=source_position,
            sample_position=sample_position,
            gravity=gravity,
        )
    )


def assemble_detector_data(
    detector: CalibratedBeamline[RunType],
    event_data: NeXusData[snx.NXdetector, RunType],
) -> DetectorData[RunType]:
    """
    Assemble a detector data array with event data.

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
    return DetectorData[RunType](
        _add_variances(grouped)
        .assign_coords(detector.coords)
        .assign_masks(detector.masks)
    )


def get_calibrated_monitor(
    monitor: NeXusComponent[MonitorType, RunType],
    offset: MonitorPositionOffset[RunType, MonitorType],
    source_position: ComponentPosition[snx.NXsource, RunType],
) -> CalibratedMonitor[RunType, MonitorType]:
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
    monitor = nexus.compute_component_position(monitor)
    return CalibratedMonitor[RunType, MonitorType](
        nexus.extract_signal_data_array(monitor).assign_coords(
            position=monitor['position'] + offset.to(unit=monitor['position'].unit),
            source_position=source_position,
        )
    )


def assemble_monitor_data(
    monitor: CalibratedMonitor[RunType, MonitorType],
    data: NeXusData[MonitorType, RunType],
) -> MonitorData[RunType, MonitorType]:
    """
    Assemble a monitor data array with event data.

    Also adds variances to the event data if they are missing.

    Parameters
    ----------
    monitor:
        Calibrated monitor data array.
    data:
        Data array with neutron counts.
    """
    da = data.assign_coords(monitor.coords).assign_masks(monitor.masks)
    return MonitorData[RunType, MonitorType](_add_variances(da))


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


definitions = snx.base_definitions()
definitions["NXdetector"] = _StrippedDetector
definitions["NXmonitor"] = _StrippedMonitor


_common_providers = (
    gravity_vector_neg_y,
    file_path_to_file_spec,
    all_pulses,
    unique_component_spec,
    get_transformation_chain,
    to_transformation,
    compute_position,
    load_nexus_data,
    data_by_name,
)

_monitor_providers = (
    no_monitor_position_offset,
    monitor_by_name,
    load_nexus_monitor,
    load_nexus_source,
    get_calibrated_monitor,
    assemble_monitor_data,
)

_detector_providers = (
    no_detector_position_offset,
    detector_by_name,
    load_nexus_detector,
    load_nexus_source,
    load_nexus_sample,
    get_calibrated_detector,
    assemble_beamline,
    assemble_detector_data,
)


def LoadMonitorWorkflow() -> sciline.Pipeline:
    """Generic workflow for loading monitor data from a NeXus file."""
    wf = sciline.Pipeline((*_common_providers, *_monitor_providers))
    wf[PreopenNeXusFile] = PreopenNeXusFile(False)
    return wf


def LoadDetectorWorkflow() -> sciline.Pipeline:
    """Generic workflow for loading detector data from a NeXus file."""
    wf = sciline.Pipeline((*_common_providers, *_detector_providers))
    wf[DetectorBankSizes] = DetectorBankSizes({})
    wf[PreopenNeXusFile] = PreopenNeXusFile(False)
    return wf


def GenericNeXusWorkflow() -> sciline.Pipeline:
    """Generic workflow for loading detector and monitor data from a NeXus file."""
    wf = sciline.Pipeline(
        (*_common_providers, *_monitor_providers, *_detector_providers)
    )
    wf[DetectorBankSizes] = DetectorBankSizes({})
    wf[PreopenNeXusFile] = PreopenNeXusFile(False)
    return wf
