# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""Workflow and workflow components for interacting with NeXus files."""

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
    Component,
    DetectorBankSizes,
    DetectorData,
    DetectorPositionOffset,
    Filename,
    GravityVector,
    MonitorData,
    MonitorPositionOffset,
    MonitorType,
    NeXusClass,
    NeXusComponent,
    NeXusComponentLocationSpec,
    NeXusData,
    NeXusDataLocationSpec,
    NeXusFileSpec,
    NeXusName,
    NeXusTransformation,
    NeXusTransformationChain,
    Position,
    PreopenNeXusFile,
    PulseSelection,
    RunType,
    UniqueComponent,
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


def component_spec_by_name(
    filename: NeXusFileSpec[RunType], name: NeXusName[Component]
) -> NeXusComponentLocationSpec[Component, RunType]:
    """
    Create a location spec for a component group in a NeXus file.

    Parameters
    ----------
    filename:
        NeXus file to use for the location spec.
    name:
        Name of the component group.
    """
    return NeXusComponentLocationSpec[Component, RunType](
        filename=filename.value, component_name=name
    )


def unique_component_spec(
    filename: NeXusFileSpec[RunType],
) -> NeXusComponentLocationSpec[UniqueComponent, RunType]:
    """
    Create a location spec for a unique component group in a NeXus file.

    Parameters
    ----------
    filename:
        NeXus file to use for the location spec.
    """
    return NeXusComponentLocationSpec[UniqueComponent, RunType](filename=filename.value)


def data_by_name(
    filename: NeXusFileSpec[RunType],
    name: NeXusName[Component],
    selection: PulseSelection[RunType],
) -> NeXusDataLocationSpec[Component, RunType]:
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
    return NeXusDataLocationSpec[Component, RunType](
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


def nx_class_for_monitor() -> NeXusClass[MonitorType]:
    return NeXusClass[MonitorType](snx.NXmonitor)


def nx_class_for_detector() -> NeXusClass[snx.NXdetector]:
    return NeXusClass[snx.NXdetector](snx.NXdetector)


def nx_class_for_source() -> NeXusClass[snx.NXsource]:
    return NeXusClass[snx.NXsource](snx.NXsource)


def nx_class_for_sample() -> NeXusClass[snx.NXsample]:
    return NeXusClass[snx.NXsample](snx.NXsample)


def load_nexus_component(
    location: NeXusComponentLocationSpec[Component, RunType],
    nx_class: NeXusClass[Component],
) -> NeXusComponent[Component, RunType]:
    """
    Load a NeXus component group from a file.

    When loading a detector or monitor, event data is replaced by placeholders.

    As the event data can be large and is not needed at this stage, it is replaced by
    a placeholder. A placeholder is used to allow for returning a scipp.DataArray, which
    is what most downstream code will expect.
    Currently the placeholder is the detector number (for detectors) or a size-0 array
    (for monitors), but this may change in the future.

    The returned object is a scipp.DataGroup, as it may contain additional information
    about the detector that cannot be represented as a single scipp.DataArray. Most
    downstream code will only be interested in the contained scipp.DataArray so this
    needs to be extracted. However, other processing steps may require the additional
    information, so it is kept in the DataGroup.

    Parameters
    ----------
    location:
        Location spec for the source group.
    nx_class:
        NX_class to identify the component.
    """
    return NeXusComponent[Component, RunType](
        nexus.load_component(location, nx_class=nx_class, definitions=definitions)
    )


def load_nexus_data(
    location: NeXusDataLocationSpec[Component, RunType],
) -> NeXusData[Component, RunType]:
    """
    Load event or histogram data from a NeXus detector group.

    Parameters
    ----------
    location:
        Location spec for the detector group.
    """
    return NeXusData[Component, RunType](
        nexus.load_data(
            file_path=location.filename,
            entry_name=location.entry_name,
            selection=location.selection,
            component_name=location.component_name,
        )
    )


def get_transformation_chain(
    detector: NeXusComponent[Component, RunType],
) -> NeXusTransformationChain[Component, RunType]:
    """
    Extract the transformation chain from a NeXus detector group.

    Parameters
    ----------
    detector:
        NeXus detector group.
    """
    chain = detector['depends_on']
    return NeXusTransformationChain[Component, RunType](chain)


def to_transformation(
    chain: NeXusTransformationChain[Component, RunType],
) -> NeXusTransformation[Component, RunType]:
    """Convert transformation chain into a single transformation matrix."""
    return NeXusTransformation[Component, RunType].from_chain(chain)


def compute_position(
    transformation: NeXusTransformation[Component, RunType],
) -> Position[Component, RunType]:
    """Compute the position of a component from a transformation matrix."""
    return Position[Component, RunType](transformation.value * origin)


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
    position = transform.value * snx.zip_pixel_offsets(da.coords)
    return CalibratedDetector[RunType](
        da.assign_coords(position=position + offset.to(unit=position.unit))
    )


def assemble_beamline(
    detector: CalibratedDetector[RunType],
    source_position: Position[snx.NXsource, RunType],
    sample_position: Position[snx.NXsample, RunType],
    gravity: GravityVector,
) -> CalibratedBeamline[RunType]:
    """
    Add beamline information (gravity vector, source- and sample-position) to detector.

    This is performed separately and after :py:func:`get_calibrated_detector` to avoid
    as false dependency of, e.g., the reshaped detector numbers on the sample position.
    The latter can change during a run, e.g., for a rotating sample. The detector
    numbers might be used, e.g., to mask certain detector pixels, and should not depend
    on the sample position.

    Parameters
    ----------
    detector:
        NeXus detector group.
    source_position:
        Position of the neutron source.
    sample_position:
        Position of the sample.
    gravity:
        Gravity vector.
    """
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
    source_position: Position[snx.NXsource, RunType],
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
    component_spec_by_name,
    unique_component_spec,  # after component_spec_by_name, partially overrides
    get_transformation_chain,
    to_transformation,
    compute_position,
    load_nexus_data,
    load_nexus_component,
    data_by_name,
    nx_class_for_detector,
    nx_class_for_monitor,
    nx_class_for_source,
    nx_class_for_sample,
)

_monitor_providers = (
    no_monitor_position_offset,
    get_calibrated_monitor,
    assemble_monitor_data,
)

_detector_providers = (
    no_detector_position_offset,
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
