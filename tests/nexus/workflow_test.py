# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
import scippnexus as snx
from scipp.testing import assert_identical

from ess.reduce import data
from ess.reduce.nexus import compute_component_position, workflow
from ess.reduce.nexus.types import (
    DetectorData,
    Filename,
    Monitor1,
    MonitorData,
    NeXusName,
    NeXusTransformation,
    SampleRun,
)
from ess.reduce.nexus.workflow import (
    GenericNeXusWorkflow,
    LoadDetectorWorkflow,
    LoadMonitorWorkflow,
)


@pytest.fixture(params=[{}, {'aux': 1}])
def group_with_no_position(request) -> workflow.NeXusComponent[snx.NXsample, SampleRun]:
    return workflow.NeXusComponent[snx.NXsample, SampleRun](sc.DataGroup(request.param))


@pytest.fixture()
def depends_on() -> snx.TransformationChain:
    translation = snx.nxtransformations.Transform(
        name='/entry/instrument/comp1/transformations/trans1',
        transformation_type='translation',
        value=sc.scalar(1.0, unit='m'),
        vector=sc.vector(value=[1.0, 2.0, 3.0], unit=''),
        depends_on=snx.DependsOn(parent='', value='.'),
        offset=None,
    )
    return snx.TransformationChain(
        parent='/entry/instrument/comp1',
        value='transformations/trans1',
        transformations={translation.name: translation},
    )


@pytest.fixture()
def transform(
    depends_on: snx.TransformationChain,
) -> NeXusTransformation[snx.NXdetector, SampleRun]:
    return NeXusTransformation.from_chain(depends_on)


def test_can_compute_position_of_group(depends_on: snx.TransformationChain) -> None:
    position = sc.vector([1.0, 2.0, 3.0], unit='m')
    group = workflow.NeXusComponent[snx.NXsource, SampleRun](
        sc.DataGroup(depends_on=depends_on)
    )
    chain = workflow.get_transformation_chain(group)
    trans = workflow.to_transformation(chain)
    assert_identical(workflow.compute_position(trans), position)


def test_given_no_sample_load_nexus_sample_returns_group_with_origin_depends_on() -> (
    None
):
    filespec = workflow.file_path_to_file_spec(
        data.loki_tutorial_sample_run_60250(), preopen=True
    )
    spec = workflow.unique_component_spec(filespec)
    assert spec.filename['/entry'][snx.NXsample] == {}
    sample = workflow.load_nexus_sample(spec)
    assert list(sample) == ['depends_on']
    chain = workflow.get_transformation_chain(sample)
    transformation = workflow.to_transformation(chain)
    position = workflow.compute_position(transformation)
    assert_identical(position, sc.vector([0.0, 0.0, 0.0], unit='m'))


def test_get_transformation_chain_raises_exception_if_position_not_found(
    group_with_no_position,
) -> None:
    with pytest.raises(KeyError, match='depends_on'):
        workflow.get_transformation_chain(group_with_no_position)


@pytest.fixture()
def nexus_detector(
    depends_on: snx.TransformationChain,
) -> workflow.NeXusComponent[snx.NXdetector, SampleRun]:
    detector_number = sc.arange('detector_number', 6, unit=None)
    data = sc.DataArray(
        sc.empty_like(detector_number),
        coords={
            'detector_number': detector_number,
            'x_pixel_offset': sc.linspace('detector_number', 0, 1, num=6, unit='m'),
        },
    )
    return workflow.NeXusComponent[snx.NXdetector, SampleRun](
        sc.DataGroup(data=data, depends_on=depends_on, nexus_component_name='detector1')
    )


@pytest.fixture()
def source_position() -> sc.Variable:
    return sc.vector([0.0, 0.0, -10.0], unit='m')


def test_get_calibrated_detector_extracts_data_field_from_nexus_detector(
    nexus_detector, transform
) -> None:
    detector = workflow.get_calibrated_detector(
        nexus_detector, offset=workflow.no_offset, bank_sizes={}, transform=transform
    )
    assert_identical(detector, compute_component_position(nexus_detector)['data'])


def test_get_calibrated_detector_folds_detector_number_if_mapping_given(
    nexus_detector, transform
) -> None:
    sizes = {'xpixel': 2, 'ypixel': 3}
    bank_sizes = {'detector1': sizes}
    detector = workflow.get_calibrated_detector(
        nexus_detector,
        offset=workflow.no_offset,
        bank_sizes=bank_sizes,
        transform=transform,
    )
    assert detector.sizes == sizes


def test_get_calibrated_detector_works_if_nexus_component_name_is_missing(
    nexus_detector, transform
):
    del nexus_detector['nexus_component_name']
    detector = workflow.get_calibrated_detector(
        nexus_detector,
        offset=workflow.no_offset,
        bank_sizes={},
        transform=transform,
    )
    assert detector.sizes == nexus_detector['data'].sizes


def test_get_calibrated_detector_adds_offset_to_position(
    nexus_detector,
    transform,
) -> None:
    offset = sc.vector([0.1, 0.2, 0.3], unit='m')
    detector = workflow.get_calibrated_detector(
        nexus_detector,
        offset=offset,
        bank_sizes={},
        transform=transform,
    )
    position = (
        compute_component_position(nexus_detector)['data'].coords['position'] + offset
    )
    assert detector.coords['position'].sizes == {'detector_number': 6}
    assert_identical(detector.coords['position'], position)


def test_get_calibrated_detector_forwards_coords(nexus_detector, transform) -> None:
    nexus_detector['data'].coords['abc'] = sc.scalar(1.2)
    detector = workflow.get_calibrated_detector(
        nexus_detector, offset=workflow.no_offset, bank_sizes={}, transform=transform
    )
    assert 'abc' in detector.coords


def test_get_calibrated_detector_forwards_masks(
    nexus_detector,
    transform,
) -> None:
    nexus_detector['data'].masks['mymask'] = sc.scalar(False)
    detector = workflow.get_calibrated_detector(
        nexus_detector, offset=workflow.no_offset, bank_sizes={}, transform=transform
    )
    assert 'mymask' in detector.masks


@pytest.fixture()
def calibrated_detector() -> workflow.CalibratedDetector[SampleRun]:
    detector_number = sc.arange('detector_number', 6, unit=None)
    return workflow.CalibratedDetector[SampleRun](
        sc.DataArray(
            sc.empty_like(detector_number),
            coords={
                'position': sc.vector([1.0, 2.0, 3.0], unit='m'),
                'detector_number': detector_number,
            },
        ).fold('detector_number', sizes={'xpixel': 2, 'ypixel': 3})
    )


@pytest.fixture()
def detector_event_data() -> workflow.NeXusData[snx.NXdetector, SampleRun]:
    content = sc.DataArray(
        sc.ones(dims=['event'], shape=[17], unit='counts'),
        coords={'event_id': sc.arange('event', 17, unit=None) % sc.index(6)},
    )
    weights = sc.bins(data=content, dim='event')
    return workflow.NeXusData[snx.NXdetector, SampleRun](
        sc.DataArray(
            weights,
            coords={
                'event_time_zero': sc.linspace(
                    dim=weights.dim, start=0, stop=1, num=weights.size, unit='s'
                )
            },
        )
    )


def test_assemble_detector_data_groups_events_by_detector_number(
    calibrated_detector, detector_event_data
) -> None:
    detector = workflow.assemble_detector_data(calibrated_detector, detector_event_data)
    assert detector.bins is not None
    assert_identical(
        detector.coords['detector_number'],
        calibrated_detector.coords['detector_number'],
    )
    # 17 events with arange%6 event_id, so 2 events in last bin
    assert_identical(
        detector.data.bins.size(),
        sc.array(dims=('xpixel', 'ypixel'), values=[[3, 3, 3], [3, 3, 2]], unit=None),
    )


def test_assemble_detector_data_does_not_add_event_id_coord(
    calibrated_detector, detector_event_data
) -> None:
    detector = workflow.assemble_detector_data(calibrated_detector, detector_event_data)
    assert 'event_id' not in detector.coords


def test_assemble_detector_data_adds_variances_to_weights(
    calibrated_detector, detector_event_data
) -> None:
    detector = workflow.assemble_detector_data(calibrated_detector, detector_event_data)
    assert detector_event_data.values[0].variances is None
    assert detector.values[0].variances is not None
    assert_identical(sc.variances(detector), sc.values(detector))


def test_assemble_detector_preserves_coords(calibrated_detector, detector_event_data):
    calibrated_detector.coords['abc'] = sc.scalar(1.2)
    detector = workflow.assemble_detector_data(calibrated_detector, detector_event_data)
    assert 'abc' in detector.coords


def test_assemble_detector_preserves_masks(calibrated_detector, detector_event_data):
    calibrated_detector.masks['mymask'] = sc.scalar(False)
    detector = workflow.assemble_detector_data(calibrated_detector, detector_event_data)
    assert 'mymask' in detector.masks


@pytest.fixture()
def nexus_monitor(
    depends_on: snx.TransformationChain,
) -> workflow.NeXusComponent[Monitor1, SampleRun]:
    data = sc.DataArray(sc.scalar(1.2), coords={'something': sc.scalar(13)})
    return workflow.NeXusComponent[Monitor1, SampleRun](
        sc.DataGroup(data=data, depends_on=depends_on)
    )


def test_get_calibrated_monitor_extracts_data_field_from_nexus_monitor(
    nexus_monitor,
) -> None:
    monitor = workflow.get_calibrated_monitor(
        nexus_monitor,
        offset=workflow.no_offset,
        source_position=sc.vector([0.0, 0.0, -10.0], unit='m'),
    )
    assert_identical(
        monitor.drop_coords(('position', 'source_position')),
        compute_component_position(nexus_monitor)['data'],
    )


def test_get_calibrated_monitor_subtracts_offset_from_position(
    nexus_monitor,
) -> None:
    offset = sc.vector([0.1, 0.2, 0.3], unit='m')
    monitor = workflow.get_calibrated_monitor(
        nexus_monitor,
        offset=offset,
        source_position=sc.vector([0.0, 0.0, -10.0], unit='m'),
    )
    assert_identical(monitor.coords['position'], sc.vector([1.1, 2.2, 3.3], unit='m'))


@pytest.fixture()
def calibrated_monitor() -> workflow.CalibratedMonitor[SampleRun, Monitor1]:
    return workflow.CalibratedMonitor[SampleRun, Monitor1](
        sc.DataArray(
            sc.scalar(0),
            coords={'position': sc.vector([1.0, 2.0, 3.0], unit='m')},
        )
    )


@pytest.fixture()
def monitor_event_data() -> workflow.NeXusData[Monitor1, SampleRun]:
    content = sc.DataArray(sc.ones(dims=['event'], shape=[17], unit='counts'))
    weights = sc.bins(data=content, dim='event')
    return workflow.NeXusData[Monitor1, SampleRun](
        sc.DataArray(
            weights,
            coords={
                'event_time_zero': sc.linspace(
                    dim=weights.dim, start=0, stop=1, num=weights.size, unit='s'
                )
            },
        )
    )


def test_assemble_monitor_data_adds_events_as_values_and_coords(
    calibrated_monitor, monitor_event_data
) -> None:
    monitor_data = workflow.assemble_monitor_data(
        calibrated_monitor, monitor_event_data
    )
    assert_identical(
        monitor_data.drop_coords(tuple(calibrated_monitor.coords)), monitor_event_data
    )


def test_assemble_monitor_data_adds_variances_to_weights(
    calibrated_monitor, monitor_event_data
) -> None:
    monitor_data = workflow.assemble_monitor_data(
        calibrated_monitor, monitor_event_data
    )
    assert_identical(
        sc.variances(monitor_data.drop_coords(tuple(calibrated_monitor.coords))),
        monitor_event_data,
    )


def test_assemble_monitor_preserves_coords(calibrated_monitor, monitor_event_data):
    calibrated_monitor.coords['abc'] = sc.scalar(1.2)
    monitor_data = workflow.assemble_monitor_data(
        calibrated_monitor, monitor_event_data
    )
    assert 'abc' in monitor_data.coords


def test_assemble_monitor_preserves_masks(calibrated_monitor, monitor_event_data):
    calibrated_monitor.masks['mymask'] = sc.scalar(False)
    monitor_data = workflow.assemble_monitor_data(
        calibrated_monitor, monitor_event_data
    )
    assert 'mymask' in monitor_data.masks


def test_load_monitor_workflow() -> None:
    wf = LoadMonitorWorkflow()
    wf[Filename[SampleRun]] = data.loki_tutorial_sample_run_60250()
    wf[NeXusName[Monitor1]] = 'monitor_1'
    da = wf.compute(MonitorData[SampleRun, Monitor1])
    assert 'position' in da.coords
    assert 'source_position' in da.coords
    assert da.bins is not None
    assert da.dims == ('event_time_zero',)


def test_load_detector_workflow() -> None:
    wf = LoadDetectorWorkflow()
    wf[Filename[SampleRun]] = data.loki_tutorial_sample_run_60250()
    wf[NeXusName[snx.NXdetector]] = 'larmor_detector'
    da = wf.compute(DetectorData[SampleRun])
    assert 'position' in da.coords
    assert 'sample_position' in da.coords
    assert 'source_position' in da.coords
    assert da.bins is not None
    assert da.dims == ('detector_number',)


def test_generic_nexus_workflow() -> None:
    wf = GenericNeXusWorkflow()
    wf[Filename[SampleRun]] = data.loki_tutorial_sample_run_60250()
    wf[NeXusName[Monitor1]] = 'monitor_1'
    wf[NeXusName[snx.NXdetector]] = 'larmor_detector'
    da = wf.compute(DetectorData[SampleRun])
    assert 'position' in da.coords
    assert 'sample_position' in da.coords
    assert 'source_position' in da.coords
    assert da.bins is not None
    assert da.dims == ('detector_number',)
    da = wf.compute(MonitorData[SampleRun, Monitor1])
    assert 'position' in da.coords
    assert 'source_position' in da.coords
    assert da.bins is not None
    assert da.dims == ('event_time_zero',)
