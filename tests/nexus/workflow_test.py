# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from datetime import UTC, datetime
from pathlib import Path

import pytest
import sciline as sl
import scipp as sc
import scippnexus as snx
from scipp.testing import assert_identical

from ess.reduce.nexus import compute_component_position, load_from_path, workflow
from ess.reduce.nexus.types import (
    BackgroundRun,
    Beamline,
    EmptyBeamRun,
    EmptyDetector,
    Filename,
    FrameMonitor0,
    FrameMonitor1,
    FrameMonitor2,
    Measurement,
    MonitorType,
    NeXusComponentLocationSpec,
    NeXusFileSpec,
    NeXusLocationSpec,
    NeXusName,
    NeXusTransformation,
    PreopenNeXusFile,
    RawChoppers,
    RawDetector,
    RawMonitor,
    RunType,
    SampleRun,
    TimeInterval,
    TransmissionMonitor,
)
from ess.reduce.nexus.workflow import (
    GenericNeXusWorkflow,
    LoadDetectorWorkflow,
    LoadMonitorWorkflow,
)


@pytest.fixture(params=[{}, {'aux': 1}])
def group_with_no_position(request) -> workflow.NeXusComponent[snx.NXsample, SampleRun]:
    return workflow.NeXusComponent[snx.NXsample, SampleRun](sc.DataGroup(request.param))


@pytest.fixture
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
        transformations=sc.DataGroup({translation.name: translation}),
    )


@pytest.fixture
def time_dependent_depends_on() -> snx.TransformationChain:
    """A chain of two transformations, the second one time-dependent."""
    trans1 = snx.nxtransformations.Transform(
        name='/entry/instrument/comp1/transformations/trans1',
        transformation_type='translation',
        value=sc.scalar(1.0, unit='m'),
        vector=sc.vector(value=[1.0, 0.0, 0.0], unit=''),
        depends_on=snx.DependsOn(
            parent='/entry/instrument/comp1/transformations', value='trans2'
        ),
    )
    trans2 = snx.nxtransformations.Transform(
        name='/entry/instrument/comp1/transformations/trans2',
        transformation_type='translation',
        value=sc.DataArray(
            sc.array(dims=['time'], values=[1.0, 2.0, 3.0], unit='m'),
            coords={'time': sc.array(dims=['time'], values=[0.0, 1.0, 2.0], unit='s')},
        ),
        vector=sc.vector(value=[0.0, 1.0, 0.0], unit=''),
        depends_on=snx.DependsOn(
            parent='/entry/instrument/comp1/transformations', value='.'
        ),
    )
    return snx.TransformationChain(
        parent='/entry/instrument/comp1',
        value='transformations/trans1',
        transformations=sc.DataGroup({trans1.name: trans1, trans2.name: trans2}),
    )


@pytest.fixture
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
    trans = workflow.to_transformation(
        chain,
        interval=TimeInterval(slice(None, None)),
    )
    assert_identical(workflow.compute_position(trans), position)


def test_to_transform_with_positional_time_interval(
    time_dependent_depends_on: snx.TransformationChain,
) -> None:
    origin = sc.vector([0.0, 0.0, 0.0], unit='m')

    transform = workflow.to_transformation(
        time_dependent_depends_on,
        TimeInterval(slice(0, 1)),
    ).value
    assert sc.identical(transform * origin, sc.vector([1.0, 1.0, 0.0], unit='m'))

    transform = workflow.to_transformation(
        time_dependent_depends_on,
        TimeInterval(slice(1, 2)),
    ).value
    assert sc.identical(transform * origin, sc.vector([1.0, 2.0, 0.0], unit='m'))

    transform = workflow.to_transformation(
        time_dependent_depends_on,
        TimeInterval(slice(2, 3)),
    ).value
    assert sc.identical(transform * origin, sc.vector([1.0, 3.0, 0.0], unit='m'))


def test_to_transform_with_label_based_time_interval_single_point(
    time_dependent_depends_on: snx.TransformationChain,
) -> None:
    origin = sc.vector([0.0, 0.0, 0.0], unit='m')

    transform = workflow.to_transformation(
        time_dependent_depends_on,
        TimeInterval(slice(sc.scalar(0.1, unit='s'), sc.scalar(0.9, unit='s'))),
    ).value
    assert sc.identical(transform * origin, sc.vector([1.0, 1.0, 0.0], unit='m'))

    transform = workflow.to_transformation(
        time_dependent_depends_on,
        TimeInterval(slice(sc.scalar(1.1, unit='s'), sc.scalar(1.9, unit='s'))),
    ).value
    assert sc.identical(transform * origin, sc.vector([1.0, 2.0, 0.0], unit='m'))

    transform = workflow.to_transformation(
        time_dependent_depends_on,
        TimeInterval(slice(sc.scalar(2.1, unit='s'), sc.scalar(2.9, unit='s'))),
    ).value
    assert sc.identical(transform * origin, sc.vector([1.0, 3.0, 0.0], unit='m'))

    # No more new values after 2 seconds
    transform = workflow.to_transformation(
        time_dependent_depends_on,
        TimeInterval(slice(sc.scalar(1000.0, unit='s'), sc.scalar(2000.0, unit='s'))),
    ).value
    assert sc.identical(transform * origin, sc.vector([1.0, 3.0, 0.0], unit='m'))


def test_to_transform_raises_if_interval_does_not_yield_unique_value(
    time_dependent_depends_on: snx.TransformationChain,
) -> None:
    with pytest.raises(ValueError, match='Transform is time-dependent'):
        workflow.to_transformation(
            time_dependent_depends_on,
            TimeInterval(slice(sc.scalar(0.1, unit='s'), sc.scalar(1.9, unit='s'))),
        )


def test_given_no_sample_load_nexus_sample_returns_group_with_origin_depends_on(
    loki_tutorial_sample_run_60250: Path,
) -> None:
    filespec = workflow.file_path_to_file_spec(
        loki_tutorial_sample_run_60250, preopen=True
    )
    spec = workflow.unique_component_spec(filespec)
    assert spec.filename['/entry'][snx.NXsample] == {}
    sample = workflow.load_nexus_sample(spec)
    assert list(sample) == ['depends_on']
    chain = workflow.get_transformation_chain(sample)
    transformation = workflow.to_transformation(
        chain,
        interval=TimeInterval(slice(None, None)),
    )
    position = workflow.compute_position(transformation)
    assert_identical(position, sc.vector([0.0, 0.0, 0.0], unit='m'))


def test_get_transformation_chain_raises_exception_if_position_not_found(
    group_with_no_position,
) -> None:
    with pytest.raises(KeyError, match='depends_on'):
        workflow.get_transformation_chain(group_with_no_position)


@pytest.fixture
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


@pytest.fixture
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
    nexus_detector, transform
) -> None:
    offset = sc.vector([0.1, 0.2, 0.3], unit='m')
    detector = workflow.get_calibrated_detector(
        nexus_detector, offset=offset, bank_sizes={}, transform=transform
    )
    position = (
        compute_component_position(nexus_detector)['data'].coords['position'] + offset
    )
    assert detector.coords['position'].sizes == {'detector_number': 6}
    assert_identical(detector.coords['position'], position)


def test_get_calibrated_detector_position_dims_matches_data_dims(
    nexus_detector, transform
) -> None:
    nexus_detector2d = nexus_detector.fold('detector_number', sizes={'y': 2, 'x': 3})
    nexus_detector2d['data'].coords['x_pixel_offset'] = sc.linspace(
        'x', 0, 1, num=3, unit='m'
    )
    nexus_detector2d['data'].coords['y_pixel_offset'] = sc.linspace(
        'y', 0, 1, num=2, unit='m'
    )
    offset = sc.vector([0.1, 0.2, 0.3], unit='m')
    detector = workflow.get_calibrated_detector(
        nexus_detector2d, offset=offset, bank_sizes={}, transform=transform
    )
    assert detector.sizes == {'y': 2, 'x': 3}
    assert detector.coords['position'].sizes == {'y': 2, 'x': 3}


@pytest.mark.parametrize(
    'transform_value',
    [
        sc.spatial.translation(value=[1.0, 2.0, 3.0], unit='m'),
        sc.spatial.rotations_from_rotvecs(sc.vector([0.1, 0.2, 0.3], unit='rad')),
    ],
)
def test_get_calibrated_detector_position_unit_matches_offset_unit(
    nexus_detector, transform_value
) -> None:
    transform = NeXusTransformation(transform_value)
    nexus_detector['data'].coords['x_pixel_offset'] = (
        nexus_detector['data'].coords['x_pixel_offset'].to(unit='mm')
    )
    offset = sc.vector([0.1, 0.2, 0.3], unit='m')
    detector = workflow.get_calibrated_detector(
        nexus_detector, offset=offset, bank_sizes={}, transform=transform
    )
    assert detector.coords['position'].unit == 'mm'


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


@pytest.fixture
def calibrated_detector() -> workflow.EmptyDetector[SampleRun]:
    detector_number = sc.arange('detector_number', 6, unit=None)
    return workflow.EmptyDetector[SampleRun](
        sc.DataArray(
            sc.empty_like(detector_number),
            coords={
                'position': sc.vector([1.0, 2.0, 3.0], unit='m'),
                'detector_number': detector_number,
            },
        ).fold('detector_number', sizes={'xpixel': 2, 'ypixel': 3})
    )


@pytest.fixture
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


@pytest.fixture
def nexus_monitor(
    depends_on: snx.TransformationChain,
) -> workflow.NeXusComponent[FrameMonitor1, SampleRun]:
    data = sc.DataArray(sc.scalar(1.2), coords={'something': sc.scalar(13)})
    return workflow.NeXusComponent[FrameMonitor1, SampleRun](
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


@pytest.fixture
def calibrated_monitor() -> workflow.EmptyMonitor[SampleRun, FrameMonitor1]:
    return workflow.EmptyMonitor[SampleRun, FrameMonitor1](
        sc.DataArray(
            sc.scalar(0),
            coords={'position': sc.vector([1.0, 2.0, 3.0], unit='m')},
        )
    )


@pytest.fixture
def monitor_event_data() -> workflow.NeXusData[FrameMonitor1, SampleRun]:
    content = sc.DataArray(sc.ones(dims=['event'], shape=[17], unit='counts'))
    weights = sc.bins(data=content, dim='event')
    return workflow.NeXusData[FrameMonitor1, SampleRun](
        sc.DataArray(
            weights,
            coords={
                'event_time_zero': sc.linspace(
                    dim=weights.dim, start=0, stop=1, num=weights.size, unit='s'
                )
            },
        )
    )


@pytest.fixture
def monitor_histogram_data() -> workflow.NeXusData[FrameMonitor1, SampleRun]:
    time = sc.epoch(unit='ns') + sc.arange('time', 1, 6, unit='s').to(unit='ns')
    frame_time = sc.arange('frame_time', 12, unit='ms').to(unit='ns')
    return workflow.NeXusData[FrameMonitor1, SampleRun](
        sc.DataArray(
            10.0
            * sc.arange('x', 5 * 12, unit='counts').fold(
                'x', sizes={'time': 5, 'frame_time': 12}
            ),
            coords={'time': time, 'frame_time': frame_time},
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


def test_assemble_monitor_data_adds_variances_to_events(
    calibrated_monitor, monitor_event_data
) -> None:
    monitor_data = workflow.assemble_monitor_data(
        calibrated_monitor, monitor_event_data
    )
    assert_identical(
        sc.variances(monitor_data.drop_coords(tuple(calibrated_monitor.coords))),
        monitor_event_data,
    )


def test_assemble_monitor_data_adds_histogram_as_values_and_coords(
    calibrated_monitor, monitor_histogram_data
) -> None:
    monitor_data = workflow.assemble_monitor_data(
        calibrated_monitor, monitor_histogram_data
    )
    assert_identical(
        monitor_data.drop_coords(tuple(calibrated_monitor.coords)),
        monitor_histogram_data,
    )


def test_assemble_monitor_data_adds_variances_to_weights(
    calibrated_monitor, monitor_histogram_data
) -> None:
    monitor_data = workflow.assemble_monitor_data(
        calibrated_monitor, monitor_histogram_data
    )
    assert_identical(
        sc.variances(monitor_data.drop_coords(tuple(calibrated_monitor.coords))),
        monitor_histogram_data,
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


def test_load_event_monitor_workflow(loki_tutorial_sample_run_60250: Path) -> None:
    wf = LoadMonitorWorkflow(run_types=[SampleRun], monitor_types=[FrameMonitor1])
    wf[Filename[SampleRun]] = loki_tutorial_sample_run_60250
    wf[NeXusName[FrameMonitor1]] = 'monitor_1'
    da = wf.compute(RawMonitor[SampleRun, FrameMonitor1])
    assert 'position' in da.coords
    assert 'source_position' in da.coords
    assert da.bins is not None
    assert da.dims == ('event_time_zero',)
    assert da.bins.constituents['data'].variances is not None


def test_load_histogram_monitor_workflow(dream_coda_test_file: Path) -> None:
    wf = LoadMonitorWorkflow(run_types=[SampleRun], monitor_types=[FrameMonitor1])
    wf[Filename[SampleRun]] = dream_coda_test_file
    wf[NeXusName[FrameMonitor1]] = 'monitor_bunker'
    da = wf.compute(RawMonitor[SampleRun, FrameMonitor1])
    assert 'position' in da.coords
    assert 'source_position' in da.coords
    assert da.bins is None
    assert set(da.dims) == {'time', 'frame_time'}
    assert 'time' in da.coords.keys()
    assert 'frame_time' in da.coords.keys()
    assert da.variances is not None


def test_load_detector_workflow(loki_tutorial_sample_run_60250: Path) -> None:
    wf = LoadDetectorWorkflow(run_types=[SampleRun])
    wf[Filename[SampleRun]] = loki_tutorial_sample_run_60250
    wf[NeXusName[snx.NXdetector]] = 'larmor_detector'
    da = wf.compute(RawDetector[SampleRun])
    assert 'position' in da.coords
    assert da.bins is not None
    assert da.dims == ('detector_number',)


def test_load_histogram_detector_workflow(tbl_commissioning_orca_file: Path) -> None:
    wf = LoadDetectorWorkflow(run_types=[SampleRun])
    wf[Filename[SampleRun]] = tbl_commissioning_orca_file
    wf[NeXusName[snx.NXdetector]] = 'orca_detector'
    da = wf.compute(RawDetector[SampleRun])
    assert 'position' in da.coords
    assert da.bins is None
    assert 'time' in da.dims
    assert da.ndim == 3


def test_load_empty_histogram_detector_workflow(
    tbl_commissioning_orca_file: Path,
) -> None:
    wf = LoadDetectorWorkflow(run_types=[SampleRun])
    wf[Filename[SampleRun]] = tbl_commissioning_orca_file
    wf[NeXusName[snx.NXdetector]] = 'orca_detector'
    da = wf.compute(EmptyDetector[SampleRun])
    assert 'position' in da.coords
    assert da.bins is None
    # The empty detector has no time dimension, only the dimensions of the geometry
    assert 'time' not in da.dims
    assert da.ndim == 2


@pytest.mark.parametrize('preopen', [True, False])
def test_generic_nexus_workflow(
    preopen: bool, loki_tutorial_sample_run_60250: Path
) -> None:
    wf = GenericNeXusWorkflow(run_types=[SampleRun], monitor_types=[FrameMonitor1])
    wf[Filename[SampleRun]] = loki_tutorial_sample_run_60250
    wf[NeXusName[FrameMonitor1]] = 'monitor_1'
    wf[NeXusName[snx.NXdetector]] = 'larmor_detector'
    wf[PreopenNeXusFile] = preopen
    da = wf.compute(RawDetector[SampleRun])
    assert 'position' in da.coords
    assert da.bins is not None
    assert da.dims == ('detector_number',)
    da = wf.compute(RawMonitor[SampleRun, FrameMonitor1])
    assert 'position' in da.coords
    assert 'source_position' in da.coords
    assert da.bins is not None
    assert da.dims == ('event_time_zero',)


def test_generic_nexus_workflow_load_choppers(bifrost_simulated_elastic: Path) -> None:
    wf = GenericNeXusWorkflow(run_types=[SampleRun], monitor_types=[])
    wf[Filename[SampleRun]] = bifrost_simulated_elastic
    choppers = wf.compute(RawChoppers[SampleRun])

    assert choppers.keys() == {
        '005_PulseShapingChopper',
        '006_PulseShapingChopper2',
        '019_FOC1',
        '048_FOC2',
        '095_BWC1',
        '096_BWC2',
    }
    chopper = choppers['005_PulseShapingChopper']
    assert 'position' in chopper
    assert 'rotation_speed' in chopper
    assert chopper['slit_edges'].shape == (2,)


def test_generic_nexus_workflow_load_beamline_metadata(
    bifrost_simulated_elastic: Path,
) -> None:
    wf = GenericNeXusWorkflow(run_types=[SampleRun], monitor_types=[])
    wf[Filename[SampleRun]] = bifrost_simulated_elastic
    beamline = wf.compute(Beamline[SampleRun])

    assert beamline.name == 'BIFROST'
    assert beamline.facility == 'ESS'
    assert beamline.site == 'ESS'


def test_generic_nexus_workflow_load_measurement_metadata(
    loki_tutorial_sample_run_60250: Path, loki_tutorial_background_run_60248: Path
) -> None:
    wf = GenericNeXusWorkflow(run_types=[SampleRun], monitor_types=[])
    wf[Filename[SampleRun]] = loki_tutorial_sample_run_60250
    wf[Filename[BackgroundRun]] = loki_tutorial_background_run_60248
    measurement = wf.compute(Measurement[SampleRun])

    assert measurement.title == 'My experiment'
    assert measurement.experiment_id == 'p1234'
    assert measurement.start_time == datetime(2022, 2, 28, 21, 15, 0, tzinfo=UTC)
    assert measurement.end_time == datetime(2032, 2, 29, 9, 15, 0, tzinfo=UTC)
    assert measurement.run_number is None
    assert measurement.experiment_doi is None


def test_generic_nexus_workflow_includes_only_given_run_and_monitor_types() -> None:
    wf = GenericNeXusWorkflow(
        run_types=[SampleRun], monitor_types=[FrameMonitor1, FrameMonitor0]
    )
    graph = wf.underlying_graph

    # Check some examples to avoid relying entirely on complicated loops below.
    assert RawDetector[SampleRun] in graph
    assert RawDetector[BackgroundRun] not in graph
    assert RawMonitor[SampleRun, FrameMonitor1] in graph
    assert RawMonitor[SampleRun, FrameMonitor2] not in graph
    assert RawMonitor[SampleRun, FrameMonitor0] in graph
    assert RawMonitor[BackgroundRun, FrameMonitor0] not in graph
    assert RawMonitor[BackgroundRun, FrameMonitor1] not in graph
    assert RawMonitor[BackgroundRun, FrameMonitor2] not in graph
    assert RawChoppers[SampleRun] in graph
    assert RawChoppers[BackgroundRun] not in graph

    assert NeXusComponentLocationSpec[FrameMonitor0, SampleRun] in graph
    assert NeXusComponentLocationSpec[FrameMonitor1, SampleRun] in graph
    assert NeXusComponentLocationSpec[FrameMonitor2, SampleRun] not in graph
    assert NeXusComponentLocationSpec[snx.NXdetector, SampleRun] in graph
    assert NeXusComponentLocationSpec[snx.NXsample, SampleRun] in graph
    assert NeXusComponentLocationSpec[snx.NXsource, SampleRun] in graph
    assert NeXusComponentLocationSpec[FrameMonitor0, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[FrameMonitor1, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[FrameMonitor2, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[snx.NXdetector, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[snx.NXsample, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[snx.NXsource, BackgroundRun] not in graph

    excluded_run_types = set(RunType.__constraints__) - {SampleRun}
    excluded_monitor_types = set(MonitorType.__constraints__) - {
        FrameMonitor1,
        FrameMonitor0,
    }
    for node in graph:
        assert_not_contains_type_arg(node, excluded_run_types)
        assert_not_contains_type_arg(node, excluded_monitor_types)


def test_generic_nexus_workflow_includes_only_given_run_types() -> None:
    wf = GenericNeXusWorkflow(
        run_types=[EmptyBeamRun],
        monitor_types=[FrameMonitor0, FrameMonitor1, FrameMonitor2],
    )
    graph = wf.underlying_graph

    # Check some examples to avoid relying entirely on complicated loops below.
    assert RawDetector[EmptyBeamRun] in graph
    assert RawDetector[SampleRun] not in graph
    assert RawMonitor[EmptyBeamRun, FrameMonitor1] in graph
    assert RawMonitor[EmptyBeamRun, FrameMonitor2] in graph
    assert RawMonitor[EmptyBeamRun, FrameMonitor0] in graph
    assert RawMonitor[SampleRun, FrameMonitor1] not in graph
    assert RawMonitor[SampleRun, FrameMonitor2] not in graph
    assert RawMonitor[SampleRun, FrameMonitor0] not in graph
    assert RawChoppers[EmptyBeamRun] in graph
    assert RawChoppers[SampleRun] not in graph

    excluded_run_types = set(RunType.__constraints__) - {EmptyBeamRun}
    for node in graph:
        assert_not_contains_type_arg(node, excluded_run_types)


def test_generic_nexus_workflow_includes_only_given_monitor_types() -> None:
    wf = GenericNeXusWorkflow(
        run_types=[SampleRun, BackgroundRun],
        monitor_types=[TransmissionMonitor, FrameMonitor1],
    )
    graph = wf.underlying_graph

    # Check some examples to avoid relying entirely on complicated loops below.
    assert RawDetector[SampleRun] in graph
    assert RawDetector[BackgroundRun] in graph
    assert RawMonitor[SampleRun, TransmissionMonitor] in graph
    assert RawMonitor[SampleRun, FrameMonitor1] in graph
    assert RawMonitor[SampleRun, FrameMonitor2] not in graph
    assert RawMonitor[SampleRun, FrameMonitor0] not in graph
    assert RawMonitor[BackgroundRun, TransmissionMonitor] in graph
    assert RawMonitor[BackgroundRun, FrameMonitor1] in graph
    assert RawMonitor[BackgroundRun, FrameMonitor2] not in graph
    assert RawMonitor[BackgroundRun, FrameMonitor0] not in graph
    assert RawChoppers[SampleRun] in graph
    assert RawChoppers[BackgroundRun] in graph

    excluded_monitor_types = set(MonitorType.__constraints__) - {
        FrameMonitor1,
        TransmissionMonitor,
    }
    for node in graph:
        assert_not_contains_type_arg(node, excluded_monitor_types)


def assert_not_contains_type_arg(node: object, excluded: set[type]) -> None:
    assert not any(
        arg in excluded for arg in getattr(node, "__args__", ())
    ), f"Node {node} contains one of {excluded!r}"


def test_generic_nexus_workflow_load_custom_field_user_affiliation(
    loki_tutorial_sample_run_60250: Path,
) -> None:
    class UserAffiliation(sl.Scope[RunType, str], str):
        """User affiliation."""

    def load_user_affiliation(
        file: NeXusFileSpec[RunType], path: NeXusName[UserAffiliation[RunType]]
    ) -> UserAffiliation[RunType]:
        return UserAffiliation[RunType](
            load_from_path(NeXusLocationSpec(filename=file.value, component_name=path))
        )

    wf = GenericNeXusWorkflow(run_types=[SampleRun], monitor_types=[])
    wf.insert(load_user_affiliation)
    wf[Filename[SampleRun]] = loki_tutorial_sample_run_60250
    # Path is relative to the top-level '/entry'
    wf[NeXusName[UserAffiliation[SampleRun]]] = 'user_0/affiliation'
    affiliation = wf.compute(UserAffiliation[SampleRun])
    assert affiliation == 'ESS'


def test_generic_nexus_workflow_load_custom_group_user(
    loki_tutorial_sample_run_60250: Path,
) -> None:
    class UserInfo(sl.Scope[RunType, str], str):
        """User info."""

    def load_user_info(
        file: NeXusFileSpec[RunType], path: NeXusName[UserInfo[RunType]]
    ) -> UserInfo[RunType]:
        return UserInfo[RunType](
            load_from_path(NeXusLocationSpec(filename=file.value, component_name=path))
        )

    wf = GenericNeXusWorkflow(run_types=[SampleRun], monitor_types=[])
    wf.insert(load_user_info)
    wf[Filename[SampleRun]] = loki_tutorial_sample_run_60250
    # Path is relative to the top-level '/entry'
    wf[NeXusName[UserInfo]] = 'user_0'
    user_info = wf.compute(UserInfo[SampleRun])
    assert user_info['affiliation'] == 'ESS'
    assert user_info['name'] == 'John Doe'
