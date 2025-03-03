# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from datetime import datetime, timezone

import pytest
import scipp as sc
import scippnexus as snx
from scipp.testing import assert_identical

from ess.reduce import data
from ess.reduce.nexus import compute_component_position, workflow
from ess.reduce.nexus.types import (
    Analyzers,
    BackgroundRun,
    Beamline,
    Choppers,
    DetectorData,
    EmptyBeamRun,
    Filename,
    Measurement,
    Monitor1,
    Monitor2,
    Monitor3,
    MonitorData,
    MonitorType,
    NeXusComponentLocationSpec,
    NeXusName,
    NeXusTransformation,
    PreopenNeXusFile,
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


@pytest.fixture
def calibrated_monitor() -> workflow.CalibratedMonitor[SampleRun, Monitor1]:
    return workflow.CalibratedMonitor[SampleRun, Monitor1](
        sc.DataArray(
            sc.scalar(0),
            coords={'position': sc.vector([1.0, 2.0, 3.0], unit='m')},
        )
    )


@pytest.fixture
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


@pytest.mark.parametrize('preopen', [True, False])
def test_generic_nexus_workflow(preopen: bool) -> None:
    wf = GenericNeXusWorkflow()
    wf[Filename[SampleRun]] = data.loki_tutorial_sample_run_60250()
    wf[NeXusName[Monitor1]] = 'monitor_1'
    wf[NeXusName[snx.NXdetector]] = 'larmor_detector'
    wf[PreopenNeXusFile] = preopen
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


def test_generic_nexus_workflow_load_choppers() -> None:
    wf = GenericNeXusWorkflow()
    wf[Filename[SampleRun]] = data.bifrost_simulated_elastic()
    choppers = wf.compute(Choppers[SampleRun])

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


def test_generic_nexus_workflow_load_analyzers() -> None:
    wf = GenericNeXusWorkflow()
    wf[Filename[SampleRun]] = data.bifrost_simulated_elastic()
    analyzers = wf.compute(Analyzers[SampleRun])

    assert len(analyzers) == 45
    analyzer = analyzers['144_channel_2_1_monochromator']
    assert 'position' in analyzer
    assert analyzer['d_spacing'].ndim == 0
    assert analyzer['usage'] == 'Bragg'


def test_generic_nexus_workflow_load_beamline_metadata() -> None:
    wf = GenericNeXusWorkflow()
    wf[Filename[SampleRun]] = data.bifrost_simulated_elastic()
    beamline = wf.compute(Beamline)

    assert beamline.name == 'BIFROST'
    assert beamline.facility == 'ESS'
    assert beamline.site == 'ESS'


def test_generic_nexus_workflow_load_measurement_metadata() -> None:
    wf = GenericNeXusWorkflow()
    wf[Filename[SampleRun]] = data.loki_tutorial_sample_run_60250()
    wf[Filename[BackgroundRun]] = data.loki_tutorial_background_run_60248()
    measurement = wf.compute(Measurement)

    assert measurement.title == 'My experiment'
    assert measurement.experiment_id == 'p1234'
    assert measurement.start_time == datetime(
        2022, 2, 28, 21, 15, 0, tzinfo=timezone.utc
    )
    assert measurement.end_time == datetime(2032, 2, 29, 9, 15, 0, tzinfo=timezone.utc)
    assert measurement.run_number is None
    assert measurement.experiment_doi is None


def test_generic_nexus_workflow_includes_only_given_run_and_monitor_types() -> None:
    wf = GenericNeXusWorkflow(run_types=[SampleRun], monitor_types=[Monitor1, Monitor3])
    graph = wf.underlying_graph

    # Check some examples to avoid relying entirely on complicated loops below.
    assert DetectorData[SampleRun] in graph
    assert DetectorData[BackgroundRun] not in graph
    assert MonitorData[SampleRun, Monitor1] in graph
    assert MonitorData[SampleRun, Monitor2] not in graph
    assert MonitorData[SampleRun, Monitor3] in graph
    assert MonitorData[BackgroundRun, Monitor1] not in graph
    assert MonitorData[BackgroundRun, Monitor2] not in graph
    assert MonitorData[BackgroundRun, Monitor3] not in graph
    assert Choppers[SampleRun] in graph
    assert Choppers[BackgroundRun] not in graph
    assert Analyzers[SampleRun] in graph
    assert Analyzers[BackgroundRun] not in graph

    assert NeXusComponentLocationSpec[Monitor1, SampleRun] in graph
    assert NeXusComponentLocationSpec[Monitor2, SampleRun] not in graph
    assert NeXusComponentLocationSpec[Monitor3, SampleRun] in graph
    assert NeXusComponentLocationSpec[snx.NXdetector, SampleRun] in graph
    assert NeXusComponentLocationSpec[snx.NXsample, SampleRun] in graph
    assert NeXusComponentLocationSpec[snx.NXsource, SampleRun] in graph
    assert NeXusComponentLocationSpec[Monitor1, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[Monitor2, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[Monitor3, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[snx.NXdetector, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[snx.NXsample, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[snx.NXsource, BackgroundRun] not in graph

    excluded_run_types = set(RunType.__constraints__) - {SampleRun}
    excluded_monitor_types = set(MonitorType.__constraints__) - {Monitor1, Monitor3}
    for node in graph:
        assert_not_contains_type_arg(node, excluded_run_types)
        assert_not_contains_type_arg(node, excluded_monitor_types)


def test_generic_nexus_workflow_includes_only_given_run_types() -> None:
    wf = GenericNeXusWorkflow(run_types=[EmptyBeamRun])
    graph = wf.underlying_graph

    # Check some examples to avoid relying entirely on complicated loops below.
    assert DetectorData[EmptyBeamRun] in graph
    assert DetectorData[SampleRun] not in graph
    assert MonitorData[EmptyBeamRun, Monitor1] in graph
    assert MonitorData[EmptyBeamRun, Monitor2] in graph
    assert MonitorData[EmptyBeamRun, Monitor3] in graph
    assert MonitorData[SampleRun, Monitor1] not in graph
    assert MonitorData[SampleRun, Monitor2] not in graph
    assert MonitorData[SampleRun, Monitor3] not in graph
    assert Choppers[EmptyBeamRun] in graph
    assert Choppers[SampleRun] not in graph
    assert Analyzers[EmptyBeamRun] in graph
    assert Analyzers[SampleRun] not in graph

    excluded_run_types = set(RunType.__constraints__) - {EmptyBeamRun}
    for node in graph:
        assert_not_contains_type_arg(node, excluded_run_types)


def test_generic_nexus_workflow_includes_only_given_monitor_types() -> None:
    wf = GenericNeXusWorkflow(monitor_types=[TransmissionMonitor, Monitor1])
    graph = wf.underlying_graph

    # Check some examples to avoid relying entirely on complicated loops below.
    assert DetectorData[SampleRun] in graph
    assert DetectorData[BackgroundRun] in graph
    assert MonitorData[SampleRun, TransmissionMonitor] in graph
    assert MonitorData[SampleRun, Monitor1] in graph
    assert MonitorData[SampleRun, Monitor2] not in graph
    assert MonitorData[SampleRun, Monitor3] not in graph
    assert MonitorData[BackgroundRun, TransmissionMonitor] in graph
    assert MonitorData[BackgroundRun, Monitor1] in graph
    assert MonitorData[BackgroundRun, Monitor2] not in graph
    assert MonitorData[BackgroundRun, Monitor3] not in graph
    assert Choppers[SampleRun] in graph
    assert Choppers[BackgroundRun] in graph
    assert Analyzers[SampleRun] in graph
    assert Analyzers[BackgroundRun] in graph

    excluded_monitor_types = set(MonitorType.__constraints__) - {
        Monitor1,
        TransmissionMonitor,
    }
    for node in graph:
        assert_not_contains_type_arg(node, excluded_monitor_types)


def assert_not_contains_type_arg(node: object, excluded: set[type]) -> None:
    assert not any(
        arg in excluded for arg in getattr(node, "__args__", ())
    ), f"Node {node} contains one of {excluded!r}"
