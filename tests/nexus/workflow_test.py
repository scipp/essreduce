# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
from scipp.testing import assert_identical

from ess.reduce.nexus import workflow


@pytest.fixture(params=[{}, {'aux': 1}])
def group_with_no_position(request) -> workflow.AnyRunNeXusSample:
    return workflow.AnyRunNeXusSample(sc.DataGroup(request.param))


def test_sample_position_returns_position_of_group() -> None:
    position = sc.vector([1.0, 2.0, 3.0], unit='m')
    sample_group = workflow.AnyRunNeXusSample(sc.DataGroup(position=position))
    assert_identical(workflow.get_sample_position(sample_group), position)


def test_get_sample_position_returns_origin_if_position_not_found(
    group_with_no_position,
) -> None:
    assert_identical(
        workflow.get_sample_position(group_with_no_position), workflow.origin
    )


def test_get_source_position_returns_position_of_group() -> None:
    position = sc.vector([1.0, 2.0, 3.0], unit='m')
    source_group = workflow.AnyRunNeXusSource(sc.DataGroup(position=position))
    assert_identical(workflow.get_source_position(source_group), position)


def test_get_source_position_raises_exception_if_position_not_found(
    group_with_no_position,
) -> None:
    with pytest.raises(KeyError, match='position'):
        workflow.get_source_position(group_with_no_position)


@pytest.fixture()
def nexus_detector() -> workflow.AnyRunNeXusDetector:
    detector_number = sc.arange('detector_number', 6, unit=None)
    x = sc.linspace('detector_number', 0, 1, num=6, unit='m')
    position = sc.spatial.as_vectors(x, sc.zeros_like(x), sc.zeros_like(x))
    data = sc.DataArray(
        sc.empty_like(detector_number),
        coords={
            'detector_number': detector_number,
            'position': position,
        },
    )
    return workflow.AnyRunNeXusDetector(
        sc.DataGroup(
            data=data,
            # Note that this position (the overall detector position) will be ignored,
            # only the pixel positions (typically defined relative to this in an
            # actual NeXus file) will be used.
            position=sc.vector([1.0, 2.0, 3.0], unit='m'),
            nexus_component_name='detector1',
        )
    )


@pytest.fixture()
def source_position() -> sc.Variable:
    return sc.vector([0.0, 0.0, -10.0], unit='m')


def test_get_calibrated_detector_extracts_data_field_from_nexus_detector(
    nexus_detector,
    source_position,
) -> None:
    detector = workflow.get_calibrated_detector(
        nexus_detector,
        offset=workflow.no_offset,
        source_position=source_position,
        sample_position=workflow.origin,
        gravity=workflow.gravity_vector_neg_y(),
        bank_sizes={},
    )
    assert_identical(
        detector.drop_coords(('sample_position', 'source_position', 'gravity')),
        nexus_detector['data'],
    )


def test_get_calibrated_detector_folds_detector_number_if_mapping_given(
    nexus_detector,
    source_position,
) -> None:
    sizes = {'xpixel': 2, 'ypixel': 3}
    bank_sizes = {'detector1': sizes}
    detector = workflow.get_calibrated_detector(
        nexus_detector,
        offset=workflow.no_offset,
        source_position=source_position,
        sample_position=workflow.origin,
        gravity=workflow.gravity_vector_neg_y(),
        bank_sizes=bank_sizes,
    )
    assert detector.sizes == sizes


def test_get_calibrated_detector_works_if_nexus_component_name_is_missing(
    nexus_detector, source_position
):
    del nexus_detector['nexus_component_name']
    detector = workflow.get_calibrated_detector(
        nexus_detector,
        offset=workflow.no_offset,
        source_position=source_position,
        sample_position=workflow.origin,
        gravity=workflow.gravity_vector_neg_y(),
        bank_sizes={},
    )
    assert detector.sizes == nexus_detector['data'].sizes


def test_get_calibrated_detector_adds_offset_to_position(
    nexus_detector,
    source_position,
) -> None:
    offset = sc.vector([0.1, 0.2, 0.3], unit='m')
    detector = workflow.get_calibrated_detector(
        nexus_detector,
        offset=offset,
        source_position=source_position,
        sample_position=workflow.origin,
        gravity=workflow.gravity_vector_neg_y(),
        bank_sizes={},
    )
    position = nexus_detector['data'].coords['position'] + offset
    assert detector.coords['position'].sizes == {'detector_number': 6}
    assert_identical(detector.coords['position'], position)


def test_get_calibrated_detector_forwards_coords(
    nexus_detector,
    source_position,
) -> None:
    nexus_detector['data'].coords['abc'] = sc.scalar(1.2)
    detector = workflow.get_calibrated_detector(
        nexus_detector,
        offset=workflow.no_offset,
        source_position=source_position,
        sample_position=workflow.origin,
        gravity=workflow.gravity_vector_neg_y(),
        bank_sizes={},
    )
    assert 'abc' in detector.coords


def test_get_calibrated_detector_forwards_masks(
    nexus_detector,
    source_position,
) -> None:
    nexus_detector['data'].masks['mymask'] = sc.scalar(False)
    detector = workflow.get_calibrated_detector(
        nexus_detector,
        offset=workflow.no_offset,
        source_position=source_position,
        sample_position=workflow.origin,
        gravity=workflow.gravity_vector_neg_y(),
        bank_sizes={},
    )
    assert 'mymask' in detector.masks


@pytest.fixture()
def calibrated_detector() -> workflow.AnyRunCalibratedDetector:
    detector_number = sc.arange('detector_number', 6, unit=None)
    return workflow.AnyRunCalibratedDetector(
        sc.DataArray(
            sc.empty_like(detector_number),
            coords={
                'position': sc.vector([1.0, 2.0, 3.0], unit='m'),
                'detector_number': detector_number,
            },
        ).fold('detector_number', sizes={'xpixel': 2, 'ypixel': 3})
    )


@pytest.fixture()
def detector_event_data() -> workflow.AnyRunNeXusDetectorEventData:
    content = sc.DataArray(
        sc.ones(dims=['event'], shape=[17], unit='counts'),
        coords={'event_id': sc.arange('event', 17, unit=None) % sc.index(6)},
    )
    weights = sc.bins(data=content, dim='event')
    return workflow.AnyRunNeXusDetectorEventData(
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
def nexus_monitor() -> workflow.AnyRunAnyNeXusMonitor:
    data = sc.DataArray(sc.scalar(1.2), coords={'something': sc.scalar(13)})
    return workflow.AnyRunAnyNeXusMonitor(
        sc.DataGroup(data=data, position=sc.vector([1.0, 2.0, 3.0], unit='m'))
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
        monitor.drop_coords(('position', 'source_position')), nexus_monitor['data']
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
def calibrated_monitor() -> workflow.AnyRunAnyCalibratedMonitor:
    return workflow.AnyRunAnyCalibratedMonitor(
        sc.DataArray(
            sc.scalar(0),
            coords={'position': sc.vector([1.0, 2.0, 3.0], unit='m')},
        )
    )


@pytest.fixture()
def monitor_event_data() -> workflow.AnyRunAnyNeXusMonitorEventData:
    content = sc.DataArray(sc.ones(dims=['event'], shape=[17], unit='counts'))
    weights = sc.bins(data=content, dim='event')
    return workflow.AnyRunAnyNeXusMonitorEventData(
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
