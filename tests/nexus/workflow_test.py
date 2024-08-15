# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
from ess.reduce.nexus import workflow
from scipp.testing import assert_identical


@pytest.fixture(params=[{}, {'aux': 1}])
def group_with_no_position(request) -> workflow.NeXusSample:
    return workflow.NeXusSample(sc.DataGroup(request.param))


def test_sample_position_returns_position_of_group() -> None:
    position = sc.vector([1.0, 2.0, 3.0], unit='m')
    sample_group = workflow.NeXusSample(sc.DataGroup(position=position))
    assert_identical(workflow.get_sample_position(sample_group), position)


def test_get_sample_position_returns_origin_if_position_not_found(
    group_with_no_position,
) -> None:
    assert_identical(
        workflow.get_sample_position(group_with_no_position), workflow.origin
    )


def test_get_source_position_returns_position_of_group() -> None:
    position = sc.vector([1.0, 2.0, 3.0], unit='m')
    source_group = workflow.NeXusSource(sc.DataGroup(position=position))
    assert_identical(workflow.get_source_position(source_group), position)


def test_get_source_position_raises_exception_if_position_not_found(
    group_with_no_position,
) -> None:
    with pytest.raises(KeyError, match='position'):
        workflow.get_source_position(group_with_no_position)


@pytest.fixture()
def nexus_monitor() -> workflow.NeXusMonitor:
    data = sc.DataArray(sc.scalar(1.2), coords={'something': sc.scalar(13)})
    return workflow.NeXusMonitor(
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
    assert_identical(monitor.coords['position'], sc.vector([0.9, 1.8, 2.7], unit='m'))


@pytest.fixture()
def calibrated_monitor() -> workflow.CalibratedMonitor:
    return workflow.CalibratedMonitor(
        sc.DataArray(
            sc.scalar(0),
            coords={'position': sc.vector([1.0, 2.0, 3.0], unit='m')},
        )
    )


@pytest.fixture()
def monitor_event_data() -> workflow.NeXusMonitorEventData:
    content = sc.DataArray(sc.ones(dims=['event'], shape=[17], unit='counts'))
    weights = sc.bins(data=content, dim='event')
    return workflow.NeXusMonitorEventData(
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