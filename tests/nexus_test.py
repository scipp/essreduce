# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import uuid
from io import BytesIO
from pathlib import Path
from typing import Union

import pytest
import scipp as sc
import scipp.testing
import scippnexus as snx

from ess.reduce import nexus


def _event_data_components() -> sc.DataGroup:
    return sc.DataGroup(
        {
            'event_id': sc.array(dims=['event'], unit=None, values=[1, 2, 4, 1, 2, 2]),
            'event_time_offset': sc.array(
                dims=['event'], unit='s', values=[456, 7, 3, 345, 632, 23]
            ),
            'event_time_zero': sc.array(
                dims=['event_time_zero'], unit='s', values=[1, 2, 3, 4]
            ),
            'event_index': sc.array(
                dims=['event_time_zero'], unit=None, values=[0, 3, 3, 6]
            ),
        }
    )


def _monitor_histogram() -> sc.DataArray:
    return sc.DataArray(
        sc.array(dims=['time'], values=[2, 4, 8, 3], unit='counts'),
        coords={
            'time': sc.epoch(unit='ms')
            + sc.array(dims=['time'], values=[2, 4, 6, 8, 10], unit='ms')
        },
    )


def _write_nexus_data(store: Union[Path, BytesIO]) -> None:
    with snx.File(store, 'w') as root:
        entry = root.create_class('entry', snx.NXentry)
        instrument = entry.create_class('reducer', snx.NXinstrument)

        detector = instrument.create_class('bank12', snx.NXdetector)
        events = detector.create_class('bank12_events', snx.NXevent_data)
        for key, val in _event_data_components().items():
            events[key] = val

        monitor_data = _monitor_histogram()
        monitor = instrument.create_class('monitor', snx.NXmonitor)
        data = monitor.create_class('data', snx.NXdata)
        signal = data.create_field('signal', monitor_data.data)
        signal.attrs['signal'] = 1
        signal.attrs['axes'] = monitor_data.dim
        data.create_field('time', monitor_data.coords['time'])


# TODO more fields
"""
h5ls 60248-2022-02-28_2215.nxs/entry/instrument/larmor_detector       15:27:23
depends_on               Dataset {1}
detector_number          Dataset {458752}
larmor_detector_events   Group
pixel_shape              Group
transformations          Group
x_pixel_offset           Dataset {458752}
y_pixel_offset           Dataset {458752}
z_pixel_offset           Dataset {458752}
"""


def _file_store(typ):
    if typ == Path:
        return Path('nexus_test', uuid.uuid4().hex, 'testfile.nxs')
    return BytesIO()


@pytest.fixture(params=[Path, BytesIO, snx.Group])
def nexus_file(fs, request):
    store = _file_store(request.param)
    _write_nexus_data(store)
    if isinstance(store, BytesIO):
        store.seek(0)

    if request.param in (Path, BytesIO):
        yield store
    else:
        with snx.File(store, 'r') as f:
            yield f


@pytest.fixture()
def expected_bank12():
    components = _event_data_components()
    buffer = sc.DataArray(
        sc.ones(sizes={'event': 6}, unit='counts'),
        coords={
            'event_id': components['event_id'],
            'event_time_offset': components['event_time_offset'],
        },
    )
    events = sc.bins(
        data=buffer,
        begin=components['event_index'],
        end=sc.concat(
            [components['event_index'][1:], components['event_index'][-1]],
            dim='event_time_zero',
        ),
        dim='event',
    )
    return sc.DataArray(
        events, coords={'event_time_zero': components['event_time_zero']}
    )


@pytest.fixture()
def expected_monitor() -> sc.DataArray:
    return _monitor_histogram()


@pytest.mark.parametrize('instrument_name', (None, nexus.InstrumentName('reducer')))
def test_load_detector(nexus_file, expected_bank12, instrument_name):
    detector = nexus.load_detector(
        nexus.NeXusGroup(nexus_file),
        detector_name=nexus.DetectorName('bank12'),
        instrument_name=instrument_name,
    )
    sc.testing.assert_identical(detector['bank12_events'], expected_bank12)


@pytest.mark.parametrize('instrument_name', (None, nexus.InstrumentName('reducer')))
def test_load_monitor(nexus_file, expected_monitor, instrument_name):
    monitor = nexus.load_monitor(
        nexus.NeXusGroup(nexus_file),
        monitor_name=nexus.MonitorName('monitor'),
        instrument_name=instrument_name,
    )
    sc.testing.assert_identical(monitor['data'], expected_monitor)
