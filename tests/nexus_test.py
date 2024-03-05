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


def _write_nexus_events(store: Union[Path, BytesIO]) -> None:
    with snx.File(store, 'w') as root:
        entry = root.create_class('entry', snx.NXentry)
        instrument = entry.create_class('reducer', snx.NXinstrument)
        detector = instrument.create_class('bank12', snx.NXdetector)

        events = detector.create_class('bank12_events', snx.NXevent_data)
        for key, val in _event_data_components().items():
            events[key] = val


def _file_store(typ):
    if typ == Path:
        return Path('nexus_test', uuid.uuid4().hex, 'testfile.nxs')
    return BytesIO()


@pytest.fixture(params=[Path, BytesIO, snx.Group])
def nexus_file(fs, request):
    store = _file_store(request.param)
    _write_nexus_events(store)
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


# TODO histogram data


@pytest.mark.parametrize('instrument_name', (None, nexus.InstrumentName('reducer')))
def test_load_detector(nexus_file, expected_bank12, instrument_name):
    detector = nexus.load_detector(
        nexus.NeXusGroup(nexus_file),
        detector_name=nexus.DetectorName('bank12'),
        instrument_name=instrument_name,
    )
    # TODO positions
    sc.testing.assert_identical(detector['data'], expected_bank12)
