# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import h5py
import pytest
import scipp as sc
import scipp.testing
import scippnexus as snx

from ess.reduce import nexus


@pytest.fixture()
def nxroot():
    """Yield NXroot containing a single NXentry named 'entry'"""
    with h5py.File('dummy.nxs', mode='w', driver="core", backing_store=False) as f:
        root = snx.Group(f, definitions=snx.base_definitions())
        root.create_class('entry', snx.NXentry)
        yield root


@pytest.fixture()
def nexus_group(nxroot):
    instrument = nxroot['entry'].create_class('reducer', snx.NXinstrument)
    detector = instrument.create_class('bank12', snx.NXdetector)

    events = detector.create_class('bank12_events', snx.NXevent_data)
    events['event_id'] = sc.array(dims=[''], unit=None, values=[1, 2, 4, 1, 2, 2])
    events['event_time_offset'] = sc.array(
        dims=[''], unit='s', values=[456, 7, 3, 345, 632, 23]
    )
    events['event_time_zero'] = sc.array(dims=[''], unit='s', values=[1, 2, 3, 4])
    events['event_index'] = sc.array(dims=[''], unit=None, values=[0, 3, 3, -1000])

    return nxroot


# TODO histogram data
# TODO test with real file + BytesIO + snx.Group


@pytest.mark.parametrize('detector_name', (None, nexus.DetectorName('bank12')))
@pytest.mark.parametrize('instrument_name', (None, nexus.InstrumentName('reducer')))
def test_load_detector_from_group(nexus_group, instrument_name, detector_name):
    detector = nexus.load_detector(
        nexus.NeXusGroup(nexus_group),
        instrument_name=instrument_name,
        detector_name=detector_name,
    )
    expected = nexus_group['entry/reducer/bank12/bank12_events'][...]
    # TODO positions
    sc.testing.assert_identical(detector, expected)