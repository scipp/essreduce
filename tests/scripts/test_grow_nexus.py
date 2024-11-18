import os
import tempfile

import h5py
import numpy as np
import pytest

from ess.reduce.scripts.grow_nexus import grow_nexus_file


@pytest.fixture
def nexus_file():
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, 'test.nxs')
        with h5py.File(path, 'a') as hf:
            entry = hf.create_group('entry')
            entry.attrs['NX_class'] = 'NXentry'

            instrument = entry.create_group('instrument')
            instrument.attrs['NX_class'] = 'NXinstrument'

            for group, nxclass in (
                ('detector', 'NXdetector'),
                ('monitor', 'NXmonitor'),
            ):
                detector = instrument.create_group(group)
                detector.attrs['NX_class'] = nxclass

                event_data = detector.create_group('event_data')
                event_data.attrs['NX_class'] = 'NXevent_data'

                event_data.create_dataset(
                    'event_index',
                    data=np.array([2, 4, 6]),
                    maxshape=(None,),
                    chunks=True,
                )
                event_data.create_dataset(
                    'event_time_zero',
                    data=np.array([0, 1, 2]),
                    maxshape=(None,),
                    chunks=True,
                )
                event_data.create_dataset(
                    'event_id',
                    data=np.array([0, 1, 2, 0, 1, 2]),
                    maxshape=(None,),
                    chunks=True,
                )
                event_data.create_dataset(
                    'event_time_offset',
                    data=np.array([1, 2, 1, 2, 1, 2]),
                    maxshape=(None,),
                    chunks=True,
                )

            yield path


@pytest.mark.parametrize('monitor_scale', [1, 2, None])
@pytest.mark.parametrize('detector_scale', [1, 2])
def test_grow_nexus(nexus_file, detector_scale, monitor_scale):
    grow_nexus_file(
        filename=nexus_file, detector_scale=detector_scale, monitor_scale=monitor_scale
    )

    monitor_scale = monitor_scale if monitor_scale is not None else detector_scale

    with h5py.File(nexus_file, 'r') as f:
        for detector, scale in zip(
            ('detector', 'monitor'), (detector_scale, monitor_scale), strict=True
        ):
            np.testing.assert_equal(
                [scale * i for i in [2, 4, 6]],
                f[f'entry/instrument/{detector}/event_data/event_index'][()],
            )
            np.testing.assert_equal(
                scale * [0, 1, 2, 0, 1, 2],
                f[f'entry/instrument/{detector}/event_data/event_id'][()],
            )
            np.testing.assert_equal(
                scale * [1, 2, 1, 2, 1, 2],
                f[f'entry/instrument/{detector}/event_data/event_time_offset'][()],
            )
