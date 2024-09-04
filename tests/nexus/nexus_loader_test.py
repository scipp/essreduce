# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
import scipp as sc
import scipp.testing
import scippnexus as snx

from ess.reduce import nexus

year_zero = sc.datetime('1970-01-01T00:00:00')


def _event_data_components() -> sc.DataGroup:
    return sc.DataGroup(
        {
            'event_id': sc.array(dims=['event'], unit=None, values=[1, 2, 4, 1, 2, 2]),
            'event_time_offset': sc.array(
                dims=['event'], unit='s', values=[456, 7, 3, 345, 632, 23]
            ),
            'event_time_zero': sc.epoch(unit='s')
            + sc.array(dims=['event_index'], unit='s', values=[1, 2, 3, 4]),
            'event_index': sc.array(
                dims=['event_index'], unit=None, values=[0, 3, 3, 6]
            ),
            'detector_number': sc.arange('detector_number', 5, unit=None),
            'pixel_offset': sc.vectors(
                dims=['detector_number'],
                values=np.arange(3 * 5).reshape((5, 3)),
                unit='m',
            ),
        }
    )


def detector_transformation_components() -> sc.DataGroup:
    return sc.DataGroup(
        {
            'offset': sc.vector([0.4, 0.0, 11.5], unit='m'),
        }
    )


def _monitor_histogram() -> sc.DataArray:
    return sc.DataArray(
        sc.array(dims=['time'], values=[2, 4, 8, 3], unit='counts'),
        coords={
            'time': sc.epoch(unit='ms')
            + sc.array(dims=['time'], values=[2, 4, 6, 8, 10], unit='ms'),
        },
    )


def _source_data() -> sc.DataGroup:
    return sc.DataGroup(
        {
            'name': 'moderator',
            'probe': 'neutron',
            'type': 'Spallation Neutron Source',
            'position': sc.vector([0, 0, 0], unit='m'),
            'transform': sc.spatial.translation(value=[0, 0, 0], unit='m'),
            'nexus_component_name': 'source',
        }
    )


def _sample_data() -> sc.DataGroup:
    return sc.DataGroup(
        {
            'name': 'water',
            'chemical_formula': 'H2O',
            'type': 'sample+can',
            'nexus_component_name': 'sample',
        }
    )


def _write_transformation(group: snx.Group, offset: sc.Variable) -> None:
    group.create_field('depends_on', sc.scalar('transformations/t1'))
    transformations = group.create_class('transformations', snx.NXtransformations)
    t1 = transformations.create_field('t1', sc.scalar(0.0, unit=offset.unit))
    t1.attrs['depends_on'] = '.'
    t1.attrs['transformation_type'] = 'translation'
    t1.attrs['offset'] = offset.values
    t1.attrs['offset_units'] = str(offset.unit)
    t1.attrs['vector'] = sc.vector([0, 0, 1]).value


def _write_nexus_data(store: Path | BytesIO) -> None:
    with snx.File(store, 'w') as root:
        entry = root.create_class('entry-001', snx.NXentry)
        instrument = entry.create_class('reducer', snx.NXinstrument)

        detector = instrument.create_class('bank12', snx.NXdetector)
        events = detector.create_class('bank12_events', snx.NXevent_data)
        detector_components = _event_data_components()
        events['event_id'] = detector_components['event_id']
        events['event_time_offset'] = detector_components['event_time_offset']
        events['event_time_zero'] = detector_components['event_time_zero']
        events['event_index'] = detector_components['event_index']
        detector['x_pixel_offset'] = detector_components['pixel_offset'].fields.x
        detector['y_pixel_offset'] = detector_components['pixel_offset'].fields.y
        detector['z_pixel_offset'] = detector_components['pixel_offset'].fields.z
        detector['detector_number'] = detector_components['detector_number']
        _write_transformation(detector, detector_transformation_components()['offset'])

        monitor_data = _monitor_histogram()
        monitor = instrument.create_class('monitor', snx.NXmonitor)
        data = monitor.create_class('data', snx.NXdata)
        signal = data.create_field('signal', monitor_data.data)
        signal.attrs['signal'] = 1
        signal.attrs['axes'] = monitor_data.dim
        data.create_field('time', monitor_data.coords['time'])

        source_data = _source_data()
        source = instrument.create_class('source', snx.NXsource)
        source.create_field('name', source_data['name'])
        source.create_field('probe', source_data['probe'])
        source.create_field('type', source_data['type'])
        _write_transformation(source, source_data['position'])

        sample_data = _sample_data()
        sample = entry.create_class('sample', snx.NXsample)
        sample.create_field('name', sample_data['name'])
        sample.create_field('chemical_formula', sample_data['chemical_formula'])
        sample.create_field('type', sample_data['type'])


@contextmanager
def _file_store(request: pytest.FixtureRequest):
    if request.param == BytesIO:
        yield BytesIO()
    else:
        # It would be good to use pyfakefs here, but h5py
        # uses C to open files and that bypasses the fake.
        base = request.getfixturevalue('tmp_path')
        yield base / 'testfile.nxs'


@pytest.fixture(params=[Path, BytesIO, snx.Group])
def nexus_file(request):
    with _file_store(request) as store:
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
        sc.ones(sizes={'event': 6}, unit='counts', dtype='float32'),
        coords={
            'detector_number': components['event_id'],
            'event_time_offset': components['event_time_offset'],
        },
    )

    # Bin by event_index tp broadcast event_time_zero to events
    binned_in_time = sc.DataArray(
        sc.bins(
            data=buffer,
            begin=components['event_index'],
            end=sc.concat(
                [components['event_index'][1:], components['event_index'][-1]],
                dim='event_index',
            ),
            dim='event',
        )
    )
    binned_in_time.bins.coords['event_time_zero'] = sc.bins_like(
        binned_in_time, components['event_time_zero']
    )

    # Bin by detector number like ScippNexus would
    binned = binned_in_time.bins.concat().group(components['detector_number'])
    binned.coords['x_pixel_offset'] = components['pixel_offset'].fields.x
    binned.coords['y_pixel_offset'] = components['pixel_offset'].fields.y
    binned.coords['z_pixel_offset'] = components['pixel_offset'].fields.z
    # Computed position
    offset = detector_transformation_components()['offset']
    binned.coords['position'] = offset + components['pixel_offset']
    return binned


@pytest.fixture()
def expected_monitor() -> sc.DataArray:
    return _monitor_histogram()


@pytest.fixture()
def expected_source() -> sc.DataGroup:
    return _source_data()


@pytest.fixture()
def expected_sample() -> sc.DataGroup:
    return _sample_data()


@pytest.mark.parametrize('entry_name', [None, nexus.types.NeXusEntryName('entry-001')])
@pytest.mark.parametrize(
    'selection',
    [
        None,
        ('event_time_zero', slice(year_zero, None)),
        ('event_time_zero', slice(None, year_zero)),
    ],
)
def test_load_detector(nexus_file, expected_bank12, entry_name, selection):
    detector = nexus.load_detector(
        nexus_file,
        **({'selection': selection} if selection is not None else {}),
        detector_name=nexus.types.NeXusDetectorName('bank12'),
        entry_name=entry_name,
    )
    if selection:
        expected = expected_bank12.bins[selection]
        expected.coords.pop(selection[0])
    else:
        expected = expected_bank12
    sc.testing.assert_identical(detector['bank12_events'], expected)
    offset = detector_transformation_components()['offset']
    sc.testing.assert_identical(
        detector['transform'],
        sc.spatial.translation(unit=offset.unit, value=offset.value),
    )


@pytest.mark.parametrize(
    'selection',
    [
        (),
        {'event_time_zero': slice(2, None)},
        {'event_time_zero': slice(None, 3)},
        {'event_time_zero': slice(1, 3)},
    ],
)
def test_load_and_group_event_data_consistent_with_load_via_detector(
    nexus_file, selection
):
    detector = nexus.load_detector(
        nexus_file,
        selection=selection,
        detector_name=nexus.types.NeXusDetectorName('bank12'),
    )['bank12_events']
    events = nexus.load_event_data(
        nexus_file,
        selection=selection,
        component_name=nexus.types.NeXusDetectorName('bank12'),
    )
    grouped = nexus.group_event_data(
        event_data=events,
        detector_number=detector.coords['detector_number'],
    )
    scipp.testing.assert_identical(detector.data, grouped.data)


def test_group_event_data_does_not_modify_input(nexus_file):
    detector = nexus.load_detector(
        nexus_file,
        detector_name=nexus.types.NeXusDetectorName('bank12'),
    )['bank12_events']
    events = nexus.load_event_data(
        nexus_file,
        component_name=nexus.types.NeXusDetectorName('bank12'),
    )
    _ = nexus.group_event_data(
        event_data=events,
        detector_number=detector.coords['detector_number'],
    )
    assert 'event_time_zero' not in events.bins.coords


def test_load_detector_open_file_with_new_definitions_raises(nexus_file):
    if isinstance(nexus_file, snx.Group):
        with pytest.raises(ValueError, match="new definitions"):
            nexus.load_detector(
                nexus_file,
                detector_name=nexus.types.NeXusDetectorName('bank12'),
                definitions={},
            )
    else:
        nexus.load_detector(
            nexus_file,
            detector_name=nexus.types.NeXusDetectorName('bank12'),
            definitions={},
        )


def test_load_detector_new_definitions_applied(nexus_file, expected_bank12):
    if not isinstance(nexus_file, snx.Group):
        new_definition_used = False

        def detector(*args, **kwargs):
            nonlocal new_definition_used
            new_definition_used = True
            return snx.base_definitions()['NXdetector'](*args, **kwargs)

        nexus.load_detector(
            nexus_file,
            detector_name=nexus.types.NeXusDetectorName('bank12'),
            definitions=dict(
                snx.base_definitions(),
                NXdetector=detector,
            ),
        )
        assert new_definition_used


def test_load_detector_requires_entry_name_if_not_unique(nexus_file):
    if not isinstance(nexus_file, Path):
        # For simplicity, only create a second entry in an actual file
        return

    with snx.File(nexus_file, 'r+') as f:
        f.create_class('entry', snx.NXentry)

    with pytest.raises(ValueError, match="Expected exactly one"):
        nexus.load_detector(
            nexus.types.FilePath(nexus_file),
            detector_name=nexus.types.NeXusDetectorName('bank12'),
            entry_name=None,
        )


def test_load_detector_select_entry_if_not_unique(nexus_file, expected_bank12):
    if not isinstance(nexus_file, Path):
        # For simplicity, only create a second entry in an actual file
        return

    with snx.File(nexus_file, 'r+') as f:
        f.create_class('entry', snx.NXentry)

    detector = nexus.load_detector(
        nexus.types.FilePath(nexus_file),
        detector_name=nexus.types.NeXusDetectorName('bank12'),
        entry_name=nexus.types.NeXusEntryName('entry-001'),
    )
    sc.testing.assert_identical(detector['bank12_events'], expected_bank12)


@pytest.mark.parametrize('entry_name', [None, nexus.types.NeXusEntryName('entry-001')])
@pytest.mark.parametrize(
    'selection',
    [
        None,
        ('time', slice(year_zero.to(unit='ms'), None)),
        ('time', slice(None, year_zero.to(unit='ms'))),
    ],
)
def test_load_monitor(nexus_file, expected_monitor, entry_name, selection):
    monitor = nexus.load_monitor(
        nexus_file,
        **({'selection': selection} if selection is not None else {}),
        monitor_name=nexus.types.AnyNeXusMonitorName('monitor'),
        entry_name=entry_name,
    )
    expected = expected_monitor[selection] if selection else expected_monitor
    sc.testing.assert_identical(monitor['data'], expected)


@pytest.mark.parametrize('entry_name', [None, nexus.types.NeXusEntryName('entry-001')])
@pytest.mark.parametrize('source_name', [None, nexus.types.NeXusSourceName('source')])
def test_load_source(nexus_file, expected_source, entry_name, source_name):
    source = nexus.load_source(
        nexus_file,
        entry_name=entry_name,
        source_name=source_name,
    )
    # NeXus details that we don't need to test as long as the positions are ok:
    del source['depends_on']
    del source['transformations']
    sc.testing.assert_identical(source, nexus.types.AnyRunNeXusSource(expected_source))


@pytest.mark.parametrize(
    ('loader', 'cls', 'name'),
    [
        (nexus.load_source, snx.NXsource, 'NXsource'),
        (nexus.load_sample, snx.NXsample, 'NXsample'),
    ],
)
def test_load_new_definitions_applied(nexus_file, loader, cls, name):
    if not isinstance(nexus_file, snx.Group):
        new_definition_used = False

        def new(*args, **kwargs):
            nonlocal new_definition_used
            new_definition_used = True
            return cls(*args, **kwargs)

        loader(
            nexus_file,
            definitions={
                **snx.base_definitions(),
                name: new,
            },
        )
        assert new_definition_used


@pytest.mark.parametrize('entry_name', [None, nexus.types.NeXusEntryName('entry-001')])
def test_load_sample(nexus_file, expected_sample, entry_name):
    sample = nexus.load_sample(nexus_file, entry_name=entry_name)
    sc.testing.assert_identical(sample, nexus.types.AnyRunNeXusSample(expected_sample))


def test_extract_detector_data():
    detector = sc.DataGroup(
        {
            'jdl2ab': sc.data.binned_x(10, 3),
            'llk': 23,
            ' _': sc.linspace('xx', 2, 3, 10),
        }
    )
    data = nexus.extract_detector_data(nexus.types.AnyRunNeXusDetector(detector))
    sc.testing.assert_identical(data, nexus.types.RawDetectorData(detector['jdl2ab']))


def test_extract_monitor_data():
    monitor = sc.DataGroup(
        {
            '(eed)': sc.data.data_xy(),
            'llk': 23,
            ' _': sc.linspace('xx', 2, 3, 10),
        }
    )
    data = nexus.extract_monitor_data(nexus.types.AnyRunAnyNeXusMonitor(monitor))
    sc.testing.assert_identical(data, nexus.types.RawMonitorData(monitor['(eed)']))


def test_extract_detector_data_requires_unique_dense_data():
    detector = sc.DataGroup(
        {
            'jdl2ab': sc.data.data_xy(),
            'llk': 23,
            'lob': sc.data.data_xy(),
            ' _': sc.linspace('xx', 2, 3, 10),
        }
    )
    with pytest.raises(
        ValueError, match="Cannot uniquely identify the data to extract"
    ):
        nexus.extract_detector_data(nexus.types.AnyRunNeXusDetector(detector))


def test_extract_detector_data_ignores_position_data_array():
    detector = sc.DataGroup(jdl2ab=sc.data.data_xy(), position=sc.data.data_xy())
    nexus.extract_detector_data(nexus.types.AnyRunNeXusDetector(detector))


def test_extract_detector_data_ignores_transform_data_array():
    detector = sc.DataGroup(jdl2ab=sc.data.data_xy(), transform=sc.data.data_xy())
    nexus.extract_detector_data(nexus.types.AnyRunNeXusDetector(detector))


def test_extract_detector_data_requires_unique_event_data():
    detector = sc.DataGroup(
        {
            'jdl2ab': sc.data.binned_x(10, 3),
            'llk': 23,
            'lob': sc.data.binned_x(14, 5),
            ' _': sc.linspace('xx', 2, 3, 10),
        }
    )
    with pytest.raises(
        ValueError, match="Cannot uniquely identify the data to extract"
    ):
        nexus.extract_detector_data(nexus.types.AnyRunNeXusDetector(detector))


def test_extract_detector_data_favors_event_data_over_histogram_data():
    detector = sc.DataGroup(
        {
            'jdl2ab': sc.data.data_xy(),
            'llk': 23,
            'lob': sc.data.binned_x(14, 5),
            ' _': sc.linspace('xx', 2, 3, 10),
        }
    )
    data = nexus.extract_detector_data(nexus.types.AnyRunNeXusDetector(detector))
    sc.testing.assert_identical(data, nexus.types.RawDetectorData(detector['lob']))
