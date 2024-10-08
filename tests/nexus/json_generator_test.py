# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
import scippnexus as snx
from scipp.testing import assert_identical

from ess.reduce import data
from ess.reduce.nexus.json_generator import event_data_generator
from ess.reduce.nexus.json_nexus import json_nexus_group


def test_event_data_generator_monitor_events_round_trip() -> None:
    filename = data.loki_tutorial_sample_run_60250()
    monitor = snx.load(filename, root='entry/instrument/monitor_1/monitor_1_events')
    generator = event_data_generator(monitor)
    for i in range(len(monitor)):
        group = json_nexus_group(next(generator))
        assert_identical(group[()], monitor[i : i + 1])
    with pytest.raises(StopIteration):
        next(generator)


def test_event_data_generator_detector_events_round_trip() -> None:
    filename = data.loki_tutorial_sample_run_60250()
    detector = snx.load(
        filename, root='entry/instrument/larmor_detector/larmor_detector_events'
    )
    generator = event_data_generator(detector)
    for i in range(100):
        group = json_nexus_group(next(generator))
        assert_identical(group[()], detector[i : i + 1])


def test_event_data_generator_without_event_id_yields_ones() -> None:
    filename = data.loki_tutorial_sample_run_60250()
    base = snx.load(filename, root='entry/instrument/monitor_1/monitor_1_events')
    monitor = base.bins.drop_coords('event_id')
    generator = event_data_generator(monitor)
    for i in range(100):
        group = json_nexus_group(next(generator))
        assert_identical(group[()], base[i : i + 1])
