# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical
from scippneutron.conversion.graph.beamline import beamline as beamline_graph
from scippneutron.conversion.graph.tof import elastic as elastic_graph

from ess.reduce import time_of_flight
from ess.reduce.time_of_flight import fakes

sl = pytest.importorskip("sciline")


def test_frame_period_is_pulse_period_if_not_pulse_skipping() -> None:
    pl = sl.Pipeline(time_of_flight.providers())
    period = sc.scalar(123.0, unit="ms")
    pl[time_of_flight.PulsePeriod] = period
    pl[time_of_flight.PulseStride] = 1
    assert_identical(pl.compute(time_of_flight.FramePeriod), period)


@pytest.mark.parametrize("stride", [1, 2, 3, 4])
def test_frame_period_is_multiple_pulse_period_if_pulse_skipping(stride) -> None:
    pl = sl.Pipeline(time_of_flight.providers())
    period = sc.scalar(123.0, unit="ms")
    pl[time_of_flight.PulsePeriod] = period
    pl[time_of_flight.PulseStride] = stride
    assert_identical(pl.compute(time_of_flight.FramePeriod), stride * period)


def test_unwrap_with_no_choppers() -> None:
    # At this small distance the frames are not overlapping (with the given wavelength
    # range), despite not using any choppers.
    distance = sc.scalar(10.0, unit="m")

    beamline = fakes.FakeBeamline(
        choppers={},
        monitors={"detector": distance},
        run_length=sc.scalar(1 / 14, unit="s") * 4,
        events_per_pulse=100_000,
    )

    mon, ref = beamline.get_monitor("detector")

    pl = sl.Pipeline(
        time_of_flight.providers(), params=time_of_flight.default_parameters()
    )

    sim = time_of_flight.simulate_beamline(choppers={}, neutrons=300_000, seed=1234)

    pl[time_of_flight.RawData] = mon
    pl[time_of_flight.SimulationResults] = sim
    pl[time_of_flight.LtotalRange] = distance, distance
    pl[time_of_flight.LookupTableRelativeErrorThreshold] = 1.0

    tofs = pl.compute(time_of_flight.TofData)

    # Convert to wavelength
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    wavs = tofs.transform_coords("wavelength", graph=graph).bins.concat().value
    ref = ref.bins.concat().value

    diff = abs(
        (wavs.coords["wavelength"] - ref.coords["wavelength"])
        / ref.coords["wavelength"]
    )
    # Most errors should be small
    assert np.nanpercentile(diff.values, 96) < 1.0


# At 80m, event_time_offset does not wrap around (all events are within the same pulse).
# At 85m, event_time_offset wraps around.
@pytest.mark.parametrize("dist", [80.0, 85.0])
def test_standard_unwrap(dist) -> None:
    distance = sc.scalar(dist, unit="m")
    choppers = fakes.psc_choppers()
    beamline = fakes.FakeBeamline(
        choppers=choppers,
        monitors={"detector": distance},
        run_length=sc.scalar(1 / 14, unit="s") * 4,
        events_per_pulse=100_000,
    )
    mon, ref = beamline.get_monitor("detector")

    sim = time_of_flight.simulate_beamline(
        choppers=choppers, neutrons=300_000, seed=1234
    )

    pl = sl.Pipeline(
        time_of_flight.providers(), params=time_of_flight.default_parameters()
    )

    pl[time_of_flight.RawData] = mon
    pl[time_of_flight.SimulationResults] = sim
    pl[time_of_flight.LtotalRange] = distance, distance

    tofs = pl.compute(time_of_flight.TofData)

    # Convert to wavelength
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    wavs = tofs.transform_coords("wavelength", graph=graph).bins.concat().value
    ref = ref.bins.concat().value

    diff = abs(
        (wavs.coords["wavelength"] - ref.coords["wavelength"])
        / ref.coords["wavelength"]
    )
    # All errors should be small
    assert np.nanpercentile(diff.values, 100) < 0.01


# At 80m, event_time_offset does not wrap around (all events are within the same pulse).
# At 85m, event_time_offset wraps around.
@pytest.mark.parametrize("dist", [80.0, 85.0])
@pytest.mark.parametrize("dim", ["time_of_flight", "tof"])
def test_standard_unwrap_histogram_mode(dist, dim) -> None:
    distance = sc.scalar(dist, unit="m")
    choppers = fakes.psc_choppers()
    beamline = fakes.FakeBeamline(
        choppers=choppers,
        monitors={"detector": distance},
        run_length=sc.scalar(1 / 14, unit="s") * 4,
        events_per_pulse=100_000,
    )
    mon, ref = beamline.get_monitor("detector")
    mon = (
        mon.hist(
            event_time_offset=sc.linspace(
                "event_time_offset", 0.0, 1000.0 / 14, num=1001, unit="ms"
            ).to(unit="s")
        )
        .sum("pulse")
        .rename(event_time_offset=dim)
    )

    sim = time_of_flight.simulate_beamline(
        choppers=choppers, neutrons=300_000, seed=1234
    )

    pl = sl.Pipeline(
        (*time_of_flight.providers(), time_of_flight.resample_tof_data),
        params=time_of_flight.default_parameters(),
    )

    pl[time_of_flight.RawData] = mon
    pl[time_of_flight.SimulationResults] = sim
    pl[time_of_flight.LtotalRange] = distance, distance
    tofs = pl.compute(time_of_flight.ResampledTofData)
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    wavs = tofs.transform_coords("wavelength", graph=graph)
    ref = ref.bins.concat().value.hist(wavelength=wavs.coords["wavelength"])
    # We divide by the maximum to avoid large relative differences at the edges of the
    # frames where the counts are low.
    diff = (wavs - ref) / ref.max()
    assert np.nanpercentile(diff.values, 96.0) < 0.3


def test_pulse_skipping_unwrap() -> None:
    distance = sc.scalar(100.0, unit="m")
    choppers = fakes.psc_choppers()
    choppers["pulse_skipping"] = fakes.pulse_skipping

    beamline = fakes.FakeBeamline(
        choppers=choppers,
        monitors={"detector": distance},
        run_length=sc.scalar(1.0, unit="s"),
        events_per_pulse=100_000,
    )
    mon, ref = beamline.get_monitor("detector")

    sim = time_of_flight.simulate_beamline(
        choppers=choppers, neutrons=300_000, seed=1234
    )

    pl = sl.Pipeline(
        time_of_flight.providers(), params=time_of_flight.default_parameters()
    )

    pl[time_of_flight.RawData] = mon
    pl[time_of_flight.SimulationResults] = sim
    pl[time_of_flight.LtotalRange] = distance, distance
    pl[time_of_flight.PulseStride] = 2

    tofs = pl.compute(time_of_flight.TofData)

    # Convert to wavelength
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    wavs = tofs.transform_coords("wavelength", graph=graph).bins.concat().value
    ref = ref.bins.concat().value

    diff = abs(
        (wavs.coords["wavelength"] - ref.coords["wavelength"])
        / ref.coords["wavelength"]
    )
    # All errors should be small
    assert np.nanpercentile(diff.values, 100) < 0.01


def test_pulse_skipping_unwrap_when_all_neutrons_arrive_after_second_pulse() -> None:
    distance = sc.scalar(150.0, unit="m")
    choppers = fakes.psc_choppers()
    choppers["pulse_skipping"] = fakes.pulse_skipping

    beamline = fakes.FakeBeamline(
        choppers=choppers,
        monitors={"detector": distance},
        run_length=sc.scalar(1.0, unit="s"),
        events_per_pulse=100_000,
    )
    mon, ref = beamline.get_monitor("detector")

    sim = time_of_flight.simulate_beamline(
        choppers=choppers, neutrons=300_000, seed=1234
    )

    pl = sl.Pipeline(
        time_of_flight.providers(), params=time_of_flight.default_parameters()
    )

    pl[time_of_flight.RawData] = mon
    pl[time_of_flight.SimulationResults] = sim
    pl[time_of_flight.LtotalRange] = distance, distance
    pl[time_of_flight.PulseStride] = 2
    pl[time_of_flight.PulseStrideOffset] = 1  # Start the stride at the second pulse

    tofs = pl.compute(time_of_flight.TofData)

    # Convert to wavelength
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    wavs = tofs.transform_coords("wavelength", graph=graph).bins.concat().value
    ref = ref.bins.concat().value

    diff = abs(
        (wavs.coords["wavelength"] - ref.coords["wavelength"])
        / ref.coords["wavelength"]
    )
    # All errors should be small
    assert np.nanpercentile(diff.values, 100) < 0.01


def test_pulse_skipping_unwrap_when_first_half_of_first_pulse_is_missing() -> None:
    distance = sc.scalar(100.0, unit="m")
    choppers = fakes.psc_choppers()
    choppers["pulse_skipping"] = fakes.pulse_skipping

    beamline = fakes.FakeBeamline(
        choppers=choppers,
        monitors={"detector": distance},
        run_length=sc.scalar(1.0, unit="s"),
        events_per_pulse=100_000,
    )
    mon, ref = beamline.get_monitor("detector")

    sim = time_of_flight.simulate_beamline(
        choppers=choppers, neutrons=300_000, seed=1234
    )

    pl = sl.Pipeline(
        time_of_flight.providers(), params=time_of_flight.default_parameters()
    )

    pl[time_of_flight.RawData] = mon[
        1:
    ].copy()  # Skip first pulse = half of the first frame
    pl[time_of_flight.SimulationResults] = sim
    pl[time_of_flight.LtotalRange] = distance, distance
    pl[time_of_flight.PulseStride] = 2
    pl[time_of_flight.PulseStrideOffset] = 1  # Start the stride at the second pulse

    tofs = pl.compute(time_of_flight.TofData)

    # Convert to wavelength
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    wavs = tofs.transform_coords("wavelength", graph=graph).bins.concat().value
    ref = ref[1:].copy().bins.concat().value

    diff = abs(
        (wavs.coords["wavelength"] - ref.coords["wavelength"])
        / ref.coords["wavelength"]
    )
    # All errors should be small
    assert np.nanpercentile(diff.values, 100) < 0.01


def test_pulse_skipping_unwrap_histogram_mode() -> None:
    distance = sc.scalar(100.0, unit="m")
    choppers = fakes.psc_choppers()
    choppers["pulse_skipping"] = fakes.pulse_skipping

    beamline = fakes.FakeBeamline(
        choppers=choppers,
        monitors={"detector": distance},
        run_length=sc.scalar(1.0, unit="s"),
        events_per_pulse=100_000,
    )
    mon, ref = beamline.get_monitor("detector")
    mon = (
        mon.hist(
            event_time_offset=sc.linspace(
                "event_time_offset", 0.0, 1000.0 / 14, num=1001, unit="ms"
            ).to(unit="s")
        )
        .sum("pulse")
        .rename(event_time_offset="time_of_flight")
    )

    sim = time_of_flight.simulate_beamline(
        choppers=choppers, neutrons=300_000, seed=1234
    )

    pl = sl.Pipeline(
        (*time_of_flight.providers(), time_of_flight.resample_tof_data),
        params=time_of_flight.default_parameters(),
    )

    pl[time_of_flight.RawData] = mon
    pl[time_of_flight.SimulationResults] = sim
    pl[time_of_flight.LtotalRange] = distance, distance
    pl[time_of_flight.PulseStride] = 2
    tofs = pl.compute(time_of_flight.ResampledTofData)
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    wavs = tofs.transform_coords("wavelength", graph=graph)
    ref = ref.bins.concat().value.hist(wavelength=wavs.coords["wavelength"])
    # We divide by the maximum to avoid large relative differences at the edges of the
    # frames where the counts are low.
    diff = (wavs - ref) / ref.max()
    assert np.nanpercentile(diff.values, 96.0) < 0.3
