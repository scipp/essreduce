# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
from scippneutron.chopper import DiskChopper
from scippneutron.conversion.graph.beamline import beamline as beamline_graph
from scippneutron.conversion.graph.tof import elastic as elastic_graph

from ess.reduce import time_of_flight
from ess.reduce.nexus.types import DetectorData, SampleRun
from ess.reduce.time_of_flight import GenericTofWorkflow, TofLookupTableWorkflow, fakes

sl = pytest.importorskip("sciline")


def make_lut_workflow(choppers, neutrons, seed, pulse_stride):
    lut_wf = TofLookupTableWorkflow()
    lut_wf[time_of_flight.DiskChoppers] = choppers
    lut_wf[time_of_flight.SourcePosition] = fakes.source_position()
    lut_wf[time_of_flight.NumberOfSimulatedNeutrons] = neutrons
    lut_wf[time_of_flight.SimulationSeed] = seed
    lut_wf[time_of_flight.PulseStride] = pulse_stride
    lut_wf[time_of_flight.SimulationResults] = lut_wf.compute(
        time_of_flight.SimulationResults
    )
    return lut_wf


@pytest.fixture(scope="module")
def lut_workflow_psc_choppers():
    return make_lut_workflow(
        choppers=fakes.psc_choppers(), neutrons=500_000, seed=1234, pulse_stride=1
    )


@pytest.fixture(scope="module")
def lut_workflow_pulse_skipping():
    return make_lut_workflow(
        choppers=fakes.pulse_skipping_choppers(),
        neutrons=500_000,
        seed=112,
        pulse_stride=2,
    )


def _make_workflow_event_mode(
    distance,
    choppers,
    lut_workflow,
    seed,
    pulse_stride_offset,
    error_threshold,
):
    beamline = fakes.FakeBeamline(
        choppers=choppers,
        monitors={"detector": distance},
        run_length=sc.scalar(1 / 14, unit="s") * 4,
        events_per_pulse=300_000,
        seed=seed,
    )
    mon, ref = beamline.get_monitor("detector")

    pl = GenericTofWorkflow(run_types=[SampleRun], monitor_types=[])
    pl[DetectorData[SampleRun]] = mon
    pl[time_of_flight.DetectorLtotal[SampleRun]] = distance
    pl[time_of_flight.PulseStrideOffset] = pulse_stride_offset

    lut_wf = lut_workflow.copy()
    lut_wf[time_of_flight.LtotalRange] = distance, distance
    lut_wf[time_of_flight.LookupTableRelativeErrorThreshold] = error_threshold

    pl[time_of_flight.TimeOfFlightLookupTable] = lut_wf.compute(
        time_of_flight.TimeOfFlightLookupTable
    )

    return pl, ref


def _make_workflow_histogram_mode(dim, distance, choppers, lut_workflow, seed):
    beamline = fakes.FakeBeamline(
        choppers=choppers,
        monitors={"detector": distance},
        run_length=sc.scalar(1 / 14, unit="s") * 4,
        events_per_pulse=100_000,
        seed=seed,
    )
    mon, ref = beamline.get_monitor("detector")
    mon = mon.hist(
        event_time_offset=sc.linspace(
            "event_time_offset", 0.0, 1000.0 / 14, num=301, unit="ms"
        ).to(unit=mon.bins.coords["event_time_offset"].bins.unit)
    ).rename(event_time_offset=dim)

    pl = GenericTofWorkflow(run_types=[SampleRun], monitor_types=[])
    pl[DetectorData[SampleRun]] = mon
    pl[time_of_flight.DetectorLtotal[SampleRun]] = distance

    lut_wf = lut_workflow.copy()
    lut_wf[time_of_flight.LtotalRange] = distance, distance

    pl[time_of_flight.TimeOfFlightLookupTable] = lut_wf.compute(
        time_of_flight.TimeOfFlightLookupTable
    )

    return pl, ref


def _validate_result_events(tofs, ref, percentile, diff_threshold, rtol):
    assert "event_time_offset" not in tofs.coords

    # Convert to wavelength
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    wavs = tofs.transform_coords("wavelength", graph=graph).bins.concat().value

    diff = abs(
        (wavs.coords["wavelength"] - ref.coords["wavelength"])
        / ref.coords["wavelength"]
    )
    # Most errors should be small
    assert np.nanpercentile(diff.values, percentile) < diff_threshold
    # Make sure that we have not lost too many events (we lose some because they may be
    # given a NaN tof from the lookup).
    nevents = sc.sum(~sc.isnan(wavs.coords['wavelength'])).to(dtype=float)
    nevents.unit = 'counts'
    assert sc.isclose(ref.data.sum(), nevents, rtol=sc.scalar(rtol))


def _validate_result_histogram_mode(tofs, ref, percentile, diff_threshold, rtol):
    assert "time_of_flight" not in tofs.coords
    assert "frame_time" not in tofs.coords

    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    wavs = tofs.transform_coords("wavelength", graph=graph)
    ref = ref.hist(wavelength=wavs.coords["wavelength"])
    # We divide by the maximum to avoid large relative differences at the edges of the
    # frames where the counts are low.
    diff = (wavs - ref) / ref.max()
    assert np.nanpercentile(diff.values, percentile) < diff_threshold
    # Make sure that we have not lost too many events (we lose some because they may be
    # given a NaN tof from the lookup).
    assert sc.isclose(ref.data.nansum(), tofs.data.nansum(), rtol=sc.scalar(rtol))


def test_unwrap_with_no_choppers() -> None:
    # At this small distance the frames are not overlapping (with the given wavelength
    # range), despite not using any choppers.
    distance = sc.scalar(10.0, unit="m")
    choppers = {}

    lut_wf = make_lut_workflow(
        choppers=choppers, neutrons=300_000, seed=1234, pulse_stride=1
    )

    pl, ref = _make_workflow_event_mode(
        distance=distance,
        choppers=choppers,
        lut_workflow=lut_wf,
        seed=1,
        pulse_stride_offset=0,
        error_threshold=1.0,
    )

    tofs = pl.compute(time_of_flight.DetectorTofData[SampleRun])

    _validate_result_events(
        tofs=tofs, ref=ref, percentile=96, diff_threshold=1.0, rtol=0.02
    )


# At 30m, event_time_offset does not wrap around (all events within the first pulse).
# At 60m, all events are within the second pulse.
# At 80m, events are split between the second and third pulse.
# At 108m, events are split between the third and fourth pulse.
@pytest.mark.parametrize("dist", [30.0, 60.0, 80.0, 108.0])
def test_standard_unwrap(dist, lut_workflow_psc_choppers) -> None:
    pl, ref = _make_workflow_event_mode(
        distance=sc.scalar(dist, unit="m"),
        choppers=fakes.psc_choppers(),
        lut_workflow=lut_workflow_psc_choppers,
        seed=2,
        # pulse_stride=1,
        pulse_stride_offset=0,
        error_threshold=0.1,
    )

    tofs = pl.compute(time_of_flight.DetectorTofData[SampleRun])

    _validate_result_events(
        tofs=tofs, ref=ref, percentile=100, diff_threshold=0.02, rtol=0.05
    )


# At 30m, event_time_offset does not wrap around (all events within the first pulse).
# At 60m, all events are within the second pulse.
# At 80m, events are split between the second and third pulse.
# At 108m, events are split between the third and fourth pulse.
@pytest.mark.parametrize("dist", [30.0, 60.0, 80.0, 108.0])
@pytest.mark.parametrize("dim", ["time_of_flight", "tof", "frame_time"])
def test_standard_unwrap_histogram_mode(dist, dim, lut_workflow_psc_choppers) -> None:
    pl, ref = _make_workflow_histogram_mode(
        dim=dim,
        distance=sc.scalar(dist, unit="m"),
        choppers=fakes.psc_choppers(),
        lut_workflow=lut_workflow_psc_choppers,
        seed=37,
    )

    tofs = pl.compute(time_of_flight.DetectorTofData[SampleRun])

    _validate_result_histogram_mode(
        tofs=tofs, ref=ref, percentile=96, diff_threshold=0.4, rtol=0.05
    )


@pytest.mark.parametrize("dist", [60.0, 100.0])
def test_pulse_skipping_unwrap(dist, lut_workflow_pulse_skipping) -> None:
    pl, ref = _make_workflow_event_mode(
        distance=sc.scalar(dist, unit="m"),
        choppers=fakes.pulse_skipping_choppers(),
        lut_workflow=lut_workflow_pulse_skipping,
        seed=432,
        # pulse_stride=2,
        pulse_stride_offset=1,
        error_threshold=0.1,
    )

    tofs = pl.compute(time_of_flight.DetectorTofData[SampleRun])

    _validate_result_events(
        tofs=tofs, ref=ref, percentile=100, diff_threshold=0.1, rtol=0.05
    )


def test_pulse_skipping_unwrap_180_phase_shift() -> None:
    choppers = fakes.pulse_skipping_choppers()
    choppers["pulse_skipping"].phase.value += 180.0

    lut_wf = make_lut_workflow(
        choppers=choppers, neutrons=500_000, seed=111, pulse_stride=2
    )

    pl, ref = _make_workflow_event_mode(
        distance=sc.scalar(100.0, unit="m"),
        choppers=choppers,
        lut_workflow=lut_wf,
        seed=55,
        pulse_stride_offset=1,
        error_threshold=0.1,
    )

    tofs = pl.compute(time_of_flight.DetectorTofData[SampleRun])

    _validate_result_events(
        tofs=tofs, ref=ref, percentile=100, diff_threshold=0.1, rtol=0.05
    )


@pytest.mark.parametrize("dist", [60.0, 100.0])
def test_pulse_skipping_stride_offset_guess_gives_expected_result(
    dist, lut_workflow_pulse_skipping
) -> None:
    pl, ref = _make_workflow_event_mode(
        distance=sc.scalar(dist, unit="m"),
        choppers=fakes.pulse_skipping_choppers(),
        lut_workflow=lut_workflow_pulse_skipping,
        seed=97,
        pulse_stride_offset=None,
        error_threshold=0.1,
    )

    tofs = pl.compute(time_of_flight.DetectorTofData[SampleRun])

    _validate_result_events(
        tofs=tofs, ref=ref, percentile=100, diff_threshold=0.1, rtol=0.05
    )


def test_pulse_skipping_unwrap_when_all_neutrons_arrive_after_second_pulse() -> None:
    choppers = fakes.pulse_skipping_choppers()
    choppers['chopper'] = DiskChopper(
        frequency=sc.scalar(-14.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(-35.0, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 8.0], unit="m"),
        slit_begin=sc.array(dims=["cutout"], values=np.array([10.0]), unit="deg"),
        slit_end=sc.array(dims=["cutout"], values=np.array([25.0]), unit="deg"),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    lut_wf = make_lut_workflow(
        choppers=choppers, neutrons=500_000, seed=222, pulse_stride=2
    )

    pl, ref = _make_workflow_event_mode(
        distance=sc.scalar(150.0, unit="m"),
        choppers=choppers,
        lut_workflow=lut_wf,
        seed=6,
        pulse_stride_offset=1,
        error_threshold=0.1,
    )

    tofs = pl.compute(time_of_flight.DetectorTofData[SampleRun])

    _validate_result_events(
        tofs=tofs, ref=ref, percentile=100, diff_threshold=0.1, rtol=0.05
    )


def test_pulse_skipping_unwrap_when_first_half_of_first_pulse_is_missing() -> None:
    distance = sc.scalar(100.0, unit="m")
    choppers = fakes.pulse_skipping_choppers()

    beamline = fakes.FakeBeamline(
        choppers=choppers,
        monitors={"detector": distance},
        run_length=sc.scalar(1.0, unit="s"),
        events_per_pulse=100_000,
        seed=21,
    )
    mon, ref = beamline.get_monitor("detector")

    lut_wf = make_lut_workflow(
        choppers=choppers, neutrons=300_000, seed=1234, pulse_stride=2
    )
    lut_wf[time_of_flight.LtotalRange] = distance, distance

    pl = GenericTofWorkflow(run_types=[SampleRun], monitor_types=[])

    # Skip first pulse = half of the first frame
    a = mon.group('event_time_zero')['event_time_zero', 1:]
    a.bins.coords['event_time_zero'] = sc.bins_like(a, a.coords['event_time_zero'])
    pl[DetectorData[SampleRun]] = a.bins.concat('event_time_zero')
    pl[time_of_flight.DetectorLtotal[SampleRun]] = distance

    pl[time_of_flight.TimeOfFlightLookupTable] = lut_wf.compute(
        time_of_flight.TimeOfFlightLookupTable
    )
    pl[time_of_flight.PulseStrideOffset] = 1  # Start the stride at the second pulse

    tofs = pl.compute(time_of_flight.DetectorTofData[SampleRun])

    # Convert to wavelength
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    wavs = tofs.transform_coords("wavelength", graph=graph).bins.concat().value
    # Bin the events in toa starting from the pulse period to skip the first pulse.
    ref = (
        ref.bin(
            toa=sc.concat(
                [
                    sc.scalar(1 / 14, unit='s').to(unit=ref.coords['toa'].unit),
                    ref.coords['toa'].max() * 1.01,
                ],
                dim='toa',
            )
        )
        .bins.concat()
        .value
    )

    # Sort the events according id to make sure we are comparing the same values.
    wavs = sc.sort(wavs, key=wavs.coords['id'])
    ref = sc.sort(ref, key=ref.coords['id'])

    diff = abs(
        (wavs.coords["wavelength"] - ref.coords["wavelength"])
        / ref.coords["wavelength"]
    )
    # All errors should be small
    assert np.nanpercentile(diff.values, 100) < 0.05
    # Make sure that we have not lost too many events (we lose some because they may be
    # given a NaN tof from the lookup).
    assert sc.isclose(
        pl.compute(DetectorData[SampleRun]).data.nansum(),
        tofs.data.nansum(),
        rtol=sc.scalar(1.0e-3),
    )


def test_pulse_skipping_stride_3() -> None:
    choppers = fakes.pulse_skipping_choppers()
    choppers["pulse_skipping"].frequency.value = -14.0 / 3.0

    lut_wf = make_lut_workflow(
        choppers=choppers, neutrons=500_000, seed=111, pulse_stride=3
    )

    pl, ref = _make_workflow_event_mode(
        distance=sc.scalar(150.0, unit="m"),
        choppers=choppers,
        lut_workflow=lut_wf,
        seed=68,
        pulse_stride_offset=None,
        error_threshold=0.1,
    )

    tofs = pl.compute(time_of_flight.DetectorTofData[SampleRun])

    _validate_result_events(
        tofs=tofs, ref=ref, percentile=100, diff_threshold=0.1, rtol=0.05
    )


def test_pulse_skipping_unwrap_histogram_mode(lut_workflow_pulse_skipping) -> None:
    pl, ref = _make_workflow_histogram_mode(
        dim='time_of_flight',
        distance=sc.scalar(50.0, unit="m"),
        choppers=fakes.pulse_skipping_choppers(),
        lut_workflow=lut_workflow_pulse_skipping,
        seed=9,
    )

    tofs = pl.compute(time_of_flight.DetectorTofData[SampleRun])

    _validate_result_histogram_mode(
        tofs=tofs, ref=ref, percentile=96, diff_threshold=0.4, rtol=0.05
    )


@pytest.mark.parametrize("dtype", ["int32", "int64"])
def test_unwrap_int(dtype, lut_workflow_psc_choppers) -> None:
    pl, ref = _make_workflow_event_mode(
        distance=sc.scalar(80.0, unit="m"),
        choppers=fakes.psc_choppers(),
        lut_workflow=lut_workflow_psc_choppers,
        seed=2,
        pulse_stride_offset=0,
        error_threshold=0.1,
    )

    mon = pl.compute(DetectorData[SampleRun]).copy()
    mon.bins.coords["event_time_offset"] = mon.bins.coords["event_time_offset"].to(
        dtype=dtype, unit="ns"
    )
    pl[DetectorData[SampleRun]] = mon

    tofs = pl.compute(time_of_flight.DetectorTofData[SampleRun])

    _validate_result_events(
        tofs=tofs, ref=ref, percentile=100, diff_threshold=0.02, rtol=0.05
    )
