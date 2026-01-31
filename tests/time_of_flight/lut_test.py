# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc
from scippneutron.chopper import DiskChopper

from ess.reduce import time_of_flight
from ess.reduce.nexus.types import AnyRun
from ess.reduce.time_of_flight import TofLookupTableWorkflow

sl = pytest.importorskip("sciline")


def test_lut_workflow_computes_table():
    wf = TofLookupTableWorkflow()
    wf[time_of_flight.DiskChoppers[AnyRun]] = {}
    wf[time_of_flight.SourcePosition] = sc.vector([0, 0, 0], unit='m')
    wf[time_of_flight.NumberOfSimulatedNeutrons] = 100_000
    wf[time_of_flight.SimulationSeed] = 60
    wf[time_of_flight.PulseStride] = 1

    lmin, lmax = sc.scalar(25.0, unit='m'), sc.scalar(35.0, unit='m')
    dres = sc.scalar(0.1, unit='m')
    tres = sc.scalar(333.0, unit='us')

    wf[time_of_flight.LtotalRange] = lmin, lmax
    wf[time_of_flight.DistanceResolution] = dres
    wf[time_of_flight.TimeResolution] = tres

    table = wf.compute(time_of_flight.TofLookupTable)

    assert table.array.coords['distance'].min() < lmin
    assert table.array.coords['distance'].max() > lmax
    assert table.array.coords['event_time_offset'].max() == sc.scalar(
        1 / 14, unit='s'
    ).to(unit=table.array.coords['event_time_offset'].unit)
    assert sc.isclose(table.distance_resolution, dres)
    # Note that the time resolution is not exactly preserved since we want the table to
    # span exactly the frame period.
    assert sc.isclose(table.time_resolution, tres, rtol=sc.scalar(0.01))


def test_lut_workflow_pulse_skipping():
    wf = TofLookupTableWorkflow()
    wf[time_of_flight.DiskChoppers[AnyRun]] = {}
    wf[time_of_flight.SourcePosition] = sc.vector([0, 0, 0], unit='m')
    wf[time_of_flight.NumberOfSimulatedNeutrons] = 100_000
    wf[time_of_flight.SimulationSeed] = 62
    wf[time_of_flight.PulseStride] = 2

    lmin, lmax = sc.scalar(55.0, unit='m'), sc.scalar(65.0, unit='m')
    dres = sc.scalar(0.1, unit='m')
    tres = sc.scalar(250.0, unit='us')

    wf[time_of_flight.LtotalRange] = lmin, lmax
    wf[time_of_flight.DistanceResolution] = dres
    wf[time_of_flight.TimeResolution] = tres

    table = wf.compute(time_of_flight.TofLookupTable)

    assert table.array.coords['event_time_offset'].max() == 2 * sc.scalar(
        1 / 14, unit='s'
    ).to(unit=table.array.coords['event_time_offset'].unit)


def test_lut_workflow_non_exact_distance_range():
    wf = TofLookupTableWorkflow()
    wf[time_of_flight.DiskChoppers[AnyRun]] = {}
    wf[time_of_flight.SourcePosition] = sc.vector([0, 0, 0], unit='m')
    wf[time_of_flight.NumberOfSimulatedNeutrons] = 100_000
    wf[time_of_flight.SimulationSeed] = 63
    wf[time_of_flight.PulseStride] = 1

    lmin, lmax = sc.scalar(25.0, unit='m'), sc.scalar(35.0, unit='m')
    dres = sc.scalar(0.33, unit='m')
    tres = sc.scalar(250.0, unit='us')

    wf[time_of_flight.LtotalRange] = lmin, lmax
    wf[time_of_flight.DistanceResolution] = dres
    wf[time_of_flight.TimeResolution] = tres

    table = wf.compute(time_of_flight.TofLookupTable)

    assert table.array.coords['distance'].min() < lmin
    assert table.array.coords['distance'].max() > lmax
    assert sc.isclose(table.distance_resolution, dres)


def _make_choppers():
    return {
        'wfm1': DiskChopper(
            axle_position=sc.vector([0, 0, 6.85], unit='m'),
            frequency=sc.scalar(-56, unit='Hz'),
            beam_position=sc.scalar(0, unit='deg'),
            phase=sc.scalar(-93.244, unit='deg'),
            slit_begin=sc.array(dims=["cutout"], values=[-1.9419, 49.5756], unit='deg'),
            slit_end=sc.array(dims=["cutout"], values=[1.9419, 55.7157], unit='deg'),
            slit_height=None,
            radius=None,
        ),
        'WFMC_2': DiskChopper(
            axle_position=sc.vector([0, 0, 7.15], unit='m'),
            frequency=sc.scalar(-56, unit='Hz'),
            beam_position=sc.scalar(0, unit='deg'),
            phase=sc.scalar(-97.128, unit='deg'),
            slit_begin=sc.array(dims=["cutout"], values=[-1.9419, 51.8318], unit='deg'),
            slit_end=sc.array(dims=["cutout"], values=[1.9419, 57.9719], unit='deg'),
            slit_height=None,
            radius=None,
        ),
        'FOC_1': DiskChopper(
            axle_position=sc.vector([0, 0, 8.4], unit='m'),
            frequency=sc.scalar(-42, unit='Hz'),
            beam_position=sc.scalar(0, unit='deg'),
            phase=sc.scalar(-81.3033, unit='deg'),
            slit_begin=sc.array(dims=["cutout"], values=[-5.1362, 42.5536], unit='deg'),
            slit_end=sc.array(dims=["cutout"], values=[5.1362, 54.2095], unit='deg'),
            slit_height=None,
            radius=None,
        ),
        'FOC_2': DiskChopper(
            axle_position=sc.vector([0, 0, 12.2], unit='m'),
            frequency=sc.scalar(-42, unit='Hz'),
            beam_position=sc.scalar(0, unit='deg'),
            phase=sc.scalar(-107.013, unit='deg'),
            slit_begin=sc.array(
                dims=["cutout"], values=[-16.3227, 53.7401], unit='deg'
            ),
            slit_end=sc.array(dims=["cutout"], values=[16.3227, 86.8303], unit='deg'),
            slit_height=None,
            radius=None,
        ),
        'FOC_5': DiskChopper(
            axle_position=sc.vector([0, 0, 33], unit='m'),
            frequency=sc.scalar(-14, unit='Hz'),
            beam_position=sc.scalar(0, unit='deg'),
            phase=sc.scalar(-82.581, unit='deg'),
            slit_begin=sc.array(
                dims=["cutout"], values=[-25.8514, 38.3239], unit='deg'
            ),
            slit_end=sc.array(dims=["cutout"], values=[25.8514, 88.4621], unit='deg'),
            slit_height=None,
            radius=None,
        ),
    }


def test_lut_workflow_computes_table_with_choppers():
    wf = TofLookupTableWorkflow()
    wf[time_of_flight.DiskChoppers[AnyRun]] = _make_choppers()
    wf[time_of_flight.SourcePosition] = sc.vector([0, 0, 0], unit='m')
    wf[time_of_flight.NumberOfSimulatedNeutrons] = 100_000
    wf[time_of_flight.SimulationSeed] = 64
    wf[time_of_flight.PulseStride] = 1

    wf[time_of_flight.LtotalRange] = (
        sc.scalar(35.0, unit='m'),
        sc.scalar(65.0, unit='m'),
    )
    wf[time_of_flight.DistanceResolution] = sc.scalar(0.1, unit='m')
    wf[time_of_flight.TimeResolution] = sc.scalar(250.0, unit='us')
    wf[time_of_flight.LookupTableRelativeErrorThreshold] = 2e3

    table = wf.compute(time_of_flight.TofLookupTable)

    # At low distance, the rays are more focussed
    low_dist = table.array['distance', 2]
    eto = low_dist.coords['event_time_offset'][sc.isfinite(low_dist.data)]
    assert eto.min() > sc.scalar(1.0e4, unit="us").to(unit=eto.unit)
    assert eto.min() < sc.scalar(1.5e4, unit="us").to(unit=eto.unit)
    assert eto.max() > sc.scalar(3.3e4, unit="us").to(unit=eto.unit)
    assert eto.max() < sc.scalar(3.8e4, unit="us").to(unit=eto.unit)

    # At high distance, the rays are more spread out
    high_dist = table.array['distance', -3]
    eto = high_dist.coords['event_time_offset'][sc.isfinite(high_dist.data)]
    assert eto.min() > sc.scalar(1.7e4, unit="us").to(unit=eto.unit)
    assert eto.min() < sc.scalar(2.2e4, unit="us").to(unit=eto.unit)
    assert eto.max() > sc.scalar(6.4e4, unit="us").to(unit=eto.unit)
    assert eto.max() < sc.scalar(6.9e4, unit="us").to(unit=eto.unit)


def test_lut_workflow_computes_table_with_choppers_full_beamline_range():
    wf = TofLookupTableWorkflow()
    wf[time_of_flight.DiskChoppers[AnyRun]] = _make_choppers()
    wf[time_of_flight.SourcePosition] = sc.vector([0, 0, 0], unit='m')
    wf[time_of_flight.NumberOfSimulatedNeutrons] = 100_000
    wf[time_of_flight.SimulationSeed] = 64
    wf[time_of_flight.PulseStride] = 1

    wf[time_of_flight.LtotalRange] = (
        sc.scalar(5.0, unit='m'),
        sc.scalar(65.0, unit='m'),
    )
    wf[time_of_flight.DistanceResolution] = sc.scalar(0.1, unit='m')
    wf[time_of_flight.TimeResolution] = sc.scalar(250.0, unit='us')
    wf[time_of_flight.LookupTableRelativeErrorThreshold] = 2e3

    table = wf.compute(time_of_flight.TofLookupTable)

    # Close to source: early times and large spread
    da = table.array['distance', 2]
    eto = da.coords['event_time_offset'][sc.isfinite(da.data)]
    assert eto.min() > sc.scalar(0.0, unit="us").to(unit=eto.unit)
    assert eto.min() < sc.scalar(1.0e3, unit="us").to(unit=eto.unit)
    assert eto.max() > sc.scalar(2.5e4, unit="us").to(unit=eto.unit)
    assert eto.max() < sc.scalar(3.0e4, unit="us").to(unit=eto.unit)

    # Just after WFM choppers, very small range
    da = table.array['distance', 27]
    eto = da.coords['event_time_offset'][sc.isfinite(da.data)]
    assert eto.min() > sc.scalar(4.0e3, unit="us").to(unit=eto.unit)
    assert eto.min() < sc.scalar(5.0e3, unit="us").to(unit=eto.unit)
    assert eto.max() > sc.scalar(7.0e3, unit="us").to(unit=eto.unit)
    assert eto.max() < sc.scalar(8.0e3, unit="us").to(unit=eto.unit)

    # Before last chopper
    da = table.array['distance', 272]
    eto = da.coords['event_time_offset'][sc.isfinite(da.data)]
    assert eto.min() > sc.scalar(9.0e3, unit="us").to(unit=eto.unit)
    assert eto.min() < sc.scalar(1.2e4, unit="us").to(unit=eto.unit)
    assert eto.max() > sc.scalar(3.1e4, unit="us").to(unit=eto.unit)
    assert eto.max() < sc.scalar(3.5e4, unit="us").to(unit=eto.unit)

    # At the top of the table
    da = table.array['distance', -3]
    eto = da.coords['event_time_offset'][sc.isfinite(da.data)]
    assert eto.min() > sc.scalar(1.7e4, unit="us").to(unit=eto.unit)
    assert eto.min() < sc.scalar(2.2e4, unit="us").to(unit=eto.unit)
    assert eto.max() > sc.scalar(6.4e4, unit="us").to(unit=eto.unit)
    assert eto.max() < sc.scalar(6.9e4, unit="us").to(unit=eto.unit)


def test_lut_workflow_raises_for_distance_before_source():
    wf = TofLookupTableWorkflow()
    wf[time_of_flight.DiskChoppers[AnyRun]] = {}
    wf[time_of_flight.SourcePosition] = sc.vector([0, 0, 10], unit='m')
    wf[time_of_flight.NumberOfSimulatedNeutrons] = 100_000
    wf[time_of_flight.SimulationSeed] = 65
    wf[time_of_flight.PulseStride] = 1

    # Setting the starting point at zero will make a table that would cover a range
    # from -0.2m to 65.0m
    wf[time_of_flight.LtotalRange] = (
        sc.scalar(0.0, unit='m'),
        sc.scalar(65.0, unit='m'),
    )
    wf[time_of_flight.DistanceResolution] = sc.scalar(0.1, unit='m')
    wf[time_of_flight.TimeResolution] = sc.scalar(250.0, unit='us')

    with pytest.raises(ValueError, match="No simulation reading found for distance"):
        _ = wf.compute(time_of_flight.TofLookupTable)
