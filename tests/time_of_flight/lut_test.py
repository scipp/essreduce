# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc

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

    table = wf.compute(time_of_flight.TimeOfFlightLookupTable)

    assert table.coords['distance'].min() < lmin
    assert table.coords['distance'].max() > lmax
    assert table.coords['event_time_offset'].max() == sc.scalar(1 / 14, unit='s').to(
        unit=table.coords['event_time_offset'].unit
    )
    assert sc.isclose(table.coords['distance_resolution'], dres)
    # Note that the time resolution is not exactly preserved since we want the table to
    # span exactly the frame period.
    assert sc.isclose(table.coords['time_resolution'], tres, rtol=sc.scalar(0.01))


def test_lut_workflow_computes_table_in_chunks():
    wf = TofLookupTableWorkflow()
    wf[time_of_flight.DiskChoppers[AnyRun]] = {}
    wf[time_of_flight.SourcePosition] = sc.vector([0, 0, 0], unit='m')
    # Lots of neutrons activates chunking
    wf[time_of_flight.NumberOfSimulatedNeutrons] = 1_000_000
    wf[time_of_flight.SimulationSeed] = 61
    wf[time_of_flight.PulseStride] = 1

    lmin, lmax = sc.scalar(25.0, unit='m'), sc.scalar(35.0, unit='m')
    dres = sc.scalar(0.1, unit='m')
    tres = sc.scalar(250.0, unit='us')

    wf[time_of_flight.LtotalRange] = lmin, lmax
    wf[time_of_flight.DistanceResolution] = dres
    wf[time_of_flight.TimeResolution] = tres

    table = wf.compute(time_of_flight.TimeOfFlightLookupTable)

    assert table.coords['distance'].min() < lmin
    assert table.coords['distance'].max() > lmax
    assert table.coords['event_time_offset'].max() == sc.scalar(1 / 14, unit='s').to(
        unit=table.coords['event_time_offset'].unit
    )
    assert sc.isclose(table.coords['distance_resolution'], dres)
    # Note that the time resolution is not exactly preserved since we want the table to
    # span exactly the frame period.
    assert sc.isclose(table.coords['time_resolution'], tres, rtol=sc.scalar(0.01))


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

    table = wf.compute(time_of_flight.TimeOfFlightLookupTable)

    assert table.coords['event_time_offset'].max() == 2 * sc.scalar(
        1 / 14, unit='s'
    ).to(unit=table.coords['event_time_offset'].unit)


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

    table = wf.compute(time_of_flight.TimeOfFlightLookupTable)

    assert table.coords['distance'].min() < lmin
    assert table.coords['distance'].max() > lmax
    assert sc.isclose(table.coords['distance_resolution'], dres)
