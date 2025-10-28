# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import sciline
import scipp as sc
import scippnexus as snx
from scipp.testing import assert_identical

from ess.reduce import time_of_flight
from ess.reduce.nexus.types import (
    AnyRun,
    DiskChoppers,
    EmptyDetector,
    NeXusData,
    Position,
    RawDetector,
    SampleRun,
)
from ess.reduce.time_of_flight import (
    GenericTofWorkflow,
    TofLookupTableWorkflow,
    fakes,
)

sl = pytest.importorskip("sciline")


@pytest.fixture
def calibrated_beamline():
    return sc.DataArray(
        data=sc.ones(dims=["detector_number"], shape=[10]),
        coords={
            "Ltotal": sc.scalar(80.0, unit="m"),
            "detector_number": sc.array(
                dims=["detector_number"], values=np.arange(10), unit=None
            ),
        },
    )


@pytest.fixture
def nexus_data():
    events = sc.DataArray(
        data=sc.ones(dims=["event"], shape=[1000]),
        coords={
            "event_time_offset": sc.linspace(
                "event", 0.0, 1000.0 / 14, num=1000, unit="ms"
            ).to(unit="ns"),
            "event_id": sc.array(
                dims=["event"], values=np.arange(1000) % 10, unit=None
            ),
        },
    )
    return sc.DataArray(
        sc.bins(
            begin=sc.array(dims=["pulse"], values=[0], unit=None),
            data=events,
            dim="event",
        )
    )


def test_TofLookupTableWorkflow_can_compute_tof_lut():
    wf = TofLookupTableWorkflow()
    wf[DiskChoppers[AnyRun]] = fakes.psc_choppers()
    wf[time_of_flight.NumberOfSimulatedNeutrons] = 10_000
    wf[time_of_flight.LtotalRange] = (
        sc.scalar(0.0, unit="m"),
        sc.scalar(100.0, unit="m"),
    )
    wf[time_of_flight.SourcePosition] = fakes.source_position()
    lut = wf.compute(time_of_flight.TimeOfFlightLookupTable)
    assert isinstance(lut, sc.DataArray)


def test_GenericTofWorkflow_with_tof_lut_from_tof_simulation(
    calibrated_beamline: sc.DataArray, nexus_data: sc.DataArray
):
    wf = GenericTofWorkflow(run_types=[SampleRun], monitor_types=[])
    wf[EmptyDetector[SampleRun]] = calibrated_beamline
    wf[NeXusData[snx.NXdetector, SampleRun]] = nexus_data
    # Unused because calibrated_beamline contains Ltotal but needed by wf structure
    wf[Position[snx.NXsample, SampleRun]] = sc.vector([1e10, 1e10, 1e10], unit='m')
    wf[Position[snx.NXsource, SampleRun]] = sc.vector([1e10, 1e10, 1e10], unit='m')

    # Should be able to compute DetectorData without chopper and simulation params
    # This contains event_time_offset (time-of-arrival).
    _ = wf.compute(RawDetector[SampleRun])
    # By default, the workflow tries to load the LUT from file
    with pytest.raises(sciline.UnsatisfiedRequirement):
        _ = wf.compute(time_of_flight.TimeOfFlightLookupTable)
    with pytest.raises(sciline.UnsatisfiedRequirement):
        _ = wf.compute(time_of_flight.TofDetector[SampleRun])

    lut_wf = TofLookupTableWorkflow()
    lut_wf[DiskChoppers[AnyRun]] = fakes.psc_choppers()
    lut_wf[time_of_flight.NumberOfSimulatedNeutrons] = 10_000
    lut_wf[time_of_flight.LtotalRange] = (
        sc.scalar(0.0, unit="m"),
        sc.scalar(100.0, unit="m"),
    )
    lut_wf[time_of_flight.SourcePosition] = fakes.source_position()
    table = lut_wf.compute(time_of_flight.TimeOfFlightLookupTable)

    wf[time_of_flight.TimeOfFlightLookupTable] = table
    # Should now be able to compute DetectorData with chopper and simulation params
    detector = wf.compute(time_of_flight.TofDetector[SampleRun])
    assert 'tof' in detector.bins.coords


def test_GenericTofWorkflow_with_tof_lut_from_file(
    calibrated_beamline: sc.DataArray,
    nexus_data: sc.DataArray,
    tmp_path: pytest.TempPathFactory,
):
    lut_wf = TofLookupTableWorkflow()
    lut_wf[DiskChoppers[AnyRun]] = fakes.psc_choppers()
    lut_wf[time_of_flight.NumberOfSimulatedNeutrons] = 10_000
    lut_wf[time_of_flight.LtotalRange] = (
        sc.scalar(0.0, unit="m"),
        sc.scalar(100.0, unit="m"),
    )
    lut_wf[time_of_flight.SourcePosition] = fakes.source_position()
    lut = lut_wf.compute(time_of_flight.TimeOfFlightLookupTable)
    lut.save_hdf5(filename=tmp_path / "lut.h5")

    wf = GenericTofWorkflow(run_types=[SampleRun], monitor_types=[])
    wf[EmptyDetector[SampleRun]] = calibrated_beamline
    wf[NeXusData[snx.NXdetector, SampleRun]] = nexus_data
    wf[time_of_flight.TimeOfFlightLookupTableFilename] = (
        tmp_path / "lut.h5"
    ).as_posix()
    # Unused because calibrated_beamline contains Ltotal but needed by wf structure
    wf[Position[snx.NXsample, SampleRun]] = sc.vector([1e10, 1e10, 1e10], unit='m')
    wf[Position[snx.NXsource, SampleRun]] = sc.vector([1e10, 1e10, 1e10], unit='m')

    loaded_lut = wf.compute(time_of_flight.TimeOfFlightLookupTable)
    assert_identical(lut, loaded_lut)

    detector = wf.compute(time_of_flight.TofDetector[SampleRun])
    assert 'tof' in detector.bins.coords
