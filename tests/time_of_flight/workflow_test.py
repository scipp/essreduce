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
    CalibratedBeamline,
    DetectorData,
    DiskChoppers,
    NeXusData,
    Position,
    SampleRun,
)
from ess.reduce.time_of_flight import GenericTofWorkflow, TofLutProvider, fakes

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


def test_GenericTofWorkflow_can_compute_tof_lut_without_nexus_file_or_detector_info():
    wf = GenericTofWorkflow(
        tof_lut_provider=TofLutProvider.TOF,
        run_types=[SampleRun],
        monitor_types=[],
    )
    wf[DiskChoppers[SampleRun]] = fakes.psc_choppers()
    wf[time_of_flight.types.NumberOfSimulatedNeutrons] = 10_000
    wf[time_of_flight.types.LtotalRange] = (
        sc.scalar(0.0, unit="m"),
        sc.scalar(100.0, unit="m"),
    )
    wf[Position[snx.NXsource, SampleRun]] = fakes.source_position()
    lut = wf.compute(time_of_flight.TimeOfFlightLookupTable)
    assert isinstance(lut, sc.DataArray)


def test_GenericTofWorkflow_with_tof_lut_from_tof_simulation(
    calibrated_beamline: sc.DataArray, nexus_data: sc.DataArray
):
    wf = GenericTofWorkflow(
        tof_lut_provider=TofLutProvider.TOF,
        run_types=[SampleRun],
        monitor_types=[],
    )
    wf[CalibratedBeamline[SampleRun]] = calibrated_beamline
    wf[NeXusData[snx.NXdetector, SampleRun]] = nexus_data

    # Should be able to compute DetectorData without chopper and simulation params
    # This contains event_time_offset (time-of-arrival).
    _ = wf.compute(DetectorData[SampleRun])
    # LUT and Tof data cannot be computed without chopper and simulation params
    with pytest.raises(sciline.UnsatisfiedRequirement):
        _ = wf.compute(time_of_flight.TimeOfFlightLookupTable)
    with pytest.raises(sciline.UnsatisfiedRequirement):
        _ = wf.compute(time_of_flight.DetectorTofData[SampleRun])

    wf[DiskChoppers[SampleRun]] = fakes.psc_choppers()
    wf[time_of_flight.types.NumberOfSimulatedNeutrons] = 10_000
    wf[time_of_flight.types.LtotalRange] = (
        sc.scalar(0.0, unit="m"),
        sc.scalar(100.0, unit="m"),
    )
    wf[Position[snx.NXsource, SampleRun]] = fakes.source_position()

    # Should be able to compute DetectorData with chopper and simulation params
    _ = wf.compute(time_of_flight.TimeOfFlightLookupTable)
    detector = wf.compute(time_of_flight.DetectorTofData[SampleRun])
    assert 'tof' in detector.bins.coords


def test_GenericTofWorkflow_with_tof_lut_from_mcstas_simulation():
    with pytest.raises(
        NotImplementedError, match="McStas simulation not implemented yet"
    ):
        GenericTofWorkflow(
            tof_lut_provider=TofLutProvider.MCSTAS,
            run_types=[SampleRun],
            monitor_types=[],
        )


def test_GenericTofWorkflow_with_tof_lut_from_file(
    calibrated_beamline: sc.DataArray,
    nexus_data: sc.DataArray,
    tmp_path: pytest.TempPathFactory,
):
    make_lut_wf = GenericTofWorkflow(
        tof_lut_provider=TofLutProvider.TOF,
        run_types=[SampleRun],
        monitor_types=[],
    )
    make_lut_wf[DiskChoppers[SampleRun]] = fakes.psc_choppers()
    make_lut_wf[time_of_flight.types.NumberOfSimulatedNeutrons] = 10_000
    make_lut_wf[time_of_flight.types.LtotalRange] = (
        sc.scalar(0.0, unit="m"),
        sc.scalar(100.0, unit="m"),
    )
    make_lut_wf[Position[snx.NXsource, SampleRun]] = fakes.source_position()
    lut = make_lut_wf.compute(time_of_flight.TimeOfFlightLookupTable)
    lut.save_hdf5(filename=tmp_path / "lut.h5")

    wf = GenericTofWorkflow(
        tof_lut_provider=TofLutProvider.FILE,
        run_types=[SampleRun],
        monitor_types=[],
    )
    wf[CalibratedBeamline[SampleRun]] = calibrated_beamline
    wf[NeXusData[snx.NXdetector, SampleRun]] = nexus_data
    wf[time_of_flight.TimeOfFlightLookupTableFilename] = (
        tmp_path / "lut.h5"
    ).as_posix()

    loaded_lut = wf.compute(time_of_flight.TimeOfFlightLookupTable)
    assert_identical(lut, loaded_lut)

    detector = wf.compute(time_of_flight.DetectorTofData[SampleRun])
    assert 'tof' in detector.bins.coords
