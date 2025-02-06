# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from functools import partial

import numpy as np
import pytest
import scipp as sc
import tof as tof_pkg
from scippneutron.chopper import DiskChopper
from scippneutron.conversion.graph.beamline import beamline as beamline_graph
from scippneutron.conversion.graph.tof import elastic as elastic_graph

from ess.reduce import time_of_flight
from ess.reduce.time_of_flight import fakes

sl = pytest.importorskip("sciline")


def dream_choppers():
    psc1 = DiskChopper(
        frequency=sc.scalar(14.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(286 - 180, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 6.145], unit="m"),
        slit_begin=sc.array(
            dims=["cutout"],
            values=[-1.23, 70.49, 84.765, 113.565, 170.29, 271.635, 286.035, 301.17],
            unit="deg",
        ),
        slit_end=sc.array(
            dims=["cutout"],
            values=[1.23, 73.51, 88.035, 116.835, 175.31, 275.565, 289.965, 303.63],
            unit="deg",
        ),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    psc2 = DiskChopper(
        frequency=sc.scalar(-14.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(-236, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 6.155], unit="m"),
        slit_begin=sc.array(
            dims=["cutout"],
            values=[-1.23, 27.0, 55.8, 142.385, 156.765, 214.115, 257.23, 315.49],
            unit="deg",
        ),
        slit_end=sc.array(
            dims=["cutout"],
            values=[1.23, 30.6, 59.4, 145.615, 160.035, 217.885, 261.17, 318.11],
            unit="deg",
        ),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    oc = DiskChopper(
        frequency=sc.scalar(14.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(297 - 180 - 90, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 6.174], unit="m"),
        slit_begin=sc.array(dims=["cutout"], values=[-27.6 * 0.5], unit="deg"),
        slit_end=sc.array(dims=["cutout"], values=[27.6 * 0.5], unit="deg"),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    bcc = DiskChopper(
        frequency=sc.scalar(112.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(240 - 180, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 9.78], unit="m"),
        slit_begin=sc.array(dims=["cutout"], values=[-36.875, 143.125], unit="deg"),
        slit_end=sc.array(dims=["cutout"], values=[36.875, 216.875], unit="deg"),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    t0 = DiskChopper(
        frequency=sc.scalar(28.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(280 - 180, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 13.05], unit="m"),
        slit_begin=sc.array(dims=["cutout"], values=[-314.9 * 0.5], unit="deg"),
        slit_end=sc.array(dims=["cutout"], values=[314.9 * 0.5], unit="deg"),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    return {"psc1": psc1, "psc2": psc2, "oc": oc, "bcc": bcc, "t0": t0}


def dream_choppers_with_frame_overlap():
    out = dream_choppers()
    out["bcc"] = DiskChopper(
        frequency=sc.scalar(112.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(240 - 180, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 9.78], unit="m"),
        slit_begin=sc.array(dims=["cutout"], values=[-36.875, 143.125], unit="deg"),
        slit_end=sc.array(dims=["cutout"], values=[56.875, 216.875], unit="deg"),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )
    return out


@pytest.fixture(scope="module")
def simulation_dream_choppers():
    return time_of_flight.simulate_beamline(
        choppers=dream_choppers(), neutrons=100_000, seed=432
    )


@pytest.mark.parametrize(
    "ltotal",
    [
        sc.array(dims=["detector_number"], values=[77.675], unit="m"),
        sc.array(dims=["detector_number"], values=[77.675, 76.5], unit="m"),
        sc.array(
            dims=["y", "x"],
            values=[[77.675, 76.1, 78.05], [77.15, 77.3, 77.675]],
            unit="m",
        ),
    ],
)
@pytest.mark.parametrize("time_offset_unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("distance_unit", ["m", "mm"])
def test_dream_wfm(simulation_dream_choppers, ltotal, time_offset_unit, distance_unit):
    monitors = {
        f"detector{i}": ltot for i, ltot in enumerate(ltotal.flatten(to="detector"))
    }

    # Create some neutron events
    beamline = fakes.FakeBeamline(
        choppers=dream_choppers(),
        monitors=monitors,
        run_length=sc.scalar(1 / 14, unit="s") * 4,
        events_per_pulse=10_000,
        seed=77,
    )

    raw = sc.concat(
        [beamline.get_monitor(key)[0].squeeze() for key in monitors.keys()],
        dim="detector",
    ).fold(dim="detector", sizes=ltotal.sizes)

    # Convert the time offset to the unit requested by the test
    raw.bins.coords["event_time_offset"] = raw.bins.coords["event_time_offset"].to(
        unit=time_offset_unit, copy=False
    )
    # Convert the distance to the unit requested by the test
    raw.coords["Ltotal"] = raw.coords["Ltotal"].to(unit=distance_unit, copy=False)

    # Save reference data
    ref = beamline.get_monitor(next(iter(monitors)))[1].squeeze()
    ref = sc.sort(ref, key='id')

    pl = sl.Pipeline(
        time_of_flight.providers(), params=time_of_flight.default_parameters()
    )

    pl[time_of_flight.RawData] = raw
    pl[time_of_flight.SimulationResults] = simulation_dream_choppers
    pl[time_of_flight.LtotalRange] = ltotal.min(), ltotal.max()

    tofs = pl.compute(time_of_flight.TofData)

    # Convert to wavelength
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    wavs = tofs.transform_coords("wavelength", graph=graph)

    for da in wavs.flatten(to='pixel'):
        x = sc.sort(da.value, key='id')
        # print(x)
        diff = abs(
            (x.coords["wavelength"] - ref.coords["wavelength"])
            / ref.coords["wavelength"]
        )
        assert np.nanpercentile(diff.values, 100) < 0.02
        assert sc.isclose(ref.data.sum(), da.data.sum(), rtol=sc.scalar(1.0e-3))

    # print(wav_wfm)

    # # Compare the computed wavelengths to the true wavelengths
    # result = wav_wfm.flatten(to="detector")

    # for j in range(len(result)):
    #     computed_wavelengths = result[j].values.coords["wavelength"]
    #     assert sc.allclose(
    #         computed_wavelengths,
    #         true_wavelengths["pulse", i],
    #         rtol=sc.scalar(1e-02),
    #     )

    # # # Compare the computed wavelengths to the true wavelengths
    # # for i in range(npulses):
    # #     result = wav_wfm["pulse", i].flatten(to="detector")
    # #     for j in range(len(result)):
    # #         computed_wavelengths = result[j].values.coords["wavelength"]
    # #         assert sc.allclose(
    # #             computed_wavelengths,
    # #             true_wavelengths["pulse", i],
    # #             rtol=sc.scalar(1e-02),
    # #         )


@pytest.fixture(scope="module")
def simulation_dream_choppers_time_overlap():
    return time_of_flight.simulate_beamline(
        choppers=dream_choppers_with_frame_overlap(), neutrons=100_000, seed=432
    )


@pytest.mark.parametrize(
    "ltotal",
    [
        sc.array(dims=["detector_number"], values=[77.675], unit="m"),
        sc.array(dims=["detector_number"], values=[77.675, 76.5], unit="m"),
        sc.array(
            dims=["y", "x"],
            values=[[77.675, 76.1, 78.05], [77.15, 77.3, 77.675]],
            unit="m",
        ),
    ],
)
@pytest.mark.parametrize("time_offset_unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("distance_unit", ["m", "mm"])
def test_dream_wfm_with_subframe_time_overlap(
    simulation_dream_choppers_time_overlap,
    ltotal,
    time_offset_unit,
    distance_unit,
):
    monitors = {
        f"detector{i}": ltot for i, ltot in enumerate(ltotal.flatten(to="detector"))
    }

    # Create some neutron events
    beamline = fakes.FakeBeamline(
        choppers=dream_choppers_with_frame_overlap(),
        monitors=monitors,
        run_length=sc.scalar(1 / 14, unit="s") * 4,
        events_per_pulse=10_000,
        seed=88,
    )

    raw = sc.concat(
        [beamline.get_monitor(key)[0].squeeze() for key in monitors.keys()],
        dim="detector",
    ).fold(dim="detector", sizes=ltotal.sizes)

    # Convert the time offset to the unit requested by the test
    raw.bins.coords["event_time_offset"] = raw.bins.coords["event_time_offset"].to(
        unit=time_offset_unit, copy=False
    )
    # Convert the distance to the unit requested by the test
    raw.coords["Ltotal"] = raw.coords["Ltotal"].to(unit=distance_unit, copy=False)

    # Save reference data
    ref = beamline.get_monitor(next(iter(monitors)))[1].squeeze()
    ref = sc.sort(ref, key='id')

    pl = sl.Pipeline(
        time_of_flight.providers(), params=time_of_flight.default_parameters()
    )

    pl[time_of_flight.RawData] = raw
    pl[time_of_flight.SimulationResults] = simulation_dream_choppers_time_overlap
    pl[time_of_flight.LtotalRange] = ltotal.min(), ltotal.max()
    pl[time_of_flight.LookupTableRelativeErrorThreshold] = 0.01

    tofs = pl.compute(time_of_flight.TofData)

    # Convert to wavelength
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    wavs = tofs.transform_coords("wavelength", graph=graph)

    for da in wavs.flatten(to='pixel'):
        x = sc.sort(da.value, key='id')
        sel = sc.isfinite(x.coords["wavelength"])
        y = ref.coords["wavelength"][sel]
        diff = abs((x.coords["wavelength"][sel] - y) / y)
        assert np.nanpercentile(diff.values, 100) < 0.02
        sum_wfm = da.hist(wavelength=100).data.sum()
        sum_ref = ref.hist(wavelength=100).data.sum()
        # Verify that we lost some neutrons that were in the overlapping region
        assert sum_wfm < sum_ref
        assert sum_wfm > sum_ref * 0.9


@pytest.fixture(scope="module")
def simulation_v20_choppers():
    return time_of_flight.simulate_beamline(
        choppers=fakes.wfm_choppers(), neutrons=300_000, seed=432
    )


@pytest.mark.parametrize("npulses", [1, 2])
@pytest.mark.parametrize(
    "ltotal",
    [
        sc.array(dims=["detector_number"], values=[26.0], unit="m"),
        sc.array(dims=["detector_number"], values=[26.0, 25.5], unit="m"),
        sc.array(
            dims=["y", "x"], values=[[26.0, 25.1, 26.33], [25.9, 26.0, 25.7]], unit="m"
        ),
    ],
)
@pytest.mark.parametrize("time_offset_unit", ["s", "ms", "us", "ns"])
@pytest.mark.parametrize("distance_unit", ["m", "mm"])
def test_v20_compute_wavelengths_from_wfm(
    simulation_v20_choppers, npulses, ltotal, time_offset_unit, distance_unit
):
    monitors = {
        f"detector{i}": ltot for i, ltot in enumerate(ltotal.flatten(to="detector"))
    }

    # Create some neutron events
    wavelengths = sc.array(
        dims=["event"], values=[2.75, 4.2, 5.4, 6.5, 7.6, 8.75], unit="angstrom"
    )
    birth_times = sc.full(sizes=wavelengths.sizes, value=1.5, unit="ms")
    ess_beamline = fakes.FakeBeamline(
        choppers=fakes.wfm_choppers(),
        monitors=monitors,
        run_length=sc.scalar(1 / 14, unit="s") * npulses,
        events_per_pulse=len(wavelengths),
        source=partial(
            tof_pkg.Source.from_neutrons,
            birth_times=birth_times,
            wavelengths=wavelengths,
            frequency=sc.scalar(14.0, unit="Hz"),
        ),
    )

    # Save the true wavelengths for later
    true_wavelengths = ess_beamline.source.data.coords["wavelength"]

    raw_data = sc.concat(
        [ess_beamline.get_monitor(key)[0] for key in monitors.keys()],
        dim="detector",
    ).fold(dim="detector", sizes=ltotal.sizes)

    # Convert the time offset to the unit requested by the test
    raw_data.bins.coords["event_time_offset"] = raw_data.bins.coords[
        "event_time_offset"
    ].to(unit=time_offset_unit, copy=False)

    raw_data.coords["Ltotal"] = ltotal.to(unit=distance_unit, copy=False)

    # Verify that all 6 neutrons made it through the chopper cascade
    assert sc.identical(
        raw_data.bins.concat("pulse").hist().data,
        sc.array(
            dims=["detector"],
            values=[len(wavelengths) * npulses] * len(monitors),
            unit="counts",
            dtype="float64",
        ).fold(dim="detector", sizes=ltotal.sizes),
    )

    # Set up the workflow
    workflow = sl.Pipeline(
        time_of_flight.providers(), params=time_of_flight.default_parameters()
    )

    workflow[time_of_flight.RawData] = raw_data
    workflow[time_of_flight.SimulationResults] = simulation_v20_choppers
    workflow[time_of_flight.LtotalRange] = ltotal.min(), ltotal.max()

    # Compute time-of-flight
    tofs = workflow.compute(time_of_flight.TofData)
    assert {dim: tofs.sizes[dim] for dim in ltotal.sizes} == ltotal.sizes

    # Convert to wavelength
    graph = {**beamline(scatter=False), **elastic("tof")}
    wav_wfm = tofs.transform_coords("wavelength", graph=graph)

    # Compare the computed wavelengths to the true wavelengths
    for i in range(npulses):
        result = wav_wfm["pulse", i].flatten(to="detector")
        for j in range(len(result)):
            computed_wavelengths = result[j].values.coords["wavelength"]
            assert sc.allclose(
                computed_wavelengths,
                true_wavelengths["pulse", i],
                rtol=sc.scalar(1e-02),
            )
