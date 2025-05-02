# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc
from scippneutron.chopper import DiskChopper
from scippneutron.conversion.graph.beamline import beamline as beamline_graph
from scippneutron.conversion.graph.tof import elastic as elastic_graph

from ess.reduce import time_of_flight
from ess.reduce.nexus.types import DetectorData, SampleRun
from ess.reduce.time_of_flight import fakes

sl = pytest.importorskip("sciline")


def dream_choppers():
    psc1 = DiskChopper(
        frequency=sc.scalar(14.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(286 - 180, unit="deg"),
        axle_position=sc.vector(value=[0, 0, -70.405], unit="m"),
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
        axle_position=sc.vector(value=[0, 0, -70.395], unit="m"),
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
        axle_position=sc.vector(value=[0, 0, -70.376], unit="m"),
        slit_begin=sc.array(dims=["cutout"], values=[-27.6 * 0.5], unit="deg"),
        slit_end=sc.array(dims=["cutout"], values=[27.6 * 0.5], unit="deg"),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    bcc = DiskChopper(
        frequency=sc.scalar(112.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(240 - 180, unit="deg"),
        axle_position=sc.vector(value=[0, 0, -66.77], unit="m"),
        slit_begin=sc.array(dims=["cutout"], values=[-36.875, 143.125], unit="deg"),
        slit_end=sc.array(dims=["cutout"], values=[36.875, 216.875], unit="deg"),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    t0 = DiskChopper(
        frequency=sc.scalar(28.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(280 - 180, unit="deg"),
        axle_position=sc.vector(value=[0, 0, -63.5], unit="m"),
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
        axle_position=sc.vector(value=[0, 0, -66.77], unit="m"),
        slit_begin=sc.array(dims=["cutout"], values=[-36.875, 143.125], unit="deg"),
        slit_end=sc.array(dims=["cutout"], values=[56.875, 216.875], unit="deg"),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )
    return out


def dream_source_position():
    return sc.vector(value=[0, 0, -76.55], unit="m")


@pytest.fixture(scope="module")
def simulation_dream_choppers():
    return time_of_flight.simulate_beamline(
        choppers=dream_choppers(),
        source_position=dream_source_position(),
        neutrons=100_000,
        seed=432,
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
        source_position=dream_source_position(),
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

    pl[DetectorData[SampleRun]] = raw
    pl[time_of_flight.DetectorLtotal[SampleRun]] = ltotal
    pl[time_of_flight.SimulationResults] = simulation_dream_choppers
    pl[time_of_flight.LtotalRange] = ltotal.min(), ltotal.max()

    tofs = pl.compute(time_of_flight.DetectorTofData[SampleRun])

    # Convert to wavelength
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    wavs = tofs.transform_coords("wavelength", graph=graph)

    for da in wavs.flatten(to='pixel'):
        x = sc.sort(da.value, key='id')
        diff = abs(
            (x.coords["wavelength"] - ref.coords["wavelength"])
            / ref.coords["wavelength"]
        )
        assert np.nanpercentile(diff.values, 100) < 0.02
        assert sc.isclose(ref.data.sum(), da.data.sum(), rtol=sc.scalar(1.0e-3))


@pytest.fixture(scope="module")
def simulation_dream_choppers_time_overlap():
    return time_of_flight.simulate_beamline(
        choppers=dream_choppers_with_frame_overlap(),
        source_position=dream_source_position(),
        neutrons=100_000,
        seed=432,
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
        source_position=dream_source_position(),
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

    pl[DetectorData[SampleRun]] = raw
    pl[time_of_flight.SimulationResults] = simulation_dream_choppers_time_overlap
    pl[time_of_flight.DetectorLtotal[SampleRun]] = ltotal
    pl[time_of_flight.LtotalRange] = ltotal.min(), ltotal.max()
    pl[time_of_flight.LookupTableRelativeErrorThreshold] = 0.01

    tofs = pl.compute(time_of_flight.DetectorTofData[SampleRun])

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


def v20_choppers():
    wfm1 = DiskChopper(
        frequency=sc.scalar(-70.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(-47.10, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 6.6], unit="m"),
        slit_begin=sc.array(
            dims=["cutout"],
            values=np.array([83.71, 140.49, 193.26, 242.32, 287.91, 330.3]) + 15.0,
            unit="deg",
        ),
        slit_end=sc.array(
            dims=["cutout"],
            values=np.array([94.7, 155.79, 212.56, 265.33, 314.37, 360.0]) + 15.0,
            unit="deg",
        ),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    wfm2 = DiskChopper(
        frequency=sc.scalar(-70.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(-76.76, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 7.1], unit="m"),
        slit_begin=sc.array(
            dims=["cutout"],
            values=np.array([65.04, 126.1, 182.88, 235.67, 284.73, 330.32]) + 15.0,
            unit="deg",
        ),
        slit_end=sc.array(
            dims=["cutout"],
            values=np.array([76.03, 141.4, 202.18, 254.97, 307.74, 360.0]) + 15.0,
            unit="deg",
        ),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    foc1 = DiskChopper(
        frequency=sc.scalar(-56.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(-62.40, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 8.8], unit="m"),
        slit_begin=sc.array(
            dims=["cutout"],
            values=np.array([74.6, 139.6, 194.3, 245.3, 294.8, 347.2]),
            unit="deg",
        ),
        slit_end=sc.array(
            dims=["cutout"],
            values=np.array([95.2, 162.8, 216.1, 263.1, 310.5, 371.6]),
            unit="deg",
        ),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    foc2 = DiskChopper(
        frequency=sc.scalar(-28.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(-12.27, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 15.9], unit="m"),
        slit_begin=sc.array(
            dims=["cutout"],
            values=np.array([98.0, 154.0, 206.8, 255.0, 299.0, 344.65]),
            unit="deg",
        ),
        slit_end=sc.array(
            dims=["cutout"],
            values=np.array([134.6, 190.06, 237.01, 280.88, 323.56, 373.76]),
            unit="deg",
        ),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )

    pol = DiskChopper(
        frequency=sc.scalar(-14.0, unit="Hz"),
        beam_position=sc.scalar(0.0, unit="deg"),
        phase=sc.scalar(0.0, unit="deg"),
        axle_position=sc.vector(value=[0, 0, 17.0], unit="m"),
        slit_begin=sc.array(
            dims=["cutout"],
            values=np.array([40.0]),
            unit="deg",
        ),
        slit_end=sc.array(
            dims=["cutout"],
            values=np.array([240.0]),
            unit="deg",
        ),
        slit_height=sc.scalar(10.0, unit="cm"),
        radius=sc.scalar(30.0, unit="cm"),
    )
    return {"wfm1": wfm1, "wfm2": wfm2, "foc1": foc1, "foc2": foc2, "pol": pol}


def v20_source_position():
    return sc.vector([0, 0, 0], unit='m')


@pytest.fixture(scope="module")
def simulation_v20_choppers():
    return time_of_flight.simulate_beamline(
        choppers=v20_choppers(),
        source_position=v20_source_position(),
        neutrons=300_000,
        seed=432,
    )


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
    simulation_v20_choppers, ltotal, time_offset_unit, distance_unit
):
    monitors = {
        f"detector{i}": ltot for i, ltot in enumerate(ltotal.flatten(to="detector"))
    }

    # Create some neutron events
    beamline = fakes.FakeBeamline(
        choppers=v20_choppers(),
        monitors=monitors,
        run_length=sc.scalar(1 / 14, unit="s") * 4,
        events_per_pulse=10_000,
        seed=99,
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

    pl[DetectorData[SampleRun]] = raw
    pl[time_of_flight.SimulationResults] = simulation_v20_choppers
    pl[time_of_flight.DetectorLtotal[SampleRun]] = ltotal
    pl[time_of_flight.LtotalRange] = ltotal.min(), ltotal.max()

    tofs = pl.compute(time_of_flight.DetectorTofData[SampleRun])

    # Convert to wavelength
    graph = {**beamline_graph(scatter=False), **elastic_graph("tof")}
    wavs = tofs.transform_coords("wavelength", graph=graph)

    for da in wavs.flatten(to='pixel'):
        x = sc.sort(da.value, key='id')
        diff = abs(
            (x.coords["wavelength"] - ref.coords["wavelength"])
            / ref.coords["wavelength"]
        )
        assert np.nanpercentile(diff.values, 99) < 0.02
        assert sc.isclose(ref.data.sum(), da.data.sum(), rtol=sc.scalar(1.0e-3))
