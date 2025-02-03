# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
A fake time-of-flight neutron beamline for documentation and testing.

This provides detector event data in a structure as typically provided in a NeXus file,
with event_time_offset and event_time_zero information.
"""

from collections.abc import Callable

import numpy as np
import scipp as sc
from scippneutron.chopper import DiskChopper


class FakeBeamline:
    def __init__(
        self,
        choppers: dict[str, DiskChopper],
        monitors: dict[str, sc.Variable],
        run_length: sc.Variable,
        events_per_pulse: int = 200000,
        source: Callable | None = None,
    ):
        import math

        import tof as tof_pkg
        from tof.facilities.ess_pulse import pulse

        self.frequency = pulse.frequency
        self.npulses = math.ceil((run_length * self.frequency).to(unit="").value)
        self.events_per_pulse = events_per_pulse

        # Create a source
        if source is None:
            self.source = tof_pkg.Source(
                facility="ess", neutrons=self.events_per_pulse, pulses=self.npulses
            )
        else:
            self.source = source(pulses=self.npulses)

        # Convert the choppers to tof.Chopper
        self.choppers = [
            tof_pkg.Chopper(
                frequency=abs(ch.frequency),
                direction=tof_pkg.AntiClockwise
                if (ch.frequency.value > 0.0)
                else tof_pkg.Clockwise,
                open=ch.slit_begin,
                close=ch.slit_end,
                phase=abs(ch.phase),
                distance=ch.axle_position.fields.z,
                name=name,
            )
            for name, ch in choppers.items()
        ]

        # Add detectors
        self.monitors = [
            tof_pkg.Detector(distance=distance, name=key)
            for key, distance in monitors.items()
        ]

        #  Propagate the neutrons
        self.model = tof_pkg.Model(
            source=self.source, choppers=self.choppers, detectors=self.monitors
        )
        self.model_result = self.model.run()

    def get_monitor(self, name: str) -> sc.DataGroup:
        nx_event_data = self.model_result.to_nxevent_data(name)
        raw_data = self.model_result.detectors[name].data.flatten(to="event")
        # Select only the neutrons that make it to the detector
        raw_data = raw_data[~raw_data.masks["blocked_by_others"]].copy()

        return nx_event_data, raw_data
        # # Create some fake pulse time zero
        # start = sc.datetime("2024-01-01T12:00:00.000000")
        # period = sc.reciprocal(self.frequency)

        # detector = self.model_result.detectors[name]
        # raw_data = detector.data.flatten(to="event")
        # # Select only the neutrons that make it to the detector
        # raw_data = raw_data[~raw_data.masks["blocked_by_others"]].copy()
        # raw_data.coords["Ltotal"] = detector.distance

        # # Format the data in a way that resembles data loaded from NeXus
        # event_data = raw_data.copy(deep=False)
        # dt = period.to(unit="us")
        # event_time_zero = (dt * (event_data.coords["toa"] // dt)).to(dtype=int) + start
        # raw_data.coords["event_time_zero"] = event_time_zero
        # event_data.coords["event_time_zero"] = event_time_zero
        # event_data.coords["event_time_offset"] = (
        #     event_data.coords.pop("toa").to(unit="s") % period
        # )
        # del event_data.coords["tof"]
        # del event_data.coords["speed"]
        # del event_data.coords["time"]
        # del event_data.coords["wavelength"]

        # return (
        #     event_data.group("event_time_zero").rename_dims(event_time_zero="pulse"),
        #     raw_data.group("event_time_zero").rename_dims(event_time_zero="pulse"),
        # )


wfm1_chopper = DiskChopper(
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

wfm2_chopper = DiskChopper(
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

foc1_chopper = DiskChopper(
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

foc2_chopper = DiskChopper(
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

pol_chopper = DiskChopper(
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

pulse_skipping = DiskChopper(
    frequency=sc.scalar(-7.0, unit="Hz"),
    beam_position=sc.scalar(0.0, unit="deg"),
    phase=sc.scalar(0.0, unit="deg"),
    axle_position=sc.vector(value=[0, 0, 30.0], unit="m"),
    slit_begin=sc.array(
        dims=["cutout"],
        values=np.array([40.0]),
        unit="deg",
    ),
    slit_end=sc.array(
        dims=["cutout"],
        values=np.array([140.0]),
        unit="deg",
    ),
    slit_height=sc.scalar(10.0, unit="cm"),
    radius=sc.scalar(30.0, unit="cm"),
)


def wfm_choppers():
    return {
        "wfm1": wfm1_chopper,
        "wfm2": wfm2_chopper,
        "foc1": foc1_chopper,
        "foc2": foc2_chopper,
        "pol": pol_chopper,
    }


def psc_choppers():
    return {
        name: DiskChopper(
            frequency=ch.frequency,
            beam_position=ch.beam_position,
            phase=ch.phase,
            axle_position=ch.axle_position,
            slit_begin=ch.slit_begin[0:1],
            slit_end=ch.slit_end[0:1],
            slit_height=ch.slit_height[0:1],
            radius=ch.radius,
        )
        for name, ch in wfm_choppers().items()
    }
