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
        seed: int | None = None,
        source: Callable | None = None,
        source_position: sc.Variable | None = None,
    ):
        import math

        import tof as tof_pkg
        from tof.facilities.ess_pulse import frequency as ess_frequency

        self.frequency = ess_frequency
        self.npulses = math.ceil((run_length * self.frequency).to(unit="").value)
        self.events_per_pulse = events_per_pulse
        if source_position is None:
            source_position = sc.vector([0, 0, 0], unit='m')

        # Create a source
        if source is None:
            self.source = tof_pkg.Source(
                facility="ess",
                neutrons=self.events_per_pulse,
                pulses=self.npulses,
                seed=seed,
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
                distance=sc.norm(ch.axle_position - source_position),
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
        raw_data = raw_data[~raw_data.masks["blocked_by_others"]].copy()
        return nx_event_data, raw_data


def psc_choppers():
    return {
        "chopper": DiskChopper(
            frequency=sc.scalar(-14.0, unit="Hz"),
            beam_position=sc.scalar(0.0, unit="deg"),
            phase=sc.scalar(-85.0, unit="deg"),
            axle_position=sc.vector(value=[0, 0, 8.0], unit="m"),
            slit_begin=sc.array(dims=["cutout"], values=[0.0], unit="deg"),
            slit_end=sc.array(dims=["cutout"], values=[3.0], unit="deg"),
            slit_height=sc.scalar(10.0, unit="cm"),
            radius=sc.scalar(30.0, unit="cm"),
        )
    }


def pulse_skipping_choppers():
    return {
        "chopper": DiskChopper(
            frequency=sc.scalar(-14.0, unit="Hz"),
            beam_position=sc.scalar(0.0, unit="deg"),
            phase=sc.scalar(-35.0, unit="deg"),
            axle_position=sc.vector(value=[0, 0, 8.0], unit="m"),
            slit_begin=sc.array(dims=["cutout"], values=np.array([0.0]), unit="deg"),
            slit_end=sc.array(dims=["cutout"], values=np.array([33.0]), unit="deg"),
            slit_height=sc.scalar(10.0, unit="cm"),
            radius=sc.scalar(30.0, unit="cm"),
        ),
        "pulse_skipping": DiskChopper(
            frequency=sc.scalar(-7.0, unit="Hz"),
            beam_position=sc.scalar(0.0, unit="deg"),
            phase=sc.scalar(-10.0, unit="deg"),
            axle_position=sc.vector(value=[0, 0, 15.0], unit="m"),
            slit_begin=sc.array(dims=["cutout"], values=np.array([0.0]), unit="deg"),
            slit_end=sc.array(dims=["cutout"], values=np.array([120.0]), unit="deg"),
            slit_height=sc.scalar(10.0, unit="cm"),
            radius=sc.scalar(30.0, unit="cm"),
        ),
    }


def source_position():
    return sc.vector([0, 0, 0], unit='m')
