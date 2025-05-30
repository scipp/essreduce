# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from collections.abc import Mapping

import scipp as sc
import scippnexus as snx
from scippneutron.chopper import DiskChopper

from ..nexus.types import DiskChoppers, Position, SampleRun
from .types import NumberOfSimulatedNeutrons, SimulationResults


def simulate_beamline(
    *,
    choppers: Mapping[str, DiskChopper],
    source_position: sc.Variable,
    neutrons: int = 1_000_000,
    pulses: int = 1,
    seed: int | None = None,
    facility: str = 'ess',
) -> SimulationResults:
    """
    Simulate a pulse of neutrons propagating through a chopper cascade using the
    ``tof`` package (https://tof.readthedocs.io).

    Parameters
    ----------
    choppers:
        A dict of DiskChopper objects representing the choppers in the beamline. See
        https://scipp.github.io/scippneutron/user-guide/chopper/processing-nexus-choppers.html#Build-DiskChopper
        for more information.
    source_position:
        A scalar variable with ``dtype=vector3`` that defines the source position.
        Must be in the same coordinate system as the choppers' axle positions.
    neutrons:
        Number of neutrons to simulate.
    pulses:
        Number of pulses to simulate.
    seed:
        Seed for the random number generator used in the simulation.
    facility:
        Facility where the experiment is performed.
    """
    import tof

    tof_choppers = [
        tof.Chopper(
            frequency=abs(ch.frequency),
            direction=tof.AntiClockwise
            if (ch.frequency.value > 0.0)
            else tof.Clockwise,
            open=ch.slit_begin,
            close=ch.slit_end,
            phase=abs(ch.phase),
            distance=sc.norm(
                ch.axle_position - source_position.to(unit=ch.axle_position.unit)
            ),
            name=name,
        )
        for name, ch in choppers.items()
    ]
    source = tof.Source(facility=facility, neutrons=neutrons, pulses=pulses, seed=seed)
    if not tof_choppers:
        events = source.data.squeeze().flatten(to='event')
        return SimulationResults(
            time_of_arrival=events.coords["birth_time"],
            speed=events.coords["speed"],
            wavelength=events.coords["wavelength"],
            weight=events.data,
            distance=0.0 * sc.units.m,
        )
    model = tof.Model(source=source, choppers=tof_choppers)
    results = model.run()
    # Find name of the furthest chopper in tof_choppers
    furthest_chopper = max(tof_choppers, key=lambda c: c.distance)
    events = results[furthest_chopper.name].data.squeeze().flatten(to='event')
    events = events[
        ~(events.masks["blocked_by_others"] | events.masks["blocked_by_me"])
    ]
    return SimulationResults(
        time_of_arrival=events.coords["toa"],
        speed=events.coords["speed"],
        wavelength=events.coords["wavelength"],
        weight=events.data,
        distance=furthest_chopper.distance,
    )


def simulate_chopper_cascade_using_tof(
    choppers: DiskChoppers[SampleRun],
    neutrons: NumberOfSimulatedNeutrons,
    source_position: Position[snx.NXsource, SampleRun],
) -> SimulationResults:
    """
    Simulate neutrons traveling through the chopper cascade using the ``tof`` package.

    Parameters
    ----------
    choppers:
        Chopper settings.
    neutrons:
        Number of neutrons to simulate.
    source_position:
        Position of the source.
    """
    return simulate_beamline(
        choppers=choppers, neutrons=neutrons, source_position=source_position
    )
