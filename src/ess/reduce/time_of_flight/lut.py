# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Utilities for computing time-of-flight lookup tables from neutron simulations.
"""

import math
from dataclasses import dataclass
from typing import NewType

import numpy as np
import sciline as sl
import scipp as sc

from ..nexus.types import AnyRun, DiskChoppers
from .types import TofLookupTable


@dataclass
class BeamlineComponentReading:
    """
    Reading at a given position along the beamline from a time-of-flight simulation.
    The data (apart from ``distance``) should be flat lists (1d arrays) of length N
    where N is the number of neutrons, containing the properties of the neutrons in the
    simulation.

    Parameters
    ----------
    time_of_arrival:
        Time of arrival of the neutrons at the position where the events were recorded
        (1d array of size N).
    wavelength:
        Wavelength of the neutrons (1d array of size N).
    weight:
        Weight/probability of the neutrons (1d array of size N).
    distance:
        Distance from the source to the position where the events were recorded
        (single value; we assume all neutrons were recorded at the same position).
        For a ``tof`` simulation, this is just the position of the detector where the
        events are recorded. For a ``McStas`` simulation, this is the distance between
        the source and the event monitor.
    """

    time_of_arrival: sc.Variable
    wavelength: sc.Variable
    weight: sc.Variable
    distance: sc.Variable

    def __post_init__(self):
        self.speed = (sc.constants.h / sc.constants.m_n) / self.wavelength


@dataclass
class SimulationResults:
    """
    Results of a time-of-flight simulation used to create a lookup table.
    It should contain readings at various positions along the beamline, e.g., at
    the source and after each chopper.
    It also contains the chopper parameters used in the simulation, so it can be
    determined if this simulation is compatible with a given experiment.

    Parameters
    ----------
    readings:
        A dict of :class:`BeamlineComponentReading` objects representing the readings at
        various positions along the beamline.
    choppers:
        The chopper parameters used in the simulation (if any).
    """

    readings: dict[str, BeamlineComponentReading]
    choppers: DiskChoppers[AnyRun] | None = None


NumberOfSimulatedNeutrons = NewType("NumberOfSimulatedNeutrons", int)
"""
Number of neutrons simulated in the simulation that is used to create the lookup table.
This is typically a large number, e.g., 1e6 or 1e7.
"""

LtotalRange = NewType("LtotalRange", tuple[sc.Variable, sc.Variable])
"""
Range (min, max) of the total length of the flight path from the source to the detector.
This is used to create the lookup table to compute the neutron time-of-flight.
Note that the resulting table will extend slightly beyond this range, as the supplied
range is not necessarily a multiple of the distance resolution.

Note also that the range of total flight paths is supplied manually to the workflow
instead of being read from the input data, as it allows us to compute the expensive part
of the workflow in advance (the lookup table) and does not need to be repeated for each
run, or for new data coming in in the case of live data collection.
"""

DistanceResolution = NewType("DistanceResolution", sc.Variable)
"""
Step size of the distance axis in the lookup table.
Should be a single scalar value with a unit of length.
This is typically of the order of 1-10 cm.
"""

TimeResolution = NewType("TimeResolution", sc.Variable)
"""
Step size of the event_time_offset axis in the lookup table.
This is basically the 'time-of-flight' resolution of the detector.
Should be a single scalar value with a unit of time.
This is typically of the order of 0.1-0.5 ms.

Since the event_time_offset range needs to span exactly one pulse period, the final
resolution in the lookup table will be at least the supplied value here, but may be
smaller if the pulse period is not an integer multiple of the time resolution.
"""


LookupTableRelativeErrorThreshold = NewType("LookupTableRelativeErrorThreshold", float)
"""
Threshold for the relative standard deviation (coefficient of variation) of the
projected time-of-flight above which values are masked.
"""

PulsePeriod = NewType("PulsePeriod", sc.Variable)
"""
Period of the source pulses, i.e., time between consecutive pulse starts.
"""

PulseStride = NewType("PulseStride", int)
"""
Stride of used pulses. Usually 1, but may be a small integer when pulse-skipping.
"""

SourcePosition = NewType("SourcePosition", sc.Variable)
"""
Position of the neutron source in the coordinate system of the choppers.
"""

SimulationSeed = NewType("SimulationSeed", int | None)
"""Seed for the random number generator used in the simulation.
"""

SimulationFacility = NewType("SimulationFacility", str)
"""
Facility where the experiment is performed, e.g., 'ess'.
"""


def _mask_large_uncertainty(table: sc.DataArray, error_threshold: float):
    """
    Mask regions with large uncertainty with NaNs.
    The values are modified in place in the input table.

    Parameters
    ----------
    table:
        Lookup table with time-of-flight as a function of distance and time-of-arrival.
    error_threshold:
        Threshold for the relative standard deviation (coefficient of variation) of the
        projected time-of-flight above which values are masked.
    """
    # Finally, mask regions with large uncertainty with NaNs.
    relative_error = sc.stddevs(table.data) / sc.values(table.data)
    mask = relative_error > sc.scalar(error_threshold)
    # Use numpy for indexing as table is 2D
    table.values[mask.values] = np.nan


def _compute_mean_tof(
    simulation: BeamlineComponentReading,
    distance: sc.Variable,
    time_bins: sc.Variable,
    time_unit: str,
    frame_period: sc.Variable,
    time_bins_half_width: sc.Variable,
) -> sc.DataArray:
    """
    Compute the mean time-of-flight inside event_time_offset bins for a given range of
    distances.

    Parameters
    ----------
    simulation:
        Results of a time-of-flight simulation used to create a lookup table.
    distance:
        Distance where table is computed.
    time_bins:
        Bin edges for the event_time_offset axis in the lookup table.
    time_unit:
        Unit of the event_time_offset axis.
    frame_period:
        Period of the source pulses, i.e., time between consecutive pulse starts.
    time_bins_half_width:
        Half width of the time bins in the event_time_offset axis.
    """
    travel_length = distance - simulation.distance.to(unit=distance.unit)
    # Compute arrival and flight times for all neutrons
    toas = simulation.time_of_arrival + (travel_length / simulation.speed).to(
        unit=time_unit, copy=False
    )
    # dist = distances + simulation_distance
    tofs = distance / simulation.speed

    data = sc.DataArray(
        data=simulation.weight,
        coords={"toa": toas, "tof": tofs.to(unit=time_unit, copy=False)},
    )

    # Add the event_time_offset coordinate, wrapped to the frame_period
    data.coords['event_time_offset'] = data.coords['toa'] % frame_period

    # Because we staggered the mesh by half a bin width, we want the values above
    # the last bin edge to wrap around to the first bin.
    # Technically, those values should end up between -0.5*bin_width and 0, but
    # a simple modulo also works here because even if they end up between 0 and
    # 0.5*bin_width, we are (below) computing the mean between -0.5*bin_width and
    # 0.5*bin_width and it yields the same result.
    # data.coords['event_time_offset'] %= pulse_period - time_bins_half_width
    data.coords['event_time_offset'] %= frame_period - time_bins_half_width

    binned = data.bin(event_time_offset=time_bins)
    binned_sum = binned.bins.sum()
    # Weighted mean of tof inside each bin
    mean_tof = (binned.bins.data * binned.bins.coords["tof"]).bins.sum() / binned_sum
    # Compute the variance of the tofs to track regions with large uncertainty
    variance = (
        binned.bins.data * (binned.bins.coords["tof"] - mean_tof) ** 2
    ).bins.sum() / binned_sum

    mean_tof.variances = variance.values
    return mean_tof


def make_tof_lookup_table(
    simulation: SimulationResults,
    ltotal_range: LtotalRange,
    distance_resolution: DistanceResolution,
    time_resolution: TimeResolution,
    pulse_period: PulsePeriod,
    pulse_stride: PulseStride,
    error_threshold: LookupTableRelativeErrorThreshold,
) -> TofLookupTable:
    """
    Compute a lookup table for time-of-flight as a function of distance and
    time-of-arrival.

    Parameters
    ----------
    simulation:
        Results of a time-of-flight simulation used to create a lookup table.
        The results should be a flat table with columns for time-of-arrival,
        wavelength, and weight.
    ltotal_range:
        Range of total flight path lengths from the source to the detector.
    distance_resolution:
        Resolution of the distance axis in the lookup table.
    time_resolution:
        Resolution of the time-of-arrival axis in the lookup table. Must be an integer.
    pulse_period:
        Period of the source pulses, i.e., time between consecutive pulse starts.
    pulse_stride:
        Stride of used pulses. Usually 1, but may be a small integer when
        pulse-skipping.
    error_threshold:
        Threshold for the relative standard deviation (coefficient of variation) of the
        projected time-of-flight above which values are masked.

    Notes
    -----

    Below are some details about the binning and wrapping around frame period in the
    time dimension.

    We have some simulated ``toa`` (events) from a Tof/McStas simulation.
    Those are absolute ``toa``, unwrapped.
    First we compute the usual ``event_time_offset = toa % frame_period``.

    Now, we want to ensure periodic boundaries. If we make a bin centered around 0,
    and a bin centered around 71ms: the first bin will use events between 0 and
    ``0.5 * dt`` (where ``dt`` is the bin width).
    The last bin will use events between ``frame_period - 0.5*dt`` and
    ``frame_period + 0.5 * dt``. So when we compute the mean inside those two bins,
    they will not yield the same results.
    It is as if the first bin is missing the events it should have between
    ``-0.5 * dt`` and 0 (because of the modulo we computed above).

    To fix this, we do not make a last bin around 71ms (the bins stop at
    ``frame_period - 0.5*dt``). Instead, we compute modulo a second time,
    but this time using ``event_time_offset %= (frame_period - 0.5*dt)``.
    (we cannot directly do ``event_time_offset = toa % (frame_period - 0.5*dt)`` in a
    single step because it would introduce a gradual shift,
    as the pulse number increases).

    This second modulo effectively takes all the events that would have gone in the
    last bin (between ``frame_period - 0.5*dt`` and ``frame_period``) and puts them in
    the first bin. Instead of placing them between ``-0.5*dt`` and 0,
    it places them between 0 and ``0.5*dt``, but this does not really matter,
    because we then take the mean inside the first bin.
    Whether the events are on the left or right side of zero does not matter.

    Finally, we make a copy of the left edge, and append it to the right of the table,
    thus ensuring that the values on the right edge are strictly the same as on the
    left edge.
    """
    distance_unit = "m"
    time_unit = "us"
    res = distance_resolution.to(unit=distance_unit)
    pulse_period = pulse_period.to(unit=time_unit)
    frame_period = pulse_period * pulse_stride

    min_dist = ltotal_range[0].to(unit=distance_unit)
    max_dist = ltotal_range[1].to(unit=distance_unit)

    # We need to bin the data below, to compute the weighted mean of the wavelength.
    # This results in data with bin edges.
    # However, the 2d interpolator expects bin centers.
    # We want to give the 2d interpolator a table that covers the requested range,
    # hence we need to extend the range by at least half a resolution in each direction.
    # Then, we make the choice that the resolution in distance is the quantity that
    # should be preserved. Because the difference between min and max distance is
    # not necessarily an integer multiple of the resolution, we need to add a pad to
    # ensure that the last bin is not cut off. We want the upper edge to be higher than
    # the maximum distance, hence we pad with an additional 1.5 x resolution.
    pad = 2.0 * res
    distances = sc.arange('distance', min_dist - pad, max_dist + pad, res)

    # Create some time bins for event_time_offset.
    # We want our final table to strictly cover the range [0, frame_period].
    # However, binning the data associates mean values inside the bins to the bin
    # centers. Instead, we stagger the mesh by half a bin width so we are computing
    # values for the final mesh edges (the bilinear interpolation needs values on the
    # edges/corners).
    nbins = int(frame_period / time_resolution.to(unit=time_unit)) + 1
    time_bins = sc.linspace(
        'event_time_offset', 0.0, frame_period.value, nbins + 1, unit=pulse_period.unit
    )
    time_bins_half_width = 0.5 * (time_bins[1] - time_bins[0])
    time_bins -= time_bins_half_width

    # Sort simulation readings by reverse distance
    sorted_simulation_results = sorted(
        simulation.readings.values(), key=lambda x: x.distance.value, reverse=True
    )

    pieces = []
    # To avoid large RAM usage, and having to split the distances into chunks
    # according to which component reading to use, we simply loop over distances one
    # by one here.
    for dist in distances:
        # Find the correct simulation reading
        simulation_reading = None
        for reading in sorted_simulation_results:
            if dist.value >= reading.distance.to(unit=dist.unit).value:
                simulation_reading = reading
                break
        if simulation_reading is None:
            raise ValueError(
                "No simulation reading found for distance "
                f"{dist.value} {dist.unit}. "
                "It is likely lower than the simulation reading closest to the source."
            )

        pieces.append(
            _compute_mean_tof(
                simulation=simulation_reading,
                distance=dist,
                time_bins=time_bins,
                time_unit=time_unit,
                frame_period=frame_period,
                time_bins_half_width=time_bins_half_width,
            )
        )

    table = sc.concat(pieces, 'distance')
    table.coords["event_time_offset"] = sc.midpoints(table.coords["event_time_offset"])

    # Copy the left edge to the right to create periodic boundary conditions
    table = sc.DataArray(
        data=sc.concat(
            [table.data, table.data['event_time_offset', 0]], dim='event_time_offset'
        ),
        coords={
            "distance": distances,
            "event_time_offset": sc.concat(
                [table.coords["event_time_offset"], frame_period],
                dim='event_time_offset',
            ),
        },
    )

    # In-place masking for better performance
    _mask_large_uncertainty(table, error_threshold)

    return TofLookupTable(
        array=table,
        pulse_period=pulse_period,
        pulse_stride=pulse_stride,
        distance_resolution=table.coords["distance"][1] - table.coords["distance"][0],
        time_resolution=table.coords["event_time_offset"][1]
        - table.coords["event_time_offset"][0],
        error_threshold=error_threshold,
        choppers=sc.DataGroup(
            {k: sc.DataGroup(ch.as_dict()) for k, ch in simulation.choppers.items()}
        )
        if simulation.choppers is not None
        else None,
    )


def _to_component_reading(component):
    events = component.data.squeeze().flatten(to='event')
    sel = sc.full(value=True, sizes=events.sizes)
    for key in {'blocked_by_others', 'blocked_by_me'} & set(events.masks.keys()):
        sel &= ~events.masks[key]
    events = events[sel]
    # If the component is a source, use 'birth_time' as 'toa'
    toa = (
        events.coords["toa"] if "toa" in events.coords else events.coords["birth_time"]
    )
    return BeamlineComponentReading(
        time_of_arrival=toa,
        wavelength=events.coords["wavelength"],
        weight=events.data,
        distance=component.distance,
    )


def simulate_chopper_cascade_using_tof(
    choppers: DiskChoppers[AnyRun],
    source_position: SourcePosition,
    neutrons: NumberOfSimulatedNeutrons,
    pulse_stride: PulseStride,
    seed: SimulationSeed,
    facility: SimulationFacility,
) -> SimulationResults:
    """
    Simulate a pulse of neutrons propagating through a chopper cascade using the
    ``tof`` package (https://scipp.github.io/tof).

    Parameters
    ----------
    choppers:
        A dict of DiskChopper objects representing the choppers in the beamline. See
        https://scipp.github.io/scippneutron/user-guide/chopper/processing-nexus-choppers.html
        for more information.
    source_position:
        A scalar variable with ``dtype=vector3`` that defines the source position.
        Must be in the same coordinate system as the choppers' axle positions.
    neutrons:
        Number of neutrons to simulate.
    pulse_stride:
        The pulse strinde; we need to simulate at least enough pulses to cover the
        requested stride.
    seed:
        Seed for the random number generator used in the simulation.
    facility:
        Facility where the experiment is performed.
    """
    import tof

    tof_choppers = []
    for name, ch in choppers.items():
        chop = tof.Chopper.from_diskchopper(ch, name=name)
        chop.distance = sc.norm(
            ch.axle_position - source_position.to(unit=ch.axle_position.unit)
        )
        tof_choppers.append(chop)

    source = tof.Source(
        facility=facility, neutrons=neutrons, pulses=pulse_stride, seed=seed
    )
    sim_readings = {"source": _to_component_reading(source)}
    if not tof_choppers:
        return SimulationResults(readings=sim_readings, choppers=None)
    model = tof.Model(source=source, choppers=tof_choppers)
    results = model.run()
    for name, ch in results.choppers.items():
        sim_readings[name] = _to_component_reading(ch)
    return SimulationResults(readings=sim_readings, choppers=choppers)


def TofLookupTableWorkflow():
    """
    Create a workflow for computing a time-of-flight lookup table from a
    simulation of neutrons propagating through a chopper cascade.
    """
    wf = sl.Pipeline(
        (make_tof_lookup_table, simulate_chopper_cascade_using_tof),
        params={
            PulsePeriod: 1.0 / sc.scalar(14.0, unit="Hz"),
            PulseStride: 1,
            DistanceResolution: sc.scalar(0.1, unit="m"),
            TimeResolution: sc.scalar(250.0, unit='us'),
            LookupTableRelativeErrorThreshold: 0.1,
            NumberOfSimulatedNeutrons: 1_000_000,
            SimulationSeed: None,
            SimulationFacility: 'ess',
        },
    )
    return wf
