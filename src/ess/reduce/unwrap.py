# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# @author Simon Heybrock
# @author Neil Vaytet
"""
Time-of-flight workflow for unwrapping the time of arrival of the neutron at the
detector.
This workflow is used to convert raw detector data with event_time_zero and
event_time_offset coordinates to data with a time-of-flight coordinate.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from functools import reduce
from typing import NewType

import numpy as np
import scipp as sc
import tof
from scipp._scipp.core import _bins_no_validate

from scippneutron._utils import elem_unit
from scippneutron.chopper import DiskChopper
from scippneutron.tof.to_events import to_events

import sciline as sl

from .nexus.types import RunType, DetectorData

Facility = NewType('Facility', str)
"""
Facility where the experiment is performed.
"""

Choppers = NewType('Choppers', Mapping[str, DiskChopper])
"""
Choppers used to define the frame parameters.
"""

class Ltotal(sl.Scope[RunType, sc.Variable], sc.Variable):
    """
    Total length of the flight path from the source to the detector.
    """

SimulationSeed = NewType('SimulationSeed', int)
"""
Seed for the random number generator used in the simulation.
"""


NumberOfNeutrons = NewType('NumberOfNeutrons', int)
"""
Number of neutrons to use in the simulation.
"""


@dataclass
class SimulationResults:
    """
    Results of a time-of-flight simulation used to create a lookup table.
    """

    time_of_arrival: sc.Variable
    speed: sc.Variable
    wavelength: sc.Variable
    weight: sc.Variable
    distance: sc.Variable


DistanceResolution = NewType('DistanceResolution', sc.Variable)
"""
Resolution of the distance axis in the lookup table.
"""

class TimeOfFlightLookupTable(sl.Scope[RunType, sc.DataArray], sc.DataArray):
    """
    Lookup table giving time-of-flight as a function of distance and time of arrival.
    """

class MaskedTimeOfFlightLookupTable(sl.Scope[RunType, sc.DataArray], sc.DataArray):
    """
    Lookup table giving time-of-flight as a function of distance and time of arrival, with
    regions of large uncertainty masked out.
    """

LookupTableVarianceThreshold = NewType('LookupTableVarianceThreshold', float)

FramePeriod = NewType('FramePeriod', sc.Variable)
"""
The period of a frame, a (small) integer multiple of the source period.
"""

class UnwrappedTimeOfArrival(sl.Scope[RunType, sc.Variable], sc.Variable):
    """
    Time of arrival of the neutron at the detector, unwrapped at the pulse period.
    """

class FrameAtDetectorStartTime(sl.Scope[RunType, sc.Variable], sc.Variable):
    """
    Time of the start of the frame at the detector.
    """

class UnwrappedTimeOfArrivalMinusStartTime(sl.Scope[RunType, sc.Variable], sc.Variable):
    """
    Time of arrival of the neutron at the detector, unwrapped at the pulse period, minus
    the start time of the frame.
    """

class TimeOfArrivalMinusStartTimeModuloPeriod(sl.Scope[RunType, sc.Variable], sc.Variable):
    """
    Time of arrival of the neutron at the detector minus the start time of the frame,
    modulo the frame period.
    """

class FrameFoldedTimeOfArrival(sl.Scope[RunType, sc.Variable], sc.Variable):
    """
    Time of arrival of the neutron at the detector, folded by the frame period.
    """


PulsePeriod = NewType('PulsePeriod', sc.Variable)
"""
Period of the source pulses, i.e., time between consecutive pulse starts.
"""

PulseStride = NewType('PulseStride', int)
"""
Stride of used pulses. Usually 1, but may be a small integer when pulse-skipping.
"""

# TODO: the pulse stride offset may be different for sample and background runs?
# It should maybe be turned into a generic type?
PulseStrideOffset = NewType('PulseStrideOffset', int)
"""
When pulse-skipping, the offset of the first pulse in the stride.
"""

class FrameFoldedTimeOfArrival(sl.Scope[RunType, sc.Variable], sc.Variable):
    """
    Time of arrival of the neutron at the detector, folded by the frame period.
    """

# RawData = NewType('RawData', sc.DataArray)
# """
# Raw detector data loaded from a NeXus file, e.g., NXdetector containing NXevent_data.
# """

class TofData(sl.Scope[RunType, sc.DataArray], sc.DataArray):
    """
    Detector data with time-of-flight coordinate.
    """

class ReHistogrammedTofData(sl.Scope[RunType, sc.DataArray], sc.DataArray):
    """
    Detector data with time-of-flight coordinate, re-histogrammed.
    """


def pulse_period_from_source(facility: Facility) -> PulsePeriod:
    """
    Return the period of the source pulses, i.e., time between consecutive pulse starts.

    Parameters
    ----------
    facility:
        Facility where the experiment is performed (used to determine the source pulse
        parameters).
    """
    return PulsePeriod(1.0 / tof.facilities[facility].frequency)


def frame_period(pulse_period: PulsePeriod, pulse_stride: PulseStride) -> FramePeriod:
    """
    Return the period of a frame, which is defined by the pulse period times the pulse
    stride.

    Parameters
    ----------
    pulse_period:
        Period of the source pulses, i.e., time between consecutive pulse starts.
    pulse_stride:
        Stride of used pulses. Usually 1, but may be a small integer when
        pulse-skipping.
    """
    return FramePeriod(pulse_period * pulse_stride)


def run_tof_model(
    facility: Facility,
    choppers: Choppers,
    seed: SimulationSeed,
    number_of_neutrons: NumberOfNeutrons,
) -> SimulationResults:
    tof_choppers = [
        tof.Chopper(
            frequency=abs(ch.frequency),
            direction=tof.AntiClockwise
            if (ch.frequency.value > 0.0)
            else tof.Clockwise,
            open=ch.slit_begin,
            close=ch.slit_end,
            phase=abs(ch.phase),
            distance=ch.axle_position.fields.z,
            name=name,
        )
        for name, ch in choppers.items()
    ]
    source = tof.Source(facility=facility, neutrons=number_of_neutrons, seed=seed)
    if not tof_choppers:
        events = source.data.squeeze()
        return SimulationResults(
            time_of_arrival=events.coords['time'],
            speed=events.coords['speed'],
            wavelength=events.coords['wavelength'],
            weight=events.data,
            distance=0.0 * sc.units.m,
        )
    model = tof.Model(source=source, choppers=tof_choppers)
    results = model.run()
    # Find name of the furthest chopper in tof_choppers
    furthest_chopper = max(tof_choppers, key=lambda c: c.distance)
    events = results[furthest_chopper.name].data.squeeze()
    events = events[
        ~(events.masks['blocked_by_others'] | events.masks['blocked_by_me'])
    ]
    return SimulationResults(
        time_of_arrival=events.coords['toa'],
        speed=events.coords['speed'],
        wavelength=events.coords['wavelength'],
        weight=events.data,
        distance=furthest_chopper.distance,
    )


def extract_ltotal(
    da: DetectorData[RunType]) -> Ltotal[RunType]:
    """
    Extract the total length of the flight path from the source to the detector from the
    detector data.

    Parameters
    ----------
    da:
        Raw detector data loaded from a NeXus file, e.g., NXdetector containing
        NXevent_data.
    """
    return Ltotal[RunType](da.coords['Ltotal'])



def compute_tof_lookup_table(
    simulation: SimulationResults,
    ltotal: Ltotal[RunType],
    distance_resolution: DistanceResolution,
) -> TimeOfFlightLookupTable[RunType]:
    simulation_distance = simulation.distance.to(unit=ltotal.unit)
    dist = ltotal - simulation_distance
    res = distance_resolution.to(unit=dist.unit)
    # Add padding to ensure that the lookup table covers the full range of distances
    # because the midpoints of the table edges are used in the 2d grid interpolator.
    min_dist, max_dist = dist.nanmin() - res, dist.nanmax() + res
    ndist = round(((max_dist - min_dist) / res).value) + 1
    distances = sc.linspace(
        'distance', min_dist.value, max_dist.value, ndist, unit=dist.unit
    )

    time_unit = simulation.time_of_arrival.unit
    toas = simulation.time_of_arrival + (distances / simulation.speed).to(
        unit=time_unit, copy=False
    )

    data = sc.DataArray(
        data=sc.broadcast(simulation.weight, sizes=toas.sizes).flatten(to='event'),
        coords={
            'toa': toas.flatten(to='event'),
            'wavelength': sc.broadcast(simulation.wavelength, sizes=toas.sizes).flatten(
                to='event'
            ),
            'distance': sc.broadcast(distances, sizes=toas.sizes).flatten(to='event'),
        },
    )

    # TODO: move toa resolution to workflow parameter
    binned = data.bin(distance=ndist, toa=500)
    # Weighted mean of wavelength inside each bin
    wavelength = (
        binned.bins.data * binned.bins.coords['wavelength']
    ).bins.sum() / binned.bins.sum()
    # Compute the variance of the wavelength to track regions with large uncertainty
    variance = (
        binned.bins.data * (binned.bins.coords['wavelength'] - wavelength) ** 2
    ).bins.sum() / binned.bins.sum()

    # Need to add the simulation distance to the distance coordinate
    wavelength.coords['distance'] = wavelength.coords['distance'] + simulation_distance
    h = sc.constants.h
    m_n = sc.constants.m_n
    velocity = (h / (wavelength * m_n)).to(unit='m/s')
    timeofflight = (sc.midpoints(wavelength.coords['distance'])) / velocity
    out = timeofflight.to(unit=time_unit, copy=False)
    # Include the variances computed above
    out.variances = variance.values
    return TimeOfFlightLookupTable[RunType](out)


def masked_tof_lookup_table(
    tof_lookup: TimeOfFlightLookupTable[RunType],
    variance_threshold: LookupTableVarianceThreshold,
) -> MaskedTimeOfFlightLookupTable[RunType]:
    """
    Mask regions of the lookup table where the variance of the projected time-of-flight
    is larger than a given threshold.

    Parameters
    ----------
    tof_lookup:
        Lookup table giving time-of-flight as a function of distance and
        time-of-arrival.
    variance_threshold:
        Threshold for the variance of the projected time-of-flight above which regions
        are masked.
    """
    variances = sc.variances(tof_lookup.data)
    mask = variances > sc.scalar(variance_threshold, unit=variances.unit)
    out = tof_lookup.copy(deep=False)
    if mask.any():
        out.masks["uncertain"] = mask
    return MaskedTimeOfFlightLookupTable[RunType](out)


def frame_at_detector_start_time(
    simulation: SimulationResults, ltotal: Ltotal[RunType]
) -> FrameAtDetectorStartTime[RunType]:
    """
    Compute the start time of the frame at the detector.
    The assumption here is that the fastest neutron in the simulation results is the one
    that arrives at the detector first.
    One could have an edge case where a slightly slower neutron which is born earlier
    could arrive at the detector first, but this edge case is most probably uncommon,
    and the difference in arrival times is likely to be small.

    Parameters
    ----------
    simulation:
        Results of a time-of-flight simulation used to create a lookup table.
    ltotal:
        Total length of the flight path from the source to the detector.
    """
    dist = ltotal - simulation.distance.to(unit=ltotal.unit)
    # Find the fastest neutron
    ind = np.argmax(simulation.speed.values)
    time_at_simulation = simulation.time_of_arrival[ind]
    toa = time_at_simulation + (dist / simulation.speed[ind]).to(
        unit=time_at_simulation.unit, copy=False
    )
    return FrameAtDetectorStartTime[RunType](toa)


def unwrapped_time_of_arrival(
    da: DetectorData[RunType], offset: PulseStrideOffset, pulse_period: PulsePeriod
) -> UnwrappedTimeOfArrival[RunType]:
    """
    Compute the unwrapped time of arrival of the neutron at the detector.
    For event data, this is essentially ``event_time_offset + event_time_zero``.

    Parameters
    ----------
    da:
        Raw detector data loaded from a NeXus file, e.g., NXdetector containing
        NXevent_data.
    offset:
        Integer offset of the first pulse in the stride (typically zero unless we are
        using pulse-skipping and the events do not begin with the first pulse in the
        stride).
    pulse_period:
        Period of the source pulses, i.e., time between consecutive pulse starts.
    """
    if da.bins is None:
        # Canonical name in NXmonitor
        toa = da.coords['time_of_flight']
    else:
        # To unwrap the time of arrival, we want to add the event_time_zero to the
        # event_time_offset. However, we do not really care about the exact datetimes,
        # we just want to know the offsets with respect to the start of the run.
        # Hence we use the smallest event_time_zero as the time origin.
        time_zero = da.coords['event_time_zero'] - da.coords['event_time_zero'].min()
        coord = da.bins.coords['event_time_offset']
        unit = elem_unit(coord)
        toa = (
            coord
            + time_zero.to(dtype=float, unit=unit, copy=False)
            - (offset * pulse_period).to(unit=unit, copy=False)
        )
    return UnwrappedTimeOfArrival[RunType](toa)


def unwrapped_time_of_arrival_minus_frame_start_time(
    toa: UnwrappedTimeOfArrival[RunType], start_time: FrameAtDetectorStartTime[RunType]
) -> UnwrappedTimeOfArrivalMinusStartTime[RunType]:
    """
    Compute the time of arrival of the neutron at the detector, unwrapped at the pulse
    period, minus the start time of the frame.
    We subtract the start time of the frame so that we can use a modulo operation to
    wrap the time of arrival at the frame period in the case of pulse-skipping.

    Parameters
    ----------
    toa:
        Time of arrival of the neutron at the detector, unwrapped at the pulse period.
    start_time:
        Time of the start of the frame at the detector.
    """
    # Order of operation to preserve dimension order
    return UnwrappedTimeOfArrivalMinusStartTime[RunType](
        -start_time.to(unit=elem_unit(toa), copy=False) + toa
    )


def time_of_arrival_minus_start_time_modulo_period(
    toa_minus_start_time: UnwrappedTimeOfArrivalMinusStartTime[RunType],
    frame_period: FramePeriod,
) -> TimeOfArrivalMinusStartTimeModuloPeriod[RunType]:
    """
    Compute the time of arrival of the neutron at the detector, unwrapped at the pulse
    period, minus the start time of the frame, modulo the frame period.

    Parameters
    ----------
    toa_minus_start_time:
        Time of arrival of the neutron at the detector, unwrapped at the pulse period,
        minus the start time of the frame.
    frame_period:
        Period of the frame, i.e., time between the start of two consecutive frames.
    """
    return TimeOfArrivalMinusStartTimeModuloPeriod[RunType](
        toa_minus_start_time
        % frame_period.to(unit=elem_unit(toa_minus_start_time), copy=False)
    )


def time_of_arrival_folded_by_frame(
    toa: TimeOfArrivalMinusStartTimeModuloPeriod[RunType],
    start_time: FrameAtDetectorStartTime[RunType],
) -> FrameFoldedTimeOfArrival[RunType]:
    """
    The time of arrival of the neutron at the detector, folded by the frame period.

    Parameters
    ----------
    toa:
        Time of arrival of the neutron at the detector, unwrapped at the pulse period,
        minus the start time of the frame, modulo the frame period.
    start_time:
        Time of the start of the frame at the detector.
    """
    return FrameFoldedTimeOfArrival[RunType](
        toa + start_time.to(unit=elem_unit(toa), copy=False)
    )


def time_of_flight_data(
    da: DetectorData[RunType],
    lookup: MaskedTimeOfFlightLookupTable[RunType],
    ltotal: Ltotal[RunType],
    toas: FrameFoldedTimeOfArrival[RunType],
) -> TofData[RunType]:
    from scipy.interpolate import RegularGridInterpolator

    lookup_values = lookup.data.to(unit=elem_unit(toas), copy=False).values
    # Merge all masks into a single mask
    if lookup.masks:
        one_mask = reduce(lambda a, b: a | b, lookup.masks.values()).values
        # Set masked values to NaN
        lookup_values[one_mask] = np.nan

    f = RegularGridInterpolator(
        (
            sc.midpoints(
                lookup.coords['toa'].to(unit=elem_unit(toas), copy=False)
            ).values,
            sc.midpoints(lookup.coords['distance']).values,
        ),
        lookup_values.T,
        method='linear',
        bounds_error=False,
    )

    if da.bins is not None:
        ltotal = sc.bins_like(toas, ltotal).bins.constituents['data']
        toas = toas.bins.constituents['data']

    tofs = sc.array(
        dims=toas.dims, values=f((toas.values, ltotal.values)), unit=elem_unit(toas)
    )

    out = da.copy(deep=False)
    if out.bins is not None:
        parts = out.bins.constituents
        out.data = sc.bins(**parts)
        parts['data'] = tofs
        out.bins.coords['tof'] = _bins_no_validate(**parts)
    else:
        out.coords['tof'] = tofs
    return TofData[RunType](out)


def re_histogram_tof_data(da: TofData[RunType]) -> ReHistogrammedTofData[RunType]:
    """
    Histogrammed data that has been converted to `tof` will typically have
    unsorted bin edges (due to either wrapping of `time_of_flight` or wavelength
    overlap between subframes).
    This function re-histograms the data to ensure that the bin edges are sorted.
    It makes use of the ``to_events`` helper which generates a number of events in each
    bin with a uniform distribution. The new events are then histogrammed using a set of
    sorted bin edges.

    WARNING:
    This function is highly experimental, has limitations and should be used with
    caution. It is a workaround to the issue that rebinning data with unsorted bin
    edges is not supported in scipp.
    We also do not support variances on the data.
    As such, this function is not part of the default set of providers, and needs to be
    inserted manually into the workflow.

    Parameters
    ----------
    da:
        TofData with the time-of-flight coordinate.
    """
    events = to_events(da.rename_dims(time_of_flight='tof'), 'event')

    # Define a new bin width, close to the original bin width.
    # TODO: this could be a workflow parameter
    coord = da.coords['tof']
    bin_width = (coord['time_of_flight', 1:] - coord['time_of_flight', :-1]).nanmedian()
    rehist = events.hist(tof=bin_width)
    for key, var in da.coords.items():
        if 'time_of_flight' not in var.dims:
            rehist.coords[key] = var
    return ReHistogrammedTofData[RunType](rehist)


def providers():
    """
    Providers of the time-of-flight workflow.
    """
    return (
        pulse_period_from_source,
        frame_period,
        run_tof_model,
        extract_ltotal,
        frame_at_detector_start_time,
        unwrapped_time_of_arrival,
        unwrapped_time_of_arrival_minus_frame_start_time,
        time_of_arrival_minus_start_time_modulo_period,
        time_of_arrival_folded_by_frame,
        time_of_flight_data,
        compute_tof_lookup_table,
        masked_tof_lookup_table,
    )


def params() -> dict:
    """
    Default parameters of the time-of-flight workflow.
    """
    return {
        PulseStride: 1,
        PulseStrideOffset: 0,
        DistanceResolution: sc.scalar(1.0, unit='cm'),
        LookupTableVarianceThreshold: 1.0e-2,
        SimulationSeed: 1234,
        NumberOfNeutrons: 1_000_000,
    }
