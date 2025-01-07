# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
""" """

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import NewType

import numpy as np
import scipp as sc
import tof

# from scipp.core.bins import Lookup
from scippneutron._utils import elem_unit

# from . import chopper_cascade
# from .to_events import to_events

Choppers = NewType('Choppers', list[tof.Chopper])
"""
Choppers used to define the frame parameters.
"""

PrimaryFlightPath = NewType('PrimaryFlightPath', sc.Variable)

SecondaryFlightPath = NewType('SecondaryFlightPath', sc.Variable)

SimulationResults = NewType('SimulationResults', tof.Result)


DistanceResolution = NewType('DistanceResolution', sc.Variable)

TimeOfFlightLookupTable = NewType('TimeOfFlightLookupTable', sc.DataArray)


# ChopperCascadeFrames = NewType(
#     'ChopperCascadeFrames', list[chopper_cascade.FrameSequence]
# )
# """
# Frames of the chopper cascade.
# """

# FrameAtDetector = NewType('FrameAtDetector', chopper_cascade.Frame)
# """
# Result of passing the source pulse through the chopper cascade to the detector.

# The detector may be a monitor or a detector after scattering off the sample. The frame
# bounds are then computed from this.
# """

FramePeriod = NewType('FramePeriod', sc.Variable)
"""
The period of a frame, a (small) integer multiple of the source period.
"""

UnwrappedTimeOfArrival = NewType('UnwrappedTimeOfArrival', sc.Variable)
"""
Time of arrival of the neutron at the detector, unwrapped at the pulse period.
"""

FrameAtDetectorStartTime = NewType('FrameAtDetectorStartTime', sc.Variable)
"""
Time of the start of the frame at the detector.
"""

UnwrappedTimeOfArrivalMinusStartTime = NewType(
    'UnwrappedTimeOfArrivalMinusStartTime', sc.Variable
)
"""
Time of arrival of the neutron at the detector, unwrapped at the pulse period, minus
the start time of the frame.
"""

TimeOfArrivalMinusStartTimeModuloPeriod = NewType(
    'TimeOfArrivalMinusStartTimeModuloPeriod', sc.Variable
)
"""
Time of arrival of the neutron at the detector minus the start time of the frame,
modulo the frame period.
"""

FrameFoldedTimeOfArrival = NewType('FrameFoldedTimeOfArrival', sc.Variable)

# @dataclass
# class TimeOfArrivalToTimeOfFlight:
#     """ """

#     slope: Lookup
#     intercept: Lookup


TofCoord = NewType('TofCoord', sc.Variable)
"""
Tof coordinate computed by the workflow.
"""

Ltotal = NewType('Ltotal', sc.Variable)
"""
Total distance between the source and the detector(s).

This is used to propagate the frame to the detector position. This will then yield
detector-dependent frame bounds. This is typically the sum of L1 and L2, except for
monitors.
"""

PulsePeriod = NewType('PulsePeriod', sc.Variable)
"""
Period of the source pulses, i.e., time between consecutive pulse starts.
"""

PulseStride = NewType('PulseStride', int)
"""
Stride of used pulses. Usually 1, but may be a small integer when pulse-skipping.
"""

PulseStrideOffset = NewType('PulseStrideOffset', int)
"""
When pulse-skipping, the offset of the first pulse in the stride.
"""

RawData = NewType('RawData', sc.DataArray)
"""
Raw detector data loaded from a NeXus file, e.g., NXdetector containing NXevent_data.
"""

# SourceTimeRange = NewType('SourceTimeRange', tuple[sc.Variable, sc.Variable])
# """
# Time range of the source pulse, used for computing frame bounds.
# """

# SourceWavelengthRange = NewType(
#     'SourceWavelengthRange', tuple[sc.Variable, sc.Variable]
# )
# """
# Wavelength range of the source pulse, used for computing frame bounds.
# """

TofData = NewType('TofData', sc.DataArray)
"""
Detector data with time-of-flight coordinate.
"""

ReHistogrammedTofData = NewType('ReHistogrammedTofData', sc.DataArray)
"""
Detector data with time-of-flight coordinate, re-histogrammed.
"""


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


def run_tof_model(choppers: Choppers, l1: PrimaryFlightPath) -> SimulationResults:
    # tof_choppers = [
    #     tof.Chopper(
    #         frequency=abs(ch.frequency),
    #         direction=tof.AntiClockwise
    #         if (ch.frequency.value > 0.0)
    #         else tof.Clockwise,
    #         open=ch.slit_begin,
    #         close=ch.slit_end,
    #         phase=abs(ch.phase),
    #         distance=ch.axle_position.fields.z,
    #         name=name,
    #     )
    #     for name, ch in choppers.items()
    # ]
    source = tof.Source(facility='ess', neutrons=1_000_000)
    detectors = [tof.Detector(distance=l1, name='sample')]
    model = tof.Model(source=source, choppers=choppers, detectors=detectors)
    results = model.run()
    events = results['sample'].data.squeeze()
    # events = events[~events.masks['blocked_by_others']]
    return SimulationResults(events[~events.masks['blocked_by_others']])


def frame_at_detector_start_time(
    events: SimulationResults,
    l2: SecondaryFlightPath,
) -> FrameAtDetectorStartTime:
    """
    Compute the start time of the frame at the detector.
    The assumption here is that the fastest neutron at the sample is also the first one
    to reach the detector.

    Parameters
    ----------
    frame:
        Frame at the detector
    """
    fastest_neutron = events[np.argmax(events.coords['speed'].values)]
    start_times = fastest_neutron.coords['toa'] + (
        l2 / fastest_neutron.coords['speed']
    ).to(unit=fastest_neutron.coords['toa'].unit)
    return FrameAtDetectorStartTime(start_times)


def tof_lookup(
    events: SimulationResults,
    l2: SecondaryFlightPath,
    distance_resolution: DistanceResolution,
) -> TimeOfFlightLookupTable:
    # events = results['sample'].data.squeeze()
    # events = events[~events.masks['blocked_by_others']]

    l2max = l2.max()
    ndist = int((l2max / distance_resolution.to(unit=l2.unit)).value) + 1
    dist = sc.linspace('distance', 0, l2max.value, ndist, unit=l2.unit)

    toas = events.coords['toa'] + (dist / events.coords['speed']).to(
        unit=events.coords['toa'].unit
    )

    data = sc.DataArray(
        data=sc.broadcast(events.data, sizes=toas.sizes).flatten(to='event'),
        coords={
            'toa': toas.flatten(to='event'),
            'wavelength': sc.broadcast(
                events.coords['wavelength'], sizes=toas.sizes
            ).flatten(to='event'),
            'distance': sc.broadcast(dist, sizes=toas.sizes).flatten(to='event'),
        },
    )

    binned = data.bin(distance=ndist, toa=500)
    # Weighted mean of wavelength inside each bin
    wavelength = (
        binned.bins.data * binned.bins.coords['wavelength']
    ).bins.sum() / binned.bins.sum()

    # Convert wavelengths to time-of-flight
    h = sc.constants.h
    m_n = sc.constants.m_n
    velocity = (h / (wavelength * m_n)).to(unit='m/s')
    timeofflight = sc.midpoints(binned.coords['distance']) / velocity
    return TimeOfFlightLookupTable(timeofflight)


# def time_of_flight_from_lookup(
#     lookup: TimeOfFlightLookupTable, l2: sc.Variable, toas
# ) -> TofCoord:
#     from scipy.interpolate import RegularGridInterpolator

#     f = RegularGridInterpolator(
#         (
#             sc.midpoints(lookup.coords['toa']).values,
#             sc.midpoints(lookup.coords['distance']).values,
#         ),
#         lookup.values.T,
#         method='linear',
#         bounds_error=False,
#     )


# def chopper_cascade_frames(
#     source_wavelength_range: SourceWavelengthRange,
#     source_time_range: SourceTimeRange,
#     choppers: Choppers,
#     pulse_stride: PulseStride,
#     pulse_period: PulsePeriod,
# ) -> ChopperCascadeFrames:
#     """
#     Return the frames of the chopper cascade.
#     This is the result of propagating the source pulse through the chopper cascade.

#     In the case of pulse-skipping, the frames are computed for each pulse in the stride,
#     to make sure that we include cases where e.g. the first pulse in the stride is
#     skipped, but the second is not.

#     Parameters
#     ----------
#     source_wavelength_range:
#         Wavelength range of the source pulse.
#     source_time_range:
#         Time range of the source pulse.
#     choppers:
#         Choppers used to define the frame parameters.
#     pulse_stride:
#         Stride of used pulses. Usually 1, but may be a small integer when
#         pulse-skipping.
#     pulse_period:
#         Period of the source pulses, i.e., time between consecutive pulse starts.
#     """
#     out = []
#     for i in range(pulse_stride):
#         offset = (pulse_period * i).to(unit=source_time_range[0].unit, copy=False)
#         frames = chopper_cascade.FrameSequence.from_source_pulse(
#             time_min=source_time_range[0] + offset,
#             time_max=source_time_range[-1] + offset,
#             wavelength_min=source_wavelength_range[0],
#             wavelength_max=source_wavelength_range[-1],
#         )
#         chopped = frames.chop(choppers.values())
#         for f in chopped:
#             for sf in f.subframes:
#                 sf.time -= offset.to(unit=sf.time.unit, copy=False)
#         out.append(chopped)
#     return ChopperCascadeFrames(out)


# def frame_at_detector(
#     frames: ChopperCascadeFrames,
#     ltotal: Ltotal,
#     period: FramePeriod,
# ) -> FrameAtDetector:
#     """
#     Return the frame at the detector.

#     This is the result of propagating the source pulse through the chopper cascade to
#     the detector. The detector may be a monitor or a detector after scattering off the
#     sample. The frame bounds are then computed from this.

#     It is assumed that the opening and closing times of the input choppers have been
#     setup correctly.

#     Parameters
#     ----------
#     frames:
#         Frames of the chopper cascade.
#     ltotal:
#         Total distance between the source and the detector(s).
#     period:
#         Period of the frame, i.e., time between the start of two consecutive frames.
#     """

#     # In the case of pulse-skipping, only one of the frames should have subframes (the
#     # others should be empty).
#     at_detector = []
#     for f in frames:
#         propagated = f[-1].propagate_to(ltotal)
#         if len(propagated.subframes) > 0:
#             at_detector.append(propagated)
#     if len(at_detector) == 0:
#         raise ValueError("FrameAtDetector: No frames with subframes found.")
#     if len(at_detector) > 1:
#         raise ValueError("FrameAtDetector: Multiple frames with subframes found.")
#     at_detector = at_detector[0]

#     # Check that the frame bounds do not span a range larger than the frame period.
#     # This would indicate that the chopper phases are not set correctly.
#     bounds = at_detector.bounds()['time']
#     diff = (bounds.max('bound') - bounds.min('bound')).flatten(to='x')
#     if any(diff > period.to(unit=diff.unit, copy=False)):
#         raise ValueError(
#             "Frames are overlapping: Computed frame bounds "
#             f"{bounds} = {diff.max()} are larger than frame period {period}."
#         )
#     return FrameAtDetector(at_detector)


def unwrapped_time_of_arrival(
    da: RawData, offset: PulseStrideOffset, period: PulsePeriod
) -> UnwrappedTimeOfArrival:
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
    period:
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
            - (offset * period).to(unit=unit, copy=False)
        )
    return UnwrappedTimeOfArrival(toa)


# def frame_at_detector_start_time(frame: FrameAtDetector) -> FrameAtDetectorStartTime:
#     """
#     Compute the start time of the frame at the detector.

#     Parameters
#     ----------
#     frame:
#         Frame at the detector
#     """
#     return FrameAtDetectorStartTime(frame.bounds()['time']['bound', 0])


def unwrapped_time_of_arrival_minus_frame_start_time(
    toa: UnwrappedTimeOfArrival, start_time: FrameAtDetectorStartTime
) -> UnwrappedTimeOfArrivalMinusStartTime:
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
    return UnwrappedTimeOfArrivalMinusStartTime(
        -start_time.to(unit=elem_unit(toa), copy=False) + toa
    )


def time_of_arrival_minus_start_time_modulo_period(
    toa_minus_start_time: UnwrappedTimeOfArrivalMinusStartTime,
    frame_period: FramePeriod,
) -> TimeOfArrivalMinusStartTimeModuloPeriod:
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
    return TimeOfArrivalMinusStartTimeModuloPeriod(
        toa_minus_start_time
        % frame_period.to(unit=elem_unit(toa_minus_start_time), copy=False)
    )


def time_of_arrival_folded_by_frame(
    toa: TimeOfArrivalMinusStartTimeModuloPeriod,
    start_time: FrameAtDetectorStartTime,
) -> FrameFoldedTimeOfArrival:
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
    return FrameFoldedTimeOfArrival(toa + start_time)


def time_of_flight_from_lookup(
    lookup: TimeOfFlightLookupTable,
    l2: SecondaryFlightPath,
    toas: FrameFoldedTimeOfArrival,
) -> TofCoord:
    from scipy.interpolate import RegularGridInterpolator

    f = RegularGridInterpolator(
        (
            sc.midpoints(lookup.coords['toa']).values,
            sc.midpoints(lookup.coords['distance']).values,
        ),
        lookup.values.T,
        method='linear',
        bounds_error=False,
    )

    if toas.bins is not None:
        l2 = sc.bins_like(toas.bins, l2).bins.concat().value
        toas = toas.bins.concat().value

    tofs = sc.array(dims=toas.dims, values=f((toas.values, l2.values)), unit=toas.unit)
    return TofCoord(tofs)
