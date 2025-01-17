# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NewType

import scipp as sc
from scippneutron.chopper import DiskChopper

Facility = NewType("Facility", str)
"""
Facility where the experiment is performed.
"""

Choppers = NewType("Choppers", Mapping[str, DiskChopper])
"""
Choppers used to define the frame parameters.
"""

Ltotal = NewType("Ltotal", sc.Variable)
"""
Total length of the flight path from the source to the detector.
"""

SimulationSeed = NewType("SimulationSeed", int)
"""
Seed for the random number generator used in the simulation.
"""


NumberOfNeutrons = NewType("NumberOfNeutrons", int)
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


@dataclass
class FastestNeutron:
    """
    Properties of the fastest neutron in the simulation results.
    """

    time_of_arrival: sc.Variable
    speed: sc.Variable
    distance: sc.Variable


LtotalRange = NewType("LtotalRange", tuple[sc.Variable, sc.Variable])
"""
Range (min, max) of the total length of the flight path from the source to the detector.
"""


DistanceResolution = NewType("DistanceResolution", sc.Variable)
"""
Resolution of the distance axis in the lookup table.
"""

TimeOfArrivalResolution = NewType("TimeOfArrivalResolution", int | sc.Variable)
"""
Resolution of the time of arrival axis in the lookup table.
Can be an integer (number of bins) or a sc.Variable (bin width).
"""

TimeOfFlightLookupTable = NewType("TimeOfFlightLookupTable", sc.DataArray)
"""
Lookup table giving time-of-flight as a function of distance and time of arrival.
"""

MaskedTimeOfFlightLookupTable = NewType("MaskedTimeOfFlightLookupTable", sc.DataArray)
"""
Lookup table giving time-of-flight as a function of distance and time of arrival, with
regions of large uncertainty masked out.
"""

LookupTableVarianceThreshold = NewType("LookupTableVarianceThreshold", float)

FramePeriod = NewType("FramePeriod", sc.Variable)
"""
The period of a frame, a (small) integer multiple of the source period.
"""

UnwrappedTimeOfArrival = NewType("UnwrappedTimeOfArrival", sc.Variable)
"""
Time of arrival of the neutron at the detector, unwrapped at the pulse period.
"""

PivotTimeAtDetector = NewType("PivotTimeAtDetector", sc.Variable)
"""
Pivot time at the detector, i.e., the time of the start of the frame at the detector.
"""

UnwrappedTimeOfArrivalMinusPivotTime = NewType(
    "UnwrappedTimeOfArrivalMinusPivotTime", sc.Variable
)
"""
Time of arrival of the neutron at the detector, unwrapped at the pulse period, minus
the start time of the frame.
"""

TimeOfArrivalMinusPivotTimeModuloPeriod = NewType(
    "TimeOfArrivalMinusPivotTimeModuloPeriod", sc.Variable
)
"""
Time of arrival of the neutron at the detector minus the start time of the frame,
modulo the frame period.
"""

FrameFoldedTimeOfArrival = NewType("FrameFoldedTimeOfArrival", sc.Variable)


PulsePeriod = NewType("PulsePeriod", sc.Variable)
"""
Period of the source pulses, i.e., time between consecutive pulse starts.
"""

PulseStride = NewType("PulseStride", int)
"""
Stride of used pulses. Usually 1, but may be a small integer when pulse-skipping.
"""

PulseStrideOffset = NewType("PulseStrideOffset", int)
"""
When pulse-skipping, the offset of the first pulse in the stride.
"""

RawData = NewType("RawData", sc.DataArray)
"""
Raw detector data loaded from a NeXus file, e.g., NXdetector containing NXevent_data.
"""

TofData = NewType("TofData", sc.DataArray)
"""
Detector data with time-of-flight coordinate.
"""

ReHistogrammedTofData = NewType("ReHistogrammedTofData", sc.DataArray)
"""
Detector data with time-of-flight coordinate, re-histogrammed.
"""
