# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NewType

import sciline as sl
import scipp as sc
from scippneutron.chopper import DiskChopper

from ..nexus.types import RunType

Facility = NewType("Facility", str)
"""
Facility where the experiment is performed.
"""

Choppers = NewType("Choppers", Mapping[str, DiskChopper])
"""
Choppers used to define the frame parameters.
"""


class Ltotal(sl.Scope[RunType, sc.Variable], sc.Variable):
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


DistanceResolution = NewType("DistanceResolution", sc.Variable)
"""
Resolution of the distance axis in the lookup table.
"""


class TimeOfFlightLookupTable(sl.Scope[RunType, sc.DataArray], sc.DataArray):
    """
    Lookup table giving time-of-flight as a function of distance and time of arrival.
    """


class MaskedTimeOfFlightLookupTable(sl.Scope[RunType, sc.DataArray], sc.DataArray):
    """
    Lookup table giving time-of-flight as a function of distance and time of arrival,
    with regions of large uncertainty masked out.
    """


LookupTableVarianceThreshold = NewType("LookupTableVarianceThreshold", float)

FramePeriod = NewType("FramePeriod", sc.Variable)
"""
The period of a frame, a (small) integer multiple of the source period.
"""


class UnwrappedTimeOfArrival(sl.Scope[RunType, sc.Variable], sc.Variable):
    """
    Time of arrival of the neutron at the detector, unwrapped at the pulse period.
    """


class PivotTimeAtDetector(sl.Scope[RunType, sc.Variable], sc.Variable):
    """
    Pivot time at the detector, i.e., the time of the start of the frame at the
    detector.
    """


class UnwrappedTimeOfArrivalMinusPivotTime(sl.Scope[RunType, sc.Variable], sc.Variable):
    """
    Time of arrival of the neutron at the detector, unwrapped at the pulse period, minus
    the start time of the frame.
    """


class TimeOfArrivalMinusPivotTimeModuloPeriod(
    sl.Scope[RunType, sc.Variable], sc.Variable
):
    """
    Time of arrival of the neutron at the detector minus the start time of the frame,
    modulo the frame period.
    """


class FrameFoldedTimeOfArrival(sl.Scope[RunType, sc.Variable], sc.Variable):
    """
    Time of arrival of the neutron at the detector, folded by the frame period.
    """


PulsePeriod = NewType("PulsePeriod", sc.Variable)
"""
Period of the source pulses, i.e., time between consecutive pulse starts.
"""

PulseStride = NewType("PulseStride", int)
"""
Stride of used pulses. Usually 1, but may be a small integer when pulse-skipping.
"""

# TODO: the pulse stride offset may be different for sample and background runs?
# It should maybe be turned into a generic type?
PulseStrideOffset = NewType("PulseStrideOffset", int)
"""
When pulse-skipping, the offset of the first pulse in the stride.
"""


class TofData(sl.Scope[RunType, sc.DataArray], sc.DataArray):
    """
    Detector data with time-of-flight coordinate.
    """


class ReHistogrammedTofData(sl.Scope[RunType, sc.DataArray], sc.DataArray):
    """
    Detector data with time-of-flight coordinate, re-histogrammed.
    """
