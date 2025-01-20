# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from collections.abc import Mapping
from dataclasses import dataclass
from typing import NewType

import scipp as sc
from scippneutron.chopper import DiskChopper

Ltotal = NewType("Ltotal", sc.Variable)
"""
Total length of the flight path from the source to the detector.
"""


@dataclass
class SimulationResults:
    """
    Results of a time-of-flight simulation used to create a lookup table.

    The results should be flat lists (1d arrays) of the properties of the neutrons in
    the simulation.

    Parameters
    ----------
    time_of_arrival:
        Time of arrival of the neutrons at the position where the events were recorded.
        For a ``tof`` simulation, this is just the position of the component (chopper or
        detector) where the events are recorded. For a ``McStas`` simulation, this is
        the position of the event monitor.
    speed:
        Speed of the neutrons (typically derived from the wavelength of the neutrons).
    wavelength:
        Wavelength of the neutrons.
    weight:
        Weight/probability of the neutrons.
    distance:
        Distance from the source to the position where the events were recorded.
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

LookupTableRelativeErrorThreshold = NewType("LookupTableRelativeErrorThreshold", float)

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

ResampledTofData = NewType("ResampledTofData", sc.DataArray)
"""
Histogrammed detector data with time-of-flight coordinate, that has been resampled.

Histogrammed data that has been converted to `tof` will typically have
unsorted bin edges (due to either wrapping of `time_of_flight` or wavelength
overlap between subframes).
We thus resample the data to ensure that the bin edges are sorted.
It makes use of the ``to_events`` helper which generates a number of events in each
bin with a uniform distribution. The new events are then histogrammed using a set of
sorted bin edges to yield a new histogram with sorted bin edges.

WARNING:
This function is highly experimental, has limitations and should be used with
caution. It is a workaround to the issue that rebinning data with unsorted bin
edges is not supported in scipp.
"""
