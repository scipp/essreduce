# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from dataclasses import dataclass
from typing import NewType

import sciline as sl
import scipp as sc

from ..nexus.types import MonitorType, RunType


@dataclass
class SimulationResults:
    """
    Results of a time-of-flight simulation used to create a lookup table.

    The results (apart from ``distance``) should be flat lists (1d arrays) of length N
    where N is the number of neutrons, containing the properties of the neutrons in the
    simulation.

    Parameters
    ----------
    time_of_arrival:
        Time of arrival of the neutrons at the position where the events were recorded
        (1d array of size N).
    speed:
        Speed of the neutrons, typically derived from the wavelength of the neutrons
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
    speed: sc.Variable
    wavelength: sc.Variable
    weight: sc.Variable
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

TimeOfFlightLookupTable = NewType("TimeOfFlightLookupTable", sc.DataArray)
"""
Lookup table giving time-of-flight as a function of distance and time of arrival.
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

PulseStrideOffset = NewType("PulseStrideOffset", int | None)
"""
When pulse-skipping, the offset of the first pulse in the stride. This is typically
zero but can be a small integer < pulse_stride. If None, a guess is made.
"""


class DetectorLtotal(sl.Scope[RunType, sc.Variable], sc.Variable):
    """Total path length of neutrons from source to detector (L1 + L2)."""


class MonitorLtotal(sl.Scope[RunType, MonitorType, sc.Variable], sc.Variable):
    """Total path length of neutrons from source to monitor."""


class DetectorTofData(sl.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data with time-of-flight coordinate."""


class MonitorTofData(sl.Scope[RunType, MonitorType, sc.DataArray], sc.DataArray):
    """Monitor data with time-of-flight coordinate."""


class ResampledDetectorTofData(sl.Scope[RunType, sc.DataArray], sc.DataArray):
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


class ResampledMonitorTofData(
    sl.Scope[RunType, MonitorType, sc.DataArray], sc.DataArray
):
    """
    Histogrammed monitor data with time-of-flight coordinate, that has been resampled.

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
