# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, NewType

import sciline as sl
import scipp as sc

from ..nexus.types import MonitorType, RunType

TofLookupTableFilename = NewType("TofLookupTableFilename", str)
"""Filename of the time-of-flight lookup table."""

TimeOfFlightLookupTableFilename = TofLookupTableFilename
"""Filename of the time-of-flight lookup table (alias)."""


@dataclass
class TofLookupTable:
    """
    Lookup table giving time-of-flight as a function of distance and time of arrival.
    """

    array: sc.DataArray
    """The lookup table data array that maps (distance, time_of_arrival) to
    time_of_flight."""
    pulse_period: sc.Variable
    """Pulse period of the neutron source."""
    pulse_stride: int
    """Pulse stride used when generating the lookup table."""
    distance_resolution: sc.Variable
    """Resolution of the distance coordinate in the lookup table."""
    time_resolution: sc.Variable
    """Resolution of the time_of_arrival coordinate in the lookup table."""
    error_threshold: float
    """The table is masked with NaNs in regions where the standard deviation of the
    time-of-flight is above this threshold."""
    choppers: sc.DataGroup | None = None
    """Chopper parameters used when generating the lookup table, if any. This is made
    optional so we can still support old lookup tables without chopper info."""

    def save_hdf5(self, filename: str | Path) -> None:
        """Save the lookup table to an HDF5 file."""
        sc.DataGroup(asdict(self)).save_hdf5(filename)

    def plot(self, *args, **kwargs) -> Any:
        """Plot the data array of the lookup table."""
        return self.array.plot(*args, **kwargs)


TimeOfFlightLookupTable = TofLookupTable
"""Lookup table giving time-of-flight as a function of distance and time of arrival
(alias)."""

PulseStrideOffset = NewType("PulseStrideOffset", int | None)
"""
When pulse-skipping, the offset of the first pulse in the stride. This is typically
zero but can be a small integer < pulse_stride. If None, a guess is made.
"""


class DetectorLtotal(sl.Scope[RunType, sc.Variable], sc.Variable):
    """Total path length of neutrons from source to detector (L1 + L2)."""


class MonitorLtotal(sl.Scope[RunType, MonitorType, sc.Variable], sc.Variable):
    """Total path length of neutrons from source to monitor."""


class TofDetector(sl.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data with time-of-flight coordinate."""


class ToaDetector(sl.Scope[RunType, sc.DataArray], sc.DataArray):
    """Detector data with time-of-arrival coordinate.

    When the pulse stride is 1 (i.e., no pulse skipping), the time-of-arrival is the
    same as the event_time_offset. When pulse skipping is used, the time-of-arrival is
    the event_time_offset + pulse_offset * pulse_period.
    This means that the time-of-arrival is basically the event_time_offset wrapped
    over the frame period instead of the pulse period
    (where frame_period = pulse_stride * pulse_period).
    """


class TofMonitor(sl.Scope[RunType, MonitorType, sc.DataArray], sc.DataArray):
    """Monitor data with time-of-flight coordinate."""
