# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from typing import NewType

import sciline as sl
import scipp as sc

from ..nexus.types import MonitorType, RunType

TimeOfFlightLookupTableFilename = NewType("TimeOfFlightLookupTableFilename", str)
"""Filename of the time-of-flight lookup table."""


TimeOfFlightLookupTable = NewType("TimeOfFlightLookupTable", sc.DataArray)
"""
Lookup table giving time-of-flight as a function of distance and time of arrival.
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
