# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""
Utilities for computing real neutron time-of-flight from chopper settings and
neutron time-of-arrival at the detectors.
"""

from .simulation import simulate_beamline
from .toa_to_tof import default_parameters, resample_tof_data, providers, TofWorkflow
from .to_events import to_events
from .types import (
    DistanceResolution,
    FrameFoldedTimeOfArrival,
    FramePeriod,
    LookupTableRelativeErrorThreshold,
    Ltotal,
    LtotalRange,
    MaskedTimeOfFlightLookupTable,
    PivotTimeAtDetector,
    PulsePeriod,
    PulseStride,
    PulseStrideOffset,
    RawData,
    ResampledTofData,
    SimulationResults,
    TimeOfArrivalMinusPivotTimeModuloPeriod,
    TimeOfFlightLookupTable,
    TimeResolution,
    TofData,
    UnwrappedTimeOfArrival,
    UnwrappedTimeOfArrivalMinusPivotTime,
)


__all__ = [
    "DistanceResolution",
    "FrameFoldedTimeOfArrival",
    "FramePeriod",
    "LookupTableRelativeErrorThreshold",
    "Ltotal",
    "LtotalRange",
    "MaskedTimeOfFlightLookupTable",
    "PivotTimeAtDetector",
    "PulsePeriod",
    "PulseStride",
    "PulseStrideOffset",
    "RawData",
    "ResampledTofData",
    "SimulationResults",
    "TimeOfArrivalMinusPivotTimeModuloPeriod",
    "TimeOfFlightLookupTable",
    "TimeResolution",
    "TofData",
    "TofWorkflow",
    "UnwrappedTimeOfArrival",
    "UnwrappedTimeOfArrivalMinusPivotTime",
    "default_parameters",
    "providers",
    "resample_tof_data",
    "simulate_beamline",
    "to_events",
]
