# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""
Utilities for computing real neutron time-of-flight from chopper settings and
neutron time-of-arrival at the detectors.
"""

from .toa_to_tof import (
    cache_results,
    default_parameters,
    resample_tof_data,
    providers,
    tof_workflow,
)
from .simulation import simulate_beamline
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
    "TofData",
    "UnwrappedTimeOfArrival",
    "UnwrappedTimeOfArrivalMinusPivotTime",
    "cache_results",
    "default_parameters",
    "providers",
    "resample_tof_data",
    "simulate_beamline",
    "tof_workflow",
]
