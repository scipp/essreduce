# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""
Utilities for computing real neutron time-of-flight from chopper settings and
neutron time-of-arrival at the detectors.
"""

from .toa_to_tof import (
    default_parameters,
    resample_tof_data,
    providers,
    TofWorkflow,
)
from .simulation import simulate_beamline
from .types import (
    DistanceResolution,
    FrameFoldedTimeOfArrival,
    FramePeriod,
    LookupTableRelativeErrorThreshold,
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
    "TofWorkflow",
    "UnwrappedTimeOfArrival",
    "UnwrappedTimeOfArrivalMinusPivotTime",
    "default_parameters",
    "providers",
    "resample_tof_data",
    "simulate_beamline",
]
