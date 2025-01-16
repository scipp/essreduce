# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""
Utilities for computing real neutron time-of-flight from chopper settings and
neutron time-of-arrival at the detectors.
"""

from .toa_to_tof import params, re_histogram_tof_data, standard_providers
from .types import (
    Facility,
    Choppers,
    SimulationSeed,
    NumberOfNeutrons,
    SimulationResults,
    DistanceResolution,
    TimeOfFlightLookupTable,
    MaskedTimeOfFlightLookupTable,
    LookupTableVarianceThreshold,
    FramePeriod,
    UnwrappedTimeOfArrival,
    PivotTimeAtDetector,
    UnwrappedTimeOfArrivalMinusPivotTime,
    TimeOfArrivalMinusPivotTimeModuloPeriod,
    FrameFoldedTimeOfArrival,
    PulsePeriod,
    PulseStride,
    PulseStrideOffset,
    TofData,
    ReHistogrammedTofData,
)


__all__ = [
    "Facility",
    "Choppers",
    "SimulationSeed",
    "NumberOfNeutrons",
    "SimulationResults",
    "DistanceResolution",
    "TimeOfFlightLookupTable",
    "MaskedTimeOfFlightLookupTable",
    "LookupTableVarianceThreshold",
    "FramePeriod",
    "UnwrappedTimeOfArrival",
    "PivotTimeAtDetector",
    "UnwrappedTimeOfArrivalMinusPivotTime",
    "TimeOfArrivalMinusPivotTimeModuloPeriod",
    "FrameFoldedTimeOfArrival",
    "PulsePeriod",
    "PulseStride",
    "PulseStrideOffset",
    "TofData",
    "ReHistogrammedTofData",
    "params",
    "re_histogram_tof_data",
    "standard_providers",
]
