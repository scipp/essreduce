# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

"""
Utilities for computing real neutron time-of-flight from chopper settings and
neutron time-of-arrival at the detectors.
"""

# import sciline as sl

from .toa_to_tof import (
    default_parameters,
    re_histogram_tof_data,
    standard_providers,
    TofWorkflow,
)
from .types import (
    Choppers,
    DistanceResolution,
    Facility,
    FrameFoldedTimeOfArrival,
    FramePeriod,
    LookupTableVarianceThreshold,
    LtotalRange,
    MaskedTimeOfFlightLookupTable,
    NumberOfNeutrons,
    PivotTimeAtDetector,
    PulsePeriod,
    PulseStride,
    PulseStrideOffset,
    RawData,
    ReHistogrammedTofData,
    SimulationResults,
    SimulationSeed,
    TimeOfArrivalMinusPivotTimeModuloPeriod,
    TimeOfFlightLookupTable,
    TofData,
    UnwrappedTimeOfArrival,
    UnwrappedTimeOfArrivalMinusPivotTime,
)


# def TofWorkflow() -> sl.Pipeline:
#     wf = sl.Pipeline(standard_providers(), params=params())
#     wf[Facility] = "ess"
#     return wf


__all__ = [
    "Choppers",
    "DistanceResolution",
    "Facility",
    "FrameFoldedTimeOfArrival",
    "FramePeriod",
    "LookupTableVarianceThreshold",
    "LtotalRange",
    "MaskedTimeOfFlightLookupTable",
    "NumberOfNeutrons",
    "PivotTimeAtDetector",
    "PulsePeriod",
    "PulseStride",
    "PulseStrideOffset",
    "RawData",
    "ReHistogrammedTofData",
    "SimulationResults",
    "SimulationSeed",
    "TimeOfArrivalMinusPivotTimeModuloPeriod",
    "TimeOfFlightLookupTable",
    "TofData",
    "TofWorkflow",
    "UnwrappedTimeOfArrival",
    "UnwrappedTimeOfArrivalMinusPivotTime",
    "default_parameters",
    "re_histogram_tof_data",
    "standard_providers",
]
