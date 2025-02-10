# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""
Utilities for computing real neutron time-of-flight from chopper settings and
neutron time-of-arrival at the detectors.
"""

from .simulation import simulate_beamline
from .toa_to_tof import default_parameters, resample_tof_data, providers
from .to_events import to_events
from .types import (
    DistanceResolution,
    LookupTableRelativeErrorThreshold,
    Ltotal,
    LtotalRange,
    PulsePeriod,
    PulseStride,
    PulseStrideOffset,
    RawData,
    ResampledTofData,
    SimulationResults,
    TimeOfFlightLookupTable,
    TimeResolution,
    TofData,
)


__all__ = [
    "DistanceResolution",
    "LookupTableRelativeErrorThreshold",
    "Ltotal",
    "LtotalRange",
    "PulsePeriod",
    "PulseStride",
    "PulseStrideOffset",
    "RawData",
    "ResampledTofData",
    "SimulationResults",
    "TimeOfFlightLookupTable",
    "TimeResolution",
    "TofData",
    "default_parameters",
    "providers",
    "resample_tof_data",
    "simulate_beamline",
    "to_events",
]
