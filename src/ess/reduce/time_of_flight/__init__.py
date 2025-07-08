# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""
Utilities for computing real neutron time-of-flight from chopper settings and
neutron time-of-arrival at the detectors.
"""

from .simulation import simulate_beamline
from .types import (
    CommonTimeOfFlightLookupTable,
    DetectorLtotal,
    DetectorTofData,
    DistanceResolution,
    LookupTableRelativeErrorThreshold,
    LtotalRange,
    MonitorLtotal,
    MonitorTofData,
    PulsePeriod,
    PulseStride,
    PulseStrideOffset,
    SimulationResults,
    TimeOfFlightLookupTable,
    TimeOfFlightLookupTableFilename,
    TimeOfFlightLookupTableFromSimulation,
    TimeResolution,
)
from .workflow import GenericTofWorkflow, TofLutProvider, default_parameters, providers

__all__ = [
    "CommonTimeOfFlightLookupTable",
    "DetectorLtotal",
    "DetectorTofData",
    "DistanceResolution",
    "GenericTofWorkflow",
    "LookupTableRelativeErrorThreshold",
    "LtotalRange",
    "MonitorLtotal",
    "MonitorTofData",
    "PulsePeriod",
    "PulseStride",
    "PulseStrideOffset",
    "SimulationResults",
    "TimeOfFlightLookupTable",
    "TimeOfFlightLookupTableFilename",
    "TimeOfFlightLookupTableFromSimulation",
    "TimeResolution",
    "TofLutProvider",
    "default_parameters",
    "providers",
    "simulate_beamline",
]
