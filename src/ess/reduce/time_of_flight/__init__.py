# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""
Utilities for computing real neutron time-of-flight from chopper settings and
neutron time-of-arrival at the detectors.
"""

from ..nexus.types import DiskChoppers
from .eto_to_tof import providers
from .lut import (
    DistanceResolution,
    LookupTableRelativeErrorThreshold,
    LtotalRange,
    NumberOfSimulatedNeutrons,
    PulsePeriod,
    PulseStride,
    SimulationResults,
    SimulationSeed,
    SourcePosition,
    TimeResolution,
    TofLutWorkflow,
    simulate_chopper_cascade_using_tof,
)
from .types import (
    DetectorLtotal,
    DetectorTofData,
    MonitorLtotal,
    MonitorTofData,
    PulseStrideOffset,
    TimeOfFlightLookupTable,
    TimeOfFlightLookupTableFilename,
)
from .workflow import GenericTofWorkflow

__all__ = [
    "DetectorLtotal",
    "DetectorTofData",
    "DiskChoppers",
    "DistanceResolution",
    "GenericTofWorkflow",
    "LookupTableRelativeErrorThreshold",
    "LtotalRange",
    "MonitorLtotal",
    "MonitorTofData",
    "NumberOfSimulatedNeutrons",
    "PulsePeriod",
    "PulseStride",
    "PulseStrideOffset",
    "SimulationResults",
    "SimulationSeed",
    "SourcePosition",
    "TimeOfFlightLookupTable",
    "TimeOfFlightLookupTableFilename",
    "TimeResolution",
    "TofLutWorkflow",
    "providers",
    "simulate_chopper_cascade_using_tof",
]
