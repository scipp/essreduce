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
    TofLookupTableWorkflow,
    simulate_chopper_cascade_using_tof,
)
from .types import (
    DetectorLtotal,
    MonitorLtotal,
    PulseStrideOffset,
    TimeOfFlightLookupTable,
    TimeOfFlightLookupTableFilename,
    ToaDetector,
    TofDetector,
    TofMonitor,
)
from .workflow import GenericTofWorkflow

__all__ = [
    "DetectorLtotal",
    "DiskChoppers",
    "DistanceResolution",
    "GenericTofWorkflow",
    "LookupTableRelativeErrorThreshold",
    "LtotalRange",
    "MonitorLtotal",
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
    "ToaDetector",
    "TofDetector",
    "TofLookupTableWorkflow",
    "TofMonitor",
    "providers",
    "simulate_chopper_cascade_using_tof",
]
