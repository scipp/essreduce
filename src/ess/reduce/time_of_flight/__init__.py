# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""
Utilities for computing real neutron time-of-flight from chopper settings and
neutron time-of-arrival at the detectors.
"""

from ..nexus.types import DiskChoppers

# from .eto_to_tof import providers
from .lut import (
    DistanceFromSampleRange,
    DistanceResolution,
    LookupTableRelativeErrorThreshold,
    NumberOfSimulatedNeutrons,
    PulsePeriod,
    PulseStride,
    SimulationResults,
    SimulationSeed,
    SourcePosition,
    TimeResolution,
    WavelengthLookupTableWorkflow,
    simulate_chopper_cascade_using_tof,
)
from .types import (
    DetectorLtotal,
    MonitorLtotal,
    PulseStrideOffset,
    ToaDetector,
    TofDetector,
    TofMonitor,
    WavelengthLookupTable,
    WavelengthLookupTableFilename,
)

# from .workflow import GenericTofWorkflow

__all__ = [
    "DetectorLtotal",
    "DiskChoppers",
    "DistanceResolution",
    "DistanceFromSampleRange",
    # "GenericTofWorkflow",
    "LookupTableRelativeErrorThreshold",
    "MonitorLtotal",
    "NumberOfSimulatedNeutrons",
    "PulsePeriod",
    "PulseStride",
    "PulseStrideOffset",
    "SimulationResults",
    "SimulationSeed",
    "SourcePosition",
    "WavelengthLookupTable",
    "WavelengthLookupTableFilename",
    "TimeResolution",
    "ToaDetector",
    "TofDetector",
    "TofMonitor",
    # "WavelengthLookupTableWorkflow",
    "WavelengthLookupTable",
    # "providers",
    "simulate_chopper_cascade_using_tof",
]
