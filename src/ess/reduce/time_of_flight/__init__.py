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
    NumberOfPulses,
    NumberOfSimulatedNeutrons,
    PulsePeriod,
    PulseStride,
    SimulationResults,
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
    "NumberOfPulses",
    "NumberOfSimulatedNeutrons",
    "PulsePeriod",
    "PulseStride",
    "PulseStrideOffset",
    "SimulationResults",
    "SourcePosition",
    "TimeOfFlightLookupTable",
    "TimeOfFlightLookupTableFilename",
    "TimeResolution",
    "TofLutWorkflow",
    "providers",
    "simulate_chopper_cascade_using_tof",
]
