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
    PulsePeriod,
    PulseStride,
    SimulationResults,
    SourcePosition,
    TimeResolution,
    TofLutWorkflow,
    simulate_chopper_cascade_using_tof,
)

# from .simulation import simulate_beamline
from .types import (
    DetectorLtotal,
    DetectorTofData,
    # DistanceResolution,
    # LookupTableRelativeErrorThreshold,
    # LtotalRange,
    MonitorLtotal,
    MonitorTofData,
    # PulsePeriod,
    # PulseStride,
    # PulseStrideOffset,
    # SimulationResults,
    TimeOfFlightLookupTable,
    TimeOfFlightLookupTableFilename,
    # TimeResolution,
)
from .workflow import GenericTofWorkflow  # , TofLutProvider

__all__ = [
    "DetectorLtotal",
    "DetectorTofData",
    "DiskChoppers",
    "DistanceResolution",
    "LookupTableRelativeErrorThreshold",
    "LtotalRange",
    "MonitorLtotal",
    "MonitorTofData",
    "PulsePeriod",
    "PulseStride",
    "PulseStrideOffset",
    "SimulationResults",
    "SourcePosition",
    "TimeOfFlightLookupTable",
    "TimeOfFlightLookupTableFilename",
    "TimeResolution",
    # "TofLutProvider",
    "providers",
    # "simulate_beamline",
    "GenericTofWorkflow",
    "TofLutWorkflow",
    "simulate_chopper_cascade_using_tof",
]

# __all__ = [
#     "DetectorLtotal",
#     "DetectorTofData",
#     "DetectorTofData",
#     "DistanceResolution",
#     "GenericTofWorkflow",
#     "LookupTableRelativeErrorThreshold",
#     "LtotalRange",
#     "MonitorLtotal",
#     "MonitorTofData",
#     "MonitorTofData",
#     "PulsePeriod",
#     "PulseStride",
#     "PulseStrideOffset",
#     "SimulationResults",
#     "TimeOfFlightLookupTable",
#     "TimeOfFlightLookupTableFilename",
#     "TimeResolution",
#     # "TofLutProvider",
#     # "default_parameters",
#     "providers",
#     "simulate_beamline",
# ]
