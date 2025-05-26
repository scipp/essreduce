# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""
Utilities for computing real neutron time-of-flight from chopper settings and
neutron time-of-arrival at the detectors.
"""

from .eto_to_tof import default_parameters, providers
from .simulation import simulate_beamline
from .types import (
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
    TimeResolution,
)
from .workflow import GenericTofWorkflow, TofLutProvider

__all__ = [
    "DetectorLtotal",
    "DetectorTofData",
    "DetectorTofData",
    "DistanceResolution",
    "GenericTofWorkflow",
    "LookupTableRelativeErrorThreshold",
    "LtotalRange",
    "MonitorLtotal",
    "MonitorTofData",
    "MonitorTofData",
    "PulsePeriod",
    "PulseStride",
    "PulseStrideOffset",
    "SimulationResults",
    "TimeOfFlightLookupTable",
    "TimeOfFlightLookupTableFilename",
    "TimeResolution",
    "TofLutProvider",
    "default_parameters",
    "providers",
    "simulate_beamline",
]
