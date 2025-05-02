# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

"""
Utilities for computing real neutron time-of-flight from chopper settings and
neutron time-of-arrival at the detectors.
"""

from .eto_to_tof import (
    default_parameters,
    providers,
    resample_detector_time_of_flight_data,
    resample_monitor_time_of_flight_data,
)
from .simulation import simulate_beamline
from .to_events import to_events
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
    ResampledDetectorTofData,
    ResampledMonitorTofData,
    SimulationResults,
    TimeOfFlightLookupTable,
    TimeResolution,
)
from .workflow import GenericTofWorkflow

__all__ = [
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
    "ResampledDetectorTofData",
    "ResampledMonitorTofData",
    "SimulationResults",
    "TimeOfFlightLookupTable",
    "TimeResolution",
    "default_parameters",
    "providers",
    "resample_detector_time_of_flight_data",
    "resample_monitor_time_of_flight_data",
    "simulate_beamline",
    "to_events",
]
