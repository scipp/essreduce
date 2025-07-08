# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable, Iterable
from enum import Enum, auto

import sciline
import scipp as sc

from ..nexus import GenericNeXusWorkflow
from . import eto_to_tof, lookup_table, simulation
from .types import (
    CommonTimeOfFlightLookupTable,
    DistanceResolution,
    LookupTableRelativeErrorThreshold,
    PulsePeriod,
    PulseStride,
    PulseStrideOffset,
    RunType,
    TimeOfFlightLookupTable,
    TimeOfFlightLookupTableFilename,
    TimeResolution,
)


class TofLutProvider(Enum):
    """Provider for the time-of-flight lookup table."""

    FILE = auto()  # From file
    TOF = auto()  # Computed with 'tof' package from chopper settings
    MCSTAS = auto()  # McStas simulation (not implemented yet)


def load_tof_lookup_table(
    filename: TimeOfFlightLookupTableFilename,
) -> CommonTimeOfFlightLookupTable:
    return CommonTimeOfFlightLookupTable(sc.io.load_hdf5(filename))


def common_tof_lookup_table_to_runtype(
    table: CommonTimeOfFlightLookupTable,
) -> TimeOfFlightLookupTable[RunType]:
    return TimeOfFlightLookupTable[RunType](table)


def providers() -> tuple[Callable]:
    """
    Providers of the time-of-flight workflow.
    """
    return (
        lookup_table.compute_tof_lookup_table,
        common_tof_lookup_table_to_runtype,
        eto_to_tof.detector_time_of_flight_data,
        eto_to_tof.monitor_time_of_flight_data,
        eto_to_tof.detector_ltotal_from_straight_line_approximation,
        eto_to_tof.monitor_ltotal_from_straight_line_approximation,
    )


def default_parameters() -> dict:
    """
    Default parameters for computing a tof lookup table from Tof/McStas simulation
    results.
    """
    return {
        PulsePeriod: 1.0 / sc.scalar(14.0, unit="Hz"),
        PulseStride: 1,
        PulseStrideOffset: None,
        DistanceResolution: sc.scalar(0.1, unit="m"),
        TimeResolution: sc.scalar(250.0, unit='us'),
        LookupTableRelativeErrorThreshold: 0.1,
    }


def GenericTofWorkflow(
    *,
    run_types: Iterable[sciline.typing.Key],
    monitor_types: Iterable[sciline.typing.Key],
    tof_lut_provider: TofLutProvider = TofLutProvider.FILE,
) -> sciline.Pipeline:
    """
    Generic workflow for computing the neutron time-of-flight for detector and monitor
    data.

    This workflow builds on the ``GenericNeXusWorkflow`` and computes time-of-flight
    from a lookup table that is created from the chopper settings, detector Ltotal and
    the neutron time-of-arrival.

    It is possible to limit which run types and monitor types
    are supported by the returned workflow.
    This is useful to reduce the size of the workflow and make it easier to inspect.
    Make sure to add *all* required run types and monitor types when using this feature.

    Attention
    ---------
    Filtering by run type and monitor type does not work with nested type vars.
    E.g., if you have a type like ``Outer[Inner[RunType]]``, this type and its
    provider will be removed.

    Parameters
    ----------
    run_types:
        List of run types to include in the workflow.
        Constrains the possible values of :class:`ess.reduce.nexus.types.RunType`.
    monitor_types:
        List of monitor types to include in the workflow.
        Constrains the possible values of :class:`ess.reduce.nexus.types.MonitorType`
        and :class:`ess.reduce.nexus.types.Component`.
    tof_lut_provider:
        Specifies how the time-of-flight lookup table is provided:
        - FILE: Read from a file
        - TOF: Computed from chopper settings using the 'tof' package
        - MCSTAS: From McStas simulation (not implemented yet)

    Returns
    -------
    :
        The workflow.
    """
    wf = GenericNeXusWorkflow(run_types=run_types, monitor_types=monitor_types)

    for provider in providers():
        wf.insert(provider)

    if tof_lut_provider == TofLutProvider.FILE:
        wf.insert(load_tof_lookup_table)
    else:
        wf.insert(lookup_table.use_tof_lookup_table_from_simulation)
        if tof_lut_provider == TofLutProvider.TOF:
            wf.insert(simulation.simulate_chopper_cascade_using_tof)
        if tof_lut_provider == TofLutProvider.MCSTAS:
            raise NotImplementedError("McStas simulation not implemented yet")

    for key, value in default_parameters().items():
        wf[key] = value

    return wf
