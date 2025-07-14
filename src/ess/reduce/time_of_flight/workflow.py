# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from collections.abc import Iterable

import sciline
import scipp as sc

from ..nexus import GenericNeXusWorkflow
from . import eto_to_tof
from .types import (
    PulseStrideOffset,
    TimeOfFlightLookupTable,
    TimeOfFlightLookupTableFilename,
)


def load_tof_lookup_table(
    filename: TimeOfFlightLookupTableFilename,
) -> TimeOfFlightLookupTable:
    return TimeOfFlightLookupTable(sc.io.load_hdf5(filename))


def GenericTofWorkflow(
    *,
    run_types: Iterable[sciline.typing.Key],
    monitor_types: Iterable[sciline.typing.Key],
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

    Returns
    -------
    :
        The workflow.
    """
    wf = GenericNeXusWorkflow(run_types=run_types, monitor_types=monitor_types)

    for provider in eto_to_tof.providers():
        wf.insert(provider)

    wf.insert(load_tof_lookup_table)

    # Default parameters
    wf[PulseStrideOffset] = None

    return wf
