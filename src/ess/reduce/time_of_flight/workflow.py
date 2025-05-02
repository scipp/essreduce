# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from collections.abc import Iterable

import sciline

from ..nexus import GenericNeXusWorkflow
from ..utils import prune_type_vars
from .eto_to_tof import default_parameters, providers


def GenericTofWorkflow(
    *,
    run_types: Iterable[sciline.typing.Key] | None = None,
    monitor_types: Iterable[sciline.typing.Key] | None = None,
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
        List of run types to include in the workflow. If not provided, all run types
        are included.
        Must be a possible value of :class:`ess.reduce.nexus.types.RunType`.
    monitor_types:
        List of monitor types to include in the workflow. If not provided, all monitor
        types are included.
        Must be a possible value of :class:`ess.reduce.nexus.types.MonitorType`.

    Returns
    -------
    :
        The workflow.
    """
    wf = GenericNeXusWorkflow(run_types=run_types, monitor_types=monitor_types)

    for provider in providers():
        wf.insert(provider)
    for key, value in default_parameters().items():
        wf[key] = value

    if run_types is not None or monitor_types is not None:
        prune_type_vars(wf, run_types=run_types, monitor_types=monitor_types)

    return wf
