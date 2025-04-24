# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
from collections.abc import Iterable

import sciline

from ..nexus import GenericNeXusWorkflow
from .eto_to_tof import default_parameters, providers


def GenericTofWorkflow(
    *,
    run_types: Iterable[sciline.typing.Key] | None = None,
    monitor_types: Iterable[sciline.typing.Key] | None = None,
) -> sciline.Pipeline:
    """ """
    wf = GenericNeXusWorkflow()

    for provider in providers():
        wf.insert(provider)

    for key, value in default_parameters().items():
        wf[key] = value

    if run_types is not None or monitor_types is not None:
        from ..utils import prune_type_vars

        prune_type_vars(wf, run_types=run_types, monitor_types=monitor_types)

    return wf
