# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

from collections.abc import Iterable
from typing import Any

import sciline

from .nexus.types import MonitorType, RunType


def prune_type_vars(
    workflow: sciline.Pipeline,
    *,
    run_types: Iterable[sciline.typing.Key] | None,
    monitor_types: Iterable[sciline.typing.Key] | None,
) -> None:
    # Remove all nodes that use a run type or monitor types that is
    # not listed in the function arguments.
    excluded_run_types = excluded_type_args(RunType, run_types)
    excluded_monitor_types = excluded_type_args(MonitorType, monitor_types)
    excluded_types = excluded_run_types | excluded_monitor_types

    graph = workflow.underlying_graph
    to_remove = [
        node for node in graph if excluded_types & set(getattr(node, "__args__", set()))
    ]
    graph.remove_nodes_from(to_remove)


def excluded_type_args(
    type_var: Any, keep: Iterable[sciline.typing.Key] | None
) -> set[sciline.typing.Key]:
    if keep is None:
        return set()
    return set(type_var.__constraints__) - set(keep)
