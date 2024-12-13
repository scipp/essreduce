# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import itertools
from collections.abc import Callable, Iterable, MutableSet, Sequence
from typing import Any, TypeVar

import networkx as nx
from sciline import Pipeline, Scope, ScopeTwoParams
from sciline._utils import key_name
from sciline.typing import Key

from .nexus.types import MonitorType
from .parameter import Parameter, keep_default, parameter_mappers, parameter_registry

T = TypeVar("T")


class WorkflowRegistry(MutableSet):
    def __init__(self):
        self._workflows: dict[str, type] = {}

    def __contains__(self, item: object) -> bool:
        return item in self._workflows.values()

    def __iter__(self):
        return iter(self._workflows.values())

    def __len__(self) -> int:
        return len(self._workflows)

    def add(self, value: type) -> None:
        key = value.__qualname__
        self._workflows[key] = value

    def discard(self, value: type) -> None:
        self._workflows = {k: v for k, v in self._workflows.items() if v != value}


workflow_registry = WorkflowRegistry()


def register_workflow(cls: Callable[[], Pipeline]) -> Callable[[], Pipeline]:
    workflow_registry.add(cls)
    return cls


def _get_defaults_from_workflow(workflow: Pipeline) -> dict[Key, Any]:
    nodes = workflow.underlying_graph.nodes
    return {key: values["value"] for key, values in nodes.items() if "value" in values}


def get_typical_outputs(pipeline: Pipeline) -> tuple[Key, ...]:
    if (typical_outputs := getattr(pipeline, "typical_outputs", None)) is None:
        graph = pipeline.underlying_graph
        sink_nodes = [node for node, degree in graph.out_degree if degree == 0]
        return sorted(_with_pretty_names(sink_nodes))
    return _with_pretty_names(typical_outputs)


def get_possible_outputs(pipeline: Pipeline) -> tuple[Key, ...]:
    return sorted(_with_pretty_names(tuple(pipeline.underlying_graph.nodes)))


def _with_pretty_names(outputs: Sequence[Key]) -> tuple[tuple[str, Key], ...]:
    """Add a more readable string representation without full module path."""
    return tuple((key_name(output), output) for output in outputs)


def get_parameters(
    pipeline: Pipeline, outputs: tuple[Key, ...]
) -> dict[Key, Parameter]:
    """Return a dictionary of parameters for the workflow."""
    subgraph = set(outputs)
    graph = pipeline.underlying_graph
    for key in outputs:
        subgraph.update(nx.ancestors(graph, key))
    defaults = _get_defaults_from_workflow(pipeline)
    return {
        key: param.with_default(defaults.get(key, keep_default))
        for key, param in parameter_registry.items()
        if key in subgraph
    }


def assign_parameter_values(pipeline: Pipeline, values: dict[Key, Any]) -> Pipeline:
    """Set a value for a parameter in the pipeline."""
    pipeline = pipeline.copy()
    for key, value in values.items():
        if (mapper := parameter_mappers.get(key)) is not None:
            pipeline = mapper(pipeline, value)
        else:
            pipeline[key] = value
    return pipeline


def prune_nexus_domain_types(
    workflow: Pipeline,
    *,
    targets: Iterable[Key] | None = None,
    targets_per_run: Iterable[type[Scope]] | None = None,
    targets_per_run_and_monitor: Iterable[type[ScopeTwoParams]] | None = None,
    run_types: Iterable[Key] | None = None,
    monitor_types: Iterable[Key] | None = None,
) -> Pipeline:
    """Remove unused types from a workflow.

    This function removes all nodes from a workflow that are not needed to compute
    given targets with given run and monitor types.

    Warning
    -------
    This modifies the input workflow.

    Parameters
    ----------
    workflow:
        Workflow to remove types from.
    targets:
        Unparametrized types to keep.
    targets_per_run:
        Types parametrized by run type to keep.
    targets_per_run_and_monitor:
        Types parametrized by run type and monitor type to keep.
    run_types:
        List of run types to include in the workflow. If not provided, all run types
        are included.
    monitor_types:
        List of monitor types to include in the workflow. If not provided, all monitor
        types are included.

    Returns
    -------
    :
        The pruned workflow.
        The same object as the `workflow` argument.

    Examples
    --------
    This creates a workflow that can compute `DetectorData[SampleRun]` and all its
    dependencies but nothing else.
    I.e., providers for `DetectorData[backgroundRun]`,
    `MonitorData[SampleRun, Monitor1]`, etc., are removed.

        >>> from ess.reduce.nexus import GenericNeXusWorkflow, types
        >>> workflow = GenericNeXusWorkflow()
        >>> prune_nexus_domain_types(
        ...    workflow,
        ...    targets_per_run=[types.DetectorData],
        ...    run_types=[types.SampleRun],
        ... )

    To also keep monitors, use, e.g.,

        >>> from ess.reduce.nexus import GenericNeXusWorkflow, types
        >>> workflow = GenericNeXusWorkflow()
        >>> prune_nexus_domain_types(
        ...    workflow,
        ...    targets_per_run=[types.DetectorData],
        ...    targets_per_run_and_monitor=[types.MonitorData],
        ...    run_types=[types.SampleRun],
        ...    monitor_types=[types.Monitor1, types.Monitor2],
        ... )

    It is also possible to specify unparametrized tppes.
    The following is similar to `workflow = workflow[NeXusSourceName]` but
    `prune_nexus_domain_types` allows combining this with parametrized types and
    specifying multiple targets.

        >>> from ess.reduce.nexus import GenericNeXusWorkflow, types
        >>> workflow = GenericNeXusWorkflow()
        >>> prune_nexus_domain_types(
        ...    workflow,
        ...    targets=[types.NeXusSourceName],
        ... )
    """
    if run_types is not None and targets_per_run is None:
        raise ValueError(
            "`targets_per_run` must be provided if" "`run_types` is provided."
        )
    if monitor_types is not None and targets_per_run_and_monitor is None:
        raise ValueError(
            "`targets_per_run_and_monitor` must be provided if"
            "`monitor_types` is provided."
        )

    graph = workflow.underlying_graph
    # Find all ancestors of the target types ...
    ancestors = set()
    if targets is not None:
        _add_ancestors(ancestors, graph, targets)
    if targets_per_run is not None:
        run_types = list(run_types)  # To support iterators because we iterate twice.
        _add_ancestors(ancestors, graph, targets_per_run, run_types)
    if targets_per_run_and_monitor is not None:
        _add_ancestors(
            ancestors,
            graph,
            targets_per_run_and_monitor,
            run_types,
            _monitor_types_or_default(monitor_types),
        )
    # ... and remove everything else.
    graph.remove_nodes_from(set(graph.nodes) - ancestors)
    return workflow


def _add_ancestors(
    out: set[type],
    graph: nx.DiGraph,
    targets: Iterable[Any],
    *constraints: Iterable[type],
) -> None:
    for target, *types in itertools.product(targets, *constraints):
        if len(types) == 0:
            t = target
        elif len(types) == 1:
            t = target[types[0]]
        else:
            t = target[types[0], types[1]]
        out |= nx.ancestors(graph, t)
        out.add(t)


def _monitor_types_or_default(
    monitor_types: Iterable[Key] | None = None,
) -> Iterable[Key]:
    if monitor_types is None:
        return MonitorType.__constraints__
    return monitor_types
