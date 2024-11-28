# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable, MutableSet, Sequence
from typing import Any, TypeVar

import networkx as nx
from sciline import Pipeline
from sciline._utils import key_name
from sciline.typing import Key

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
