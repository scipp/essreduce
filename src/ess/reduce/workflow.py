# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Iterable, MutableSet
from types import UnionType
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import networkx as nx
from sciline import Pipeline
from sciline.data_graph import DataGraph
from sciline.typing import Key

from .parameter import Parameter, keep_default, parameter_registry

if TYPE_CHECKING:
    import graphviz

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


def register_workflow(cls: type[Workflow]) -> type[Workflow]:
    workflow_registry.add(cls)
    return cls


def _get_defaults_from_workflow(workflow: Pipeline) -> dict[Key, Any]:
    nodes = workflow.underlying_graph.nodes
    return {key: values["value"] for key, values in nodes.items() if "value" in values}


class Workflow:
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    @property
    def typical_outputs(self) -> tuple[Key, ...]:
        """Return a tuple of outputs that are used regularly."""
        return self.pipeline.typical_outputs

    @property
    def possible_outputs(self) -> tuple[Key, ...]:
        """All possible outputs."""
        return tuple(self.pipeline.underlying_graph.nodes)

    @property
    def _param_value_setters(self) -> dict[Key, Callable[[Pipeline, Any], Pipeline]]:
        return {}

    def parameters(self, outputs: tuple[Key, ...]) -> dict[Key, Parameter]:
        """Return a dictionary of parameters for the workflow."""
        subgraph = set(outputs)
        graph = self.pipeline.underlying_graph
        for key in outputs:
            subgraph.update(nx.ancestors(graph, key))
        defaults = _get_defaults_from_workflow(self.pipeline)
        return {
            key: param.with_default(defaults.get(key, keep_default))
            for key, param in parameter_registry.items()
            if key in subgraph
        }

    def __setitem__(self, key: Key, value: DataGraph | Any) -> None:
        """Set a value for a Key."""
        if (
            isinstance(value, tuple)
            and (setter := self._param_value_setters.get(key)) is not None
        ):
            self.pipeline = setter(self.pipeline, value)
        else:
            self.pipeline[key] = value

    def compute(self, tp: type | Iterable[type] | UnionType, **kwargs: Any) -> Any:
        """Run the workflow to compute outputs."""
        return self.pipeline.compute(tp, **kwargs)

    def visualize(self, tp: type | Iterable[type], **kwargs: Any) -> graphviz.Digraph:
        return self.pipeline.visualize(tp, **kwargs)

    def insert(self, provider, /) -> None:
        self.pipeline.insert(provider)
