# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from types import UnionType
from typing import TYPE_CHECKING, Any, Callable, TypeVar

import networkx as nx
from sciline import Pipeline
from sciline.data_graph import DataGraph
from sciline.typing import Key

from .parameter import Parameter

if TYPE_CHECKING:
    import graphviz

T = TypeVar('T')


class Workflow(ABC):
    def __init__(self, pipeline: Pipeline) -> None:
        self.pipeline = pipeline

    @property
    @abstractmethod
    def typical_outputs(self) -> tuple[Key, ...]:
        """Return a tuple of outputs that are used regularly."""

    @property
    def possible_outputs(self) -> tuple[Key, ...]:
        """All possible outputs."""
        return tuple(self.pipeline.underlying_graph.nodes)

    @abstractmethod
    def _parameters(self) -> dict[Key, Parameter]:
        """Return a dictionary of parameters for the workflow."""

    @property
    def _param_value_setters(self) -> dict[Key, Callable[[Pipeline, Any], Pipeline]]:
        return {}

    def parameters(self, outputs: tuple[Key, ...]) -> dict[Key, Parameter]:
        """Return a dictionary of parameters for the workflow."""
        subgraph = set(outputs)
        graph = self.pipeline.underlying_graph
        for key in outputs:
            subgraph.update(nx.ancestors(graph, key))
        return {
            key: param for key, param in self._parameters().items() if key in subgraph
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
