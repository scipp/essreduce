# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""This module provides tools for running workflows in a streaming fashion."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any, Generic, TypeVar

import networkx as nx
import sciline
import scipp as sc

T = TypeVar('T')


def maybe_hist(value: T) -> T:
    """
    Convert value to a histogram if it is not already a histogram.

    This is the default pre-processing used by accumulators.

    Parameters
    ----------
    value:
        Value to be converted to a histogram.

    Returns
    -------
    :
        Histogram.
    """
    return value if value.bins is None else value.hist()


class Accumulator(ABC, Generic[T]):
    """
    Abstract base class for accumulators.

    Accumulators are used to accumulate values over multiple chunks.
    """

    def __init__(self, preprocess: Callable[[T], T] | None = maybe_hist) -> None:
        """
        Parameters
        ----------
        preprocess:
            Preprocessing function to be applied to pushed values prior to accumulation.
        """
        self._preprocess = preprocess

    def push(self, value: T) -> None:
        """
        Push a value to the accumulator.

        Parameters
        ----------
        value:
            Value to be pushed to the accumulator.
        """
        if self._preprocess is not None:
            value = self._preprocess(value)
        self._do_push(value)

    @abstractmethod
    def _do_push(self, value: T) -> None: ...

    @property
    @abstractmethod
    def value(self) -> T:
        """
        Get the accumulated value.

        Returns
        -------
        :
            Accumulated value.
        """


class EternalAccumulator(Accumulator[T]):
    """
    Simple accumulator that adds pushed values immediately.

    Does not support event data.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._value: T | None = None

    @property
    def value(self) -> T:
        return self._value.copy()

    def _do_push(self, value: T) -> None:
        if self._value is None:
            self._value = value.copy()
        else:
            self._value += value


class RollingAccumulator(Accumulator[T]):
    """
    Accumulator that adds pushed values to a rolling window.

    Does not support event data.
    """

    def __init__(self, window: int = 10, **kwargs: Any) -> None:
        """
        Parameters
        ----------
        window:
            Size of the rolling window.
        """
        super().__init__(**kwargs)
        self._window = window
        self._values: list[T] = []

    @property
    def value(self) -> T:
        # Naive and potentially slow implementation if values and/or window are large!
        return sc.reduce(self._values).sum()

    def _do_push(self, value: T) -> None:
        self._values.append(value)
        if len(self._values) > self._window:
            self._values.pop(0)


class Streaming:
    """Wrap a base workflow for streaming processing of chunks."""

    def __init__(
        self,
        base_workflow: sciline.Pipeline,
        dynamic_keys: tuple[sciline.typing.Key, ...],
        accumulation_keys: tuple[sciline.typing.Key, ...],
        target_keys: tuple[sciline.typing.Key, ...],
        accumulator: type[Accumulator] = EternalAccumulator,
    ) -> None:
        """
        Parameters
        ----------
        base_workflow:
            Workflow to be used for processing chunks.
        dynamic_keys:
            Keys that are expected to be updated with each chunk.
        accumulation_keys:
            Keys for which to accumulate values.
        target_keys:
            Keys to be computed and returned.
        accumulator:
            Accumulator class to use for accumulation.
        """
        workflow = sciline.Pipeline()
        for key in target_keys:
            workflow[key] = base_workflow[key]
        for key in dynamic_keys:
            workflow[key] = None  # hack to prune branches
        # Find static nodes as far down the graph as possible
        nodes = _find_descendants(workflow, dynamic_keys)
        parents = _find_parents(workflow, nodes) - _find_input_nodes(workflow) - nodes
        for key, value in base_workflow.compute(parents).items():
            workflow[key] = value
        self._process_chunk_workflow = workflow.copy()
        self._finalize_workflow = workflow.copy()
        self._accumulators = {key: accumulator() for key in accumulation_keys}
        self._target_keys = target_keys

    def add_chunk(
        self, chunks: dict[sciline.typing.Key, Any]
    ) -> dict[sciline.typing.Key, Any]:
        for key, value in chunks.items():
            self._process_chunk_workflow[key] = value
        to_accumulate = self._process_chunk_workflow.compute(self._accumulators)
        for key, processed in to_accumulate.items():
            self._accumulators[key].push(processed)
            self._finalize_workflow[key] = self._accumulators[key].value
        return self._finalize_workflow.compute(self._target_keys)


def _find_descendants(
    workflow: sciline.Pipeline, keys: tuple[sciline.typing.Key, ...]
) -> set[sciline.typing.Key]:
    graph = workflow.underlying_graph
    descendants = set()
    for key in keys:
        descendants |= nx.descendants(graph, key)
    return descendants


def _find_input_nodes(workflow: sciline.Pipeline) -> set[sciline.typing.Key]:
    graph = workflow.underlying_graph
    return {key for key in graph if graph.in_degree(key) == 0}


def _find_parents(
    workflow: sciline.Pipeline, keys: tuple[sciline.typing.Key, ...]
) -> set[sciline.typing.Key]:
    graph = workflow.underlying_graph
    parents = set()
    for key in keys:
        parents |= set(graph.predecessors(key))
    return parents
