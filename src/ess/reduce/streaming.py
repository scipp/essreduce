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


class StreamProcessor:
    """
    Wrap a base workflow for streaming processing of chunks.

    Note that this class can not determine if the workflow is valid for streamed
    processing based on the input keys. In particular, it is the responsibility of the
    user to ensure that the workflow is "linear" with respect to the dynamic keys up to
    the accumulation keys.
    """

    def __init__(
        self,
        base_workflow: sciline.Pipeline,
        *,
        dynamic_keys: tuple[sciline.typing.Key, ...],
        target_keys: tuple[sciline.typing.Key, ...],
        accumulators: dict[sciline.typing.Key, Accumulator]
        | tuple[sciline.typing.Key, ...],
    ) -> None:
        """
        Create a stream processor.

        Parameters
        ----------
        base_workflow:
            Workflow to be used for processing chunks.
        dynamic_keys:
            Keys that are expected to be updated with each chunk.
        target_keys:
            Keys to be computed and returned.
        accumulators:
            Keys at which to accumulate values and their accumulators. If a tuple is
            passed, :py:class:`EternalAccumulator` is used for all keys. Otherwise, a
            dict mapping keys to accumulator instances can be passed.
        """
        workflow = sciline.Pipeline()
        for key in target_keys:
            workflow[key] = base_workflow[key]
        for key in dynamic_keys:
            workflow[key] = None  # hack to prune branches

        # Find and pre-compute static nodes as far down the graph as possible
        # See also https://github.com/scipp/sciline/issues/148.
        nodes = _find_descendants(workflow, dynamic_keys)
        parents = _find_parents(workflow, nodes) - nodes
        for key, value in base_workflow.compute(parents).items():
            workflow[key] = value

        self._process_chunk_workflow = workflow.copy()
        self._finalize_workflow = workflow.copy()
        self._accumulators = (
            accumulators
            if isinstance(accumulators, dict)
            else {key: EternalAccumulator() for key in accumulators}
        )
        # Depending on the target_keys, some accumulators can be unused and should not
        # be computed when adding a chunk.
        self._accumulators = {
            key: value
            for key, value in self._accumulators.items()
            if key in self._process_chunk_workflow.underlying_graph
        }
        self._target_keys = target_keys

    def add_chunk(
        self, chunks: dict[sciline.typing.Key, Any]
    ) -> dict[sciline.typing.Key, Any]:
        for key, value in chunks.items():
            self._process_chunk_workflow[key] = value
            # There can be dynamic keys that do not "terminate" in any accumulator. In
            # that case, we need to make sure they can be and are used when computing
            # the target keys.
            self._finalize_workflow[key] = value
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
    return descendants | set(keys)


def _find_parents(
    workflow: sciline.Pipeline, keys: tuple[sciline.typing.Key, ...]
) -> set[sciline.typing.Key]:
    graph = workflow.underlying_graph
    parents = set()
    for key in keys:
        parents |= set(graph.predecessors(key))
    return parents
