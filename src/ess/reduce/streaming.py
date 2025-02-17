# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
"""This module provides tools for running workflows in a streaming fashion."""

from abc import ABC, abstractmethod
from collections.abc import Callable
from copy import deepcopy
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
    if not isinstance(value, sc.Variable | sc.DataArray):
        return value
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
        return deepcopy(self._value)

    def _do_push(self, value: T) -> None:
        if self._value is None:
            self._value = deepcopy(value)
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
        accumulators: dict[sciline.typing.Key, Accumulator | Callable[..., Accumulator]]
        | tuple[sciline.typing.Key, ...],
        allow_bypass: bool = False,
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
            dict mapping keys to accumulator instances can be passed. If a dict value is
            a callable, base_workflow.bind_and_call(value) is used to make an instance.
        allow_bypass:
            If True, allow bypassing accumulators for keys that are not in the
            accumulators dict. This is useful for dynamic keys that are not "terminated"
            in any accumulator. USE WITH CARE! This will lead to incorrect results
            unless the values for these keys are valid for all chunks comprised in the
            final accumulators at the point where :py:meth:`finalize` is called.
        """
        workflow = sciline.Pipeline()
        for key in target_keys:
            workflow[key] = base_workflow[key]
        for key in dynamic_keys:
            workflow[key] = None  # hack to prune branches

        self._dynamic_keys = set(dynamic_keys)

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

        # Map each accumulator to its dependent dynamic keys
        graph = workflow.underlying_graph
        self._accumulator_dependencies = {
            acc_key: nx.ancestors(graph, acc_key) & self._dynamic_keys
            for acc_key in self._accumulators
            if acc_key in graph
        }

        # Depending on the target_keys, some accumulators can be unused and should not
        # be computed when adding a chunk.
        self._accumulators = {
            key: value for key, value in self._accumulators.items() if key in graph
        }
        # Create accumulators unless instances were passed. This allows for initializing
        # accumulators with arguments that depend on the workflow such as bin edges,
        # which would otherwise be hard to obtain.
        self._accumulators = {
            key: value
            if isinstance(value, Accumulator)
            else base_workflow.bind_and_call(value)
            for key, value in self._accumulators.items()
        }
        self._target_keys = target_keys
        self._allow_bypass = allow_bypass

    def add_chunk(
        self, chunks: dict[sciline.typing.Key, Any]
    ) -> dict[sciline.typing.Key, Any]:
        """
        Legacy interface for accumulating values from chunks and finalizing the result.

        It is recommended to use :py:meth:`accumulate` and :py:meth:`finalize` instead.

        Parameters
        ----------
        chunks:
            Chunks to be processed.

        Returns
        -------
        :
            Finalized result.
        """
        self.accumulate(chunks)
        return self.finalize()

    def accumulate(self, chunks: dict[sciline.typing.Key, Any]) -> None:
        """
        Accumulate values from chunks without finalizing the result.

        Parameters
        ----------
        chunks:
            Chunks to be processed.

        Raises
        ------
        ValueError
            If non-dynamic keys are provided in chunks.
            If accumulator computation requires dynamic keys not provided in chunks.
        """
        non_dynamic = set(chunks) - self._dynamic_keys
        if non_dynamic:
            raise ValueError(
                f"Can only update dynamic keys. Got non-dynamic keys: {non_dynamic}"
            )

        accumulators_to_update = []
        for acc_key, deps in self._accumulator_dependencies.items():
            if deps.isdisjoint(chunks.keys()):
                continue
            if not deps.issubset(chunks.keys()):
                raise ValueError(
                    f"Accumulator '{acc_key}' requires dynamic keys "
                    f"{deps - chunks.keys()} not provided in the current chunk."
                )
            accumulators_to_update.append(acc_key)

        for key, value in chunks.items():
            self._process_chunk_workflow[key] = value
            # There can be dynamic keys that do not "terminate" in any accumulator. In
            # that case, we need to make sure they can be and are used when computing
            # the target keys.
            if self._allow_bypass:
                self._finalize_workflow[key] = value
        to_accumulate = self._process_chunk_workflow.compute(accumulators_to_update)
        for key, processed in to_accumulate.items():
            self._accumulators[key].push(processed)

    def finalize(self) -> dict[sciline.typing.Key, Any]:
        """
        Get the final result by computing the target keys based on accumulated values.

        Returns
        -------
        :
            Finalized result.
        """
        for key in self._accumulators:
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
