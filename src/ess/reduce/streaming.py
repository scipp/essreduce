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
    def is_empty(self) -> bool:
        """
        Check if the accumulator is empty.

        Returns
        -------
        :
            True if the accumulator is empty, False otherwise.
        """
        return False

    @property
    def value(self) -> T:
        """
        Get the accumulated value.

        Returns
        -------
        :
            Accumulated value.

        Raises
        ------
        ValueError
            If the accumulator is empty.
        """
        if self.is_empty:
            raise ValueError("Cannot get value from empty accumulator")
        return self._get_value()

    @abstractmethod
    def _get_value(self) -> T:
        """Return the accumulated value, assuming it exists."""

    @abstractmethod
    def clear(self) -> None:
        """
        Clear the accumulator, resetting it to its initial state.
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
    def is_empty(self) -> bool:
        return self._value is None

    def _get_value(self) -> T:
        return deepcopy(self._value)

    def _do_push(self, value: T) -> None:
        if self._value is None:
            self._value = deepcopy(value)
        else:
            self._value += value

    def clear(self) -> None:
        """Clear the accumulated value."""
        self._value = None


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
    def is_empty(self) -> bool:
        return len(self._values) == 0

    def _get_value(self) -> T:
        # Naive and potentially slow implementation if values and/or window are large!
        return sc.reduce(self._values).sum()

    def _do_push(self, value: T) -> None:
        self._values.append(value)
        if len(self._values) > self._window:
            self._values.pop(0)

    def clear(self) -> None:
        """Clear the accumulated values."""
        self._values = []


class MinAccumulator(Accumulator):
    """Keeps the minimum value seen so far.

    Only supports scalar values.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cur_min: sc.Variable | None = None

    def _do_push(self, value: sc.Variable) -> None:
        if self._cur_min is None:
            self._cur_min = value
        else:
            self._cur_min = min(self._cur_min, value)

    @property
    def is_empty(self) -> bool:
        """Check if the accumulator has collected a minimum value."""
        return self._cur_min is None

    def _get_value(self) -> Any:
        return self._cur_min

    def clear(self) -> None:
        """Clear the accumulated minimum value."""
        self._cur_min = None


class MaxAccumulator(Accumulator):
    """Keeps the maximum value seen so far.

    Only supports scalar values.
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._cur_max: sc.Variable | None = None

    @property
    def is_empty(self) -> bool:
        """Check if the accumulator has collected a maximum value."""
        return self._cur_max is None

    def _do_push(self, value: sc.Variable) -> None:
        if self._cur_max is None:
            self._cur_max = value
        else:
            self._cur_max = max(self._cur_max, value)

    def _get_value(self) -> sc.Variable | None:
        return self._cur_max

    def clear(self) -> None:
        """Clear the accumulated maximum value."""
        self._cur_max = None


class StreamProcessor:
    """
    Wrap a base workflow for streaming processing of chunks.

    Note that this class can not determine if the workflow is valid for streamed
    processing based on the input keys. In particular, it is the responsibility of the
    user to ensure that the workflow is "linear" with respect to the dynamic keys up to
    the accumulation keys.

    Similarly, the stream processor cannot determine from the workflow structure whether
    context updates are compatible with the accumulated data. Accumulators are not
    cleared automatically. This is best illustrated with an example:

    - If the context is the detector rotation angle, and we accumulate I(Q) (or a
      prerequisite of I(Q)), then updating the detector angle context is compatible with
      previous data, assuming Q for each new chunk is computed based on the angle.
    - If the context is the sample temperature, and we accumulate I(Q), then updating
      the temperature context is not compatible with previous data. Accumulating I(Q, T)
      could be compatible in this case.

    Since the correctness cannot be determined from the workflow structure, we recommend
    implementing processing steps in a way to catch such problems. For example, adding
    the temperature as a coordinate to the I(Q) data array should allow for
    automatically raising in the accumulator if the temperature changes.
    """

    def __init__(
        self,
        base_workflow: sciline.Pipeline,
        *,
        dynamic_keys: tuple[sciline.typing.Key, ...],
        context_keys: tuple[sciline.typing.Key, ...] = (),
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
            Keys that are expected to be updated with each chunk. These keys cannot
            depend on each other or on context_keys.
        context_keys:
            Keys that define context for processing chunks and may change occasionally.
            These keys cannot overlap with dynamic_keys or depend on each other or on
            dynamic_keys.
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
        self._dynamic_keys = set(dynamic_keys)
        self._context_keys = set(context_keys)

        # Validate that dynamic and context keys do not overlap
        overlap = self._dynamic_keys & self._context_keys
        if overlap:
            raise ValueError(f"Keys cannot be both dynamic and context: {overlap}")

        # Check dynamic/context keys don't depend on other dynamic/context keys
        graph = base_workflow.underlying_graph
        special_keys = self._dynamic_keys | self._context_keys
        for key in special_keys:
            if key not in graph:
                continue
            ancestors = nx.ancestors(graph, key)
            special_ancestors = ancestors & special_keys
            downstream = 'Dynamic' if key in self._dynamic_keys else 'Context'
            if special_ancestors:
                raise ValueError(
                    f"{downstream} key '{key}' depends on other dynamic/context keys: "
                    f"{special_ancestors}. This is not supported."
                )

        workflow = sciline.Pipeline()
        for key in target_keys:
            workflow[key] = base_workflow[key]
        for key in dynamic_keys:
            workflow[key] = None  # hack to prune branches
        for key in context_keys:
            workflow[key] = None

        # Find and pre-compute static nodes as far down the graph as possible
        nodes = _find_descendants(workflow, dynamic_keys + context_keys)
        last_static = _find_parents(workflow, nodes) - nodes
        for key, value in base_workflow.compute(last_static).items():
            workflow[key] = value

        # Nodes that may need updating on context change but should be cached otherwise.
        dynamic_nodes = _find_descendants(workflow, dynamic_keys)
        # Nodes as far "down" in the graph as possible, right before the dynamic nodes.
        # This also includes target keys that are not dynamic but context-dependent.
        context_to_cache = (
            (_find_parents(workflow, dynamic_nodes) | set(target_keys)) - dynamic_nodes
        ) & _find_descendants(workflow, context_keys)
        graph = workflow.underlying_graph
        self._context_key_to_cached_context_nodes_map = {
            context_key: ({context_key} | nx.descendants(graph, context_key))
            & context_to_cache
            for context_key in self._context_keys
            if context_key in graph
        }

        self._context_workflow = workflow.copy()
        self._process_chunk_workflow = workflow.copy()
        self._finalize_workflow = workflow.copy()
        self._accumulators = (
            accumulators
            if isinstance(accumulators, dict)
            else {key: EternalAccumulator() for key in accumulators}
        )

        # Map each accumulator to its dependent dynamic keys
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

    def set_context(self, context: dict[sciline.typing.Key, Any]) -> None:
        """
        Set the context for processing chunks.

        Parameters
        ----------
        context:
            Context to be set.
        """
        needs_recompute = set()
        for key in context:
            if key not in self._context_keys:
                raise ValueError(f"Key '{key}' is not a context key")
            needs_recompute |= self._context_key_to_cached_context_nodes_map[key]
        for key, value in context.items():
            self._context_workflow[key] = value
        results = self._context_workflow.compute(needs_recompute)
        for key, value in results.items():
            if key in self._target_keys:
                # Context-dependent key is direct target, independent of dynamic nodes.
                self._finalize_workflow[key] = value
            else:
                self._process_chunk_workflow[key] = value

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

    def clear(self) -> None:
        """
        Clear all accumulators, resetting them to their initial state.

        This is useful for restarting a streaming computation without
        creating a new StreamProcessor instance.
        """
        for accumulator in self._accumulators.values():
            accumulator.clear()


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
