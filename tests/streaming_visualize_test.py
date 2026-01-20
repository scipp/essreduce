# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from typing import Generic, NewType, TypeVar

import graphviz
import sciline

from ess.reduce import streaming

# Test types for building workflows
DynamicA = NewType('DynamicA', float)
DynamicB = NewType('DynamicB', float)
StaticA = NewType('StaticA', float)
StaticB = NewType('StaticB', float)
Context1 = NewType('Context1', float)
ContextB = NewType('ContextB', float)
AccumA = NewType('AccumA', float)
AccumB = NewType('AccumB', float)
Intermediate = NewType('Intermediate', float)
Target = NewType('Target', float)


def make_static_a() -> StaticA:
    return StaticA(1.0)


def make_static_b(a: StaticA) -> StaticB:
    return StaticB(a * 2)


def make_intermediate(dyn: DynamicA, static: StaticB) -> Intermediate:
    return Intermediate(dyn + static)


def make_accum_a(inter: Intermediate) -> AccumA:
    return AccumA(inter)


def make_accum_b(dyn: DynamicB) -> AccumB:
    return AccumB(dyn)


def make_target(a: AccumA, b: AccumB) -> Target:
    return Target(a / b)


def make_context_dependent(ctx: Context1) -> StaticB:
    return StaticB(ctx * 2)


class TestClassifyNodes:
    """Tests for the _classify_nodes method."""

    def test_static_nodes_have_no_dynamic_or_context_ancestors(self) -> None:
        workflow = sciline.Pipeline(
            (make_static_a, make_static_b, make_intermediate, make_accum_a, make_target)
        )
        workflow[DynamicA] = None
        workflow[AccumB] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            target_keys=(Target,),
            accumulators=(AccumA,),
        )

        graph = workflow.underlying_graph
        classifications = processor._classify_nodes(graph)

        assert StaticA in classifications['static']
        assert StaticB in classifications['static']

    def test_dynamic_keys_are_classified(self) -> None:
        workflow = sciline.Pipeline(
            (
                make_static_a,
                make_static_b,
                make_intermediate,
                make_accum_a,
                make_accum_b,
                make_target,
            )
        )
        workflow[DynamicA] = None
        workflow[DynamicB] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA, DynamicB),
            target_keys=(Target,),
            accumulators=(AccumA, AccumB),
        )

        graph = workflow.underlying_graph
        classifications = processor._classify_nodes(graph)

        assert DynamicA in classifications['dynamic_keys']
        assert DynamicB in classifications['dynamic_keys']
        # Dynamic keys should not be in dynamic_nodes
        assert DynamicA not in classifications['dynamic_nodes']
        assert DynamicB not in classifications['dynamic_nodes']

    def test_dynamic_nodes_are_descendants_of_dynamic_keys(self) -> None:
        workflow = sciline.Pipeline(
            (make_static_a, make_static_b, make_intermediate, make_accum_a, make_target)
        )
        workflow[DynamicA] = None
        workflow[AccumB] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            target_keys=(Target,),
            accumulators=(AccumA,),
        )

        graph = workflow.underlying_graph
        classifications = processor._classify_nodes(graph)

        # Intermediate depends on DynamicA, so it's a dynamic node
        assert Intermediate in classifications['dynamic_nodes']

    def test_accumulator_keys_are_classified(self) -> None:
        workflow = sciline.Pipeline(
            (make_static_a, make_static_b, make_intermediate, make_accum_a, make_target)
        )
        workflow[DynamicA] = None
        workflow[AccumB] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            target_keys=(Target,),
            accumulators=(AccumA,),
        )

        graph = workflow.underlying_graph
        classifications = processor._classify_nodes(graph)

        assert AccumA in classifications['accumulator_keys']
        # Accumulators should not be in dynamic_nodes
        assert AccumA not in classifications['dynamic_nodes']

    def test_finalize_nodes_are_downstream_of_accumulators(self) -> None:
        workflow = sciline.Pipeline(
            (make_static_a, make_static_b, make_intermediate, make_accum_a, make_target)
        )
        workflow[DynamicA] = None
        workflow[AccumB] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            target_keys=(Target,),
            accumulators=(AccumA,),
        )

        graph = workflow.underlying_graph
        classifications = processor._classify_nodes(graph)

        # Target depends on AccumA, so it's computed in finalize
        assert Target in classifications['finalize_nodes']
        # Target is also a target key
        assert Target in classifications['target_keys']

    def test_target_keys_are_classified(self) -> None:
        workflow = sciline.Pipeline(
            (make_static_a, make_static_b, make_intermediate, make_accum_a, make_target)
        )
        workflow[DynamicA] = None
        workflow[AccumB] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            target_keys=(Target,),
            accumulators=(AccumA,),
        )

        graph = workflow.underlying_graph
        classifications = processor._classify_nodes(graph)

        assert Target in classifications['target_keys']

    def test_context_keys_are_classified(self) -> None:
        workflow = sciline.Pipeline((make_context_dependent, make_intermediate))
        workflow[DynamicA] = None
        workflow[Context1] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            context_keys=(Context1,),
            target_keys=(Intermediate,),
            accumulators=(Intermediate,),
        )

        graph = workflow.underlying_graph
        classifications = processor._classify_nodes(graph)

        assert Context1 in classifications['context_keys']
        assert Context1 not in classifications['context_dependent']

    def test_context_dependent_nodes_are_downstream_of_context_not_dynamic(
        self,
    ) -> None:
        # Build a workflow where StaticB depends on Context1 but not on dynamic keys
        def from_context(ctx: Context1) -> StaticB:
            return StaticB(ctx)

        def from_static_and_dynamic(s: StaticB, d: DynamicA) -> AccumA:
            return AccumA(s + d)

        workflow = sciline.Pipeline((from_context, from_static_and_dynamic))
        workflow[DynamicA] = None
        workflow[Context1] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            context_keys=(Context1,),
            target_keys=(AccumA,),
            accumulators=(AccumA,),
        )

        graph = workflow.underlying_graph
        classifications = processor._classify_nodes(graph)

        # StaticB depends on Context1 but not on DynamicA
        assert StaticB in classifications['context_dependent']
        # AccumA depends on both context and dynamic, so it's dynamic
        assert AccumA not in classifications['context_dependent']

    def test_nodes_not_in_graph_are_excluded(self) -> None:
        workflow = sciline.Pipeline((make_static_a,))

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),  # Not in graph
            target_keys=(StaticA,),
            accumulators=(),
        )

        graph = workflow.underlying_graph
        classifications = processor._classify_nodes(graph)

        # DynamicA is not in the graph, so it shouldn't appear
        assert DynamicA not in classifications['dynamic_keys']

    def test_cached_keys_are_subset_of_static_at_dynamic_boundary(self) -> None:
        """Cached keys are the static nodes directly feeding into dynamic nodes."""
        workflow = sciline.Pipeline(
            (make_static_a, make_static_b, make_intermediate, make_accum_a, make_target)
        )
        workflow[DynamicA] = None
        workflow[AccumB] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            target_keys=(Target,),
            accumulators=(AccumA,),
        )

        graph = workflow.underlying_graph
        classifications = processor._classify_nodes(graph)

        # StaticB is cached because it's directly upstream of the dynamic node
        assert StaticB in classifications['cached_keys']
        # StaticA is NOT cached - it's a dependency of StaticB but not at the boundary
        assert StaticA not in classifications['cached_keys']
        # Both are still static
        assert StaticA in classifications['static']
        assert StaticB in classifications['static']


class TestFormatKeyForGraphviz:
    """Tests for the _format_key_for_graphviz helper."""

    def test_simple_newtype(self) -> None:
        result = streaming._format_key_for_graphviz(DynamicA)
        assert result == 'DynamicA'

    def test_generic_type(self) -> None:
        T = TypeVar('T')

        class Container(Generic[T]):
            pass

        result = streaming._format_key_for_graphviz(Container[int])
        assert result == 'Container[int]'

    def test_nested_generic(self) -> None:
        T = TypeVar('T')

        class Outer(Generic[T]):
            pass

        class Inner(Generic[T]):
            pass

        result = streaming._format_key_for_graphviz(Outer[Inner[int]])
        assert result == 'Outer[Inner[int]]'


class TestGetNodeStyle:
    """Tests for the _get_node_style helper."""

    def test_returns_empty_for_unclassified_node(self) -> None:
        classifications = {
            'static': set(),
            'cached_keys': set(),
            'dynamic_keys': set(),
            'dynamic_nodes': set(),
            'context_keys': set(),
            'context_dependent': set(),
            'accumulator_keys': set(),
            'target_keys': set(),
            'finalize_nodes': set(),
        }
        result = streaming._get_node_style('unknown', classifications)
        assert result == {}

    def test_returns_style_for_static_node(self) -> None:
        classifications = {
            'static': {StaticA},
            'cached_keys': set(),
            'dynamic_keys': set(),
            'dynamic_nodes': set(),
            'context_keys': set(),
            'context_dependent': set(),
            'accumulator_keys': set(),
            'target_keys': set(),
            'finalize_nodes': set(),
        }
        result = streaming._get_node_style(StaticA, classifications)
        assert result['fillcolor'] == '#e8e8e8'
        assert result['style'] == 'filled'

    def test_merges_styles_for_multiple_categories(self) -> None:
        # A node that is both an accumulator and a target
        classifications = {
            'static': set(),
            'cached_keys': set(),
            'dynamic_keys': set(),
            'dynamic_nodes': set(),
            'context_keys': set(),
            'context_dependent': set(),
            'accumulator_keys': {AccumA},
            'target_keys': {AccumA},
            'finalize_nodes': set(),
        }
        result = streaming._get_node_style(AccumA, classifications)
        # Should have accumulator style (orange, cylinder)
        assert result['fillcolor'] == '#FFB347'
        assert result['shape'] == 'cylinder'
        # Should also have target style (double border)
        assert result['peripheries'] == '2'

    def test_higher_priority_overrides_lower(self) -> None:
        # Dynamic key takes precedence over static for fill color
        classifications = {
            'static': {DynamicA},  # Lower priority
            'cached_keys': set(),
            'dynamic_keys': {DynamicA},  # Higher priority
            'dynamic_nodes': set(),
            'context_keys': set(),
            'context_dependent': set(),
            'accumulator_keys': set(),
            'target_keys': set(),
            'finalize_nodes': set(),
        }
        result = streaming._get_node_style(DynamicA, classifications)
        # Dynamic key color should win
        assert result['fillcolor'] == '#90EE90'

    def test_cached_keys_have_thick_border(self) -> None:
        classifications = {
            'static': {StaticB},
            'cached_keys': {StaticB},  # StaticB is cached
            'dynamic_keys': set(),
            'dynamic_nodes': set(),
            'context_keys': set(),
            'context_dependent': set(),
            'accumulator_keys': set(),
            'target_keys': set(),
            'finalize_nodes': set(),
        }
        result = streaming._get_node_style(StaticB, classifications)
        # Should have gray fill (from static/cached)
        assert result['fillcolor'] == '#e8e8e8'
        # Should have thick border (from cached_keys)
        assert result['penwidth'] == '2.5'


class TestVisualize:
    """Integration tests for the visualize method."""

    def test_returns_graphviz_digraph(self) -> None:
        workflow = sciline.Pipeline(
            (make_static_a, make_static_b, make_intermediate, make_accum_a, make_target)
        )
        workflow[DynamicA] = None
        workflow[AccumB] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            target_keys=(Target,),
            accumulators=(AccumA,),
        )

        result = processor.visualize()
        assert isinstance(result, graphviz.Digraph)

    def test_includes_legend_by_default(self) -> None:
        workflow = sciline.Pipeline((make_static_a,))

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(),
            target_keys=(StaticA,),
            accumulators=(),
        )

        result = processor.visualize()
        source = result.source
        assert 'cluster_legend' in source
        assert 'Legend' in source

    def test_legend_can_be_disabled(self) -> None:
        workflow = sciline.Pipeline((make_static_a,))

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(),
            target_keys=(StaticA,),
            accumulators=(),
        )

        result = processor.visualize(show_legend=False)
        source = result.source
        assert 'cluster_legend' not in source

    def test_dynamic_keys_have_thick_border(self) -> None:
        workflow = sciline.Pipeline((make_intermediate,))
        workflow[DynamicA] = None
        workflow[StaticB] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            target_keys=(Intermediate,),
            accumulators=(Intermediate,),
        )

        result = processor.visualize()
        source = result.source
        # Check that DynamicA node has penwidth styling
        assert 'DynamicA' in source
        assert 'penwidth=' in source

    def test_accumulator_keys_have_cylinder_shape(self) -> None:
        workflow = sciline.Pipeline((make_accum_a,))
        workflow[Intermediate] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(Intermediate,),
            target_keys=(AccumA,),
            accumulators=(AccumA,),
        )

        result = processor.visualize()
        source = result.source
        assert 'shape=cylinder' in source

    def test_passes_kwargs_to_sciline_visualize(self) -> None:
        workflow = sciline.Pipeline((make_static_a,))

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(),
            target_keys=(StaticA,),
            accumulators=(),
        )

        # Test that compact mode is passed through
        result = processor.visualize(compact=True)
        assert isinstance(result, graphviz.Digraph)

    def test_excludes_nodes_unreachable_from_targets(self) -> None:
        """Visualization should only show nodes that are ancestors of target_keys."""
        # Create a workflow with two independent branches
        # Branch 1: StaticA -> StaticB -> Intermediate -> AccumA -> Target
        # Branch 2: DynamicB -> AccumB (not used if we only target AccumA)
        workflow = sciline.Pipeline(
            (
                make_static_a,
                make_static_b,
                make_intermediate,
                make_accum_a,
                make_accum_b,  # This creates an independent branch via DynamicB
            )
        )
        workflow[DynamicA] = None
        workflow[DynamicB] = None

        # Only target AccumA - AccumB and DynamicB should NOT appear
        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA, DynamicB),
            target_keys=(AccumA,),
            accumulators=(AccumA,),
        )

        result = processor.visualize()
        source = result.source

        # Nodes in the path to AccumA should be present
        assert 'AccumA' in source
        assert 'Intermediate' in source
        assert 'DynamicA' in source
        assert 'StaticA' in source
        assert 'StaticB' in source

        # Nodes NOT in the path to AccumA should be excluded
        assert 'AccumB' not in source
        assert 'DynamicB' not in source

    def test_excludes_ancestors_of_input_keys(self) -> None:
        """Ancestors of dynamic/context keys should not appear in visualization."""
        # Create a workflow where dynamic key has an ancestor
        UpstreamOfDynamic = NewType('UpstreamOfDynamic', float)

        def make_dynamic_from_upstream(u: UpstreamOfDynamic) -> DynamicA:
            return DynamicA(u)

        workflow = sciline.Pipeline(
            (make_dynamic_from_upstream, make_intermediate, make_accum_a)
        )
        workflow[UpstreamOfDynamic] = 1.0
        workflow[StaticB] = StaticB(2.0)

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            target_keys=(AccumA,),
            accumulators=(AccumA,),
        )

        result = processor.visualize()
        source = result.source

        # DynamicA should appear (it's an input key)
        assert 'DynamicA' in source
        # But UpstreamOfDynamic should NOT appear (it's upstream of an input key)
        assert 'UpstreamOfDynamic' not in source

    def test_show_static_dependencies_false_hides_ancestors_of_cached(self) -> None:
        """Setting show_static_dependencies=False hides ancestors of cached nodes."""
        workflow = sciline.Pipeline(
            (make_static_a, make_static_b, make_intermediate, make_accum_a, make_target)
        )
        workflow[DynamicA] = None
        workflow[AccumB] = None

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(DynamicA,),
            target_keys=(Target,),
            accumulators=(AccumA,),
        )

        # With show_static_dependencies=True (default), StaticA should appear
        result_with = processor.visualize(show_static_dependencies=True)
        assert 'StaticA' in result_with.source

        # With show_static_dependencies=False, StaticA should be hidden
        result_without = processor.visualize(show_static_dependencies=False)
        assert 'StaticA' not in result_without.source
        # But StaticB (the cached node) should still appear
        assert 'StaticB' in result_without.source

    def test_show_static_dependencies_false_updates_legend(self) -> None:
        """Legend should not show 'Static' entry when show_static_dependencies=False."""
        workflow = sciline.Pipeline((make_static_a,))

        processor = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(),
            target_keys=(StaticA,),
            accumulators=(),
        )

        result = processor.visualize(show_static_dependencies=False)
        source = result.source

        # Should have 'Static (cached)' but not standalone 'Static'
        assert 'Static (cached)' in source
        # The legend_static node should not be present
        assert 'legend_static' not in source
