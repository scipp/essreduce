# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from typing import NewType

import pytest
import sciline
import scipp as sc

from ess.reduce import streaming


def test_eternal_accumulator_sums_everything() -> None:
    accum = streaming.EternalAccumulator()
    var = sc.linspace(dim='x', start=0, stop=1, num=10)
    for i in range(10):
        accum.push(var[i].copy())
    assert sc.identical(accum.value, sc.sum(var))


def test_eternal_accumulator_sums_everything_with_preprocess() -> None:
    accum = streaming.EternalAccumulator(preprocess=lambda x: x**0.5)
    var = sc.linspace(dim='x', start=0, stop=1, num=10)
    for i in range(10):
        accum.push(var[i].copy())
    assert sc.identical(accum.value, sc.sum(var**0.5))


def test_eternal_accumulator_works_if_output_value_is_modified() -> None:
    accum = streaming.EternalAccumulator()
    var = sc.linspace(dim='x', start=0, stop=1, num=10)
    for i in range(10):
        accum.push(var[i].copy())
    value = accum.value
    value += 1.0
    assert sc.identical(accum.value, sc.sum(var))


def test_eternal_accumulator_does_not_modify_pushed_values() -> None:
    accum = streaming.EternalAccumulator()
    var = sc.linspace(dim='x', start=0, stop=1, num=10)
    original = var.copy()
    for i in range(10):
        accum.push(var[i])
    assert sc.identical(var, original)


def test_eternal_accumulator_clear() -> None:
    accum = streaming.EternalAccumulator()
    var = sc.linspace(dim='x', start=0, stop=1, num=10)
    for i in range(10):
        accum.push(var[i].copy())
    assert sc.identical(accum.value, sc.sum(var))
    accum.clear()
    assert accum.is_empty
    with pytest.raises(ValueError, match="Cannot get value from empty accumulator"):
        _ = accum.value


def test_rolling_accumulator_sums_over_window() -> None:
    accum = streaming.RollingAccumulator(window=3)
    var = sc.linspace(dim='x', start=0, stop=1, num=10)
    accum.push(var[0].copy())
    assert sc.identical(accum.value, var[0])
    accum.push(var[1].copy())
    assert sc.identical(accum.value, var[0:2].sum())
    accum.push(var[2].copy())
    assert sc.identical(accum.value, var[0:3].sum())
    accum.push(var[3].copy())
    assert sc.identical(accum.value, var[1:4].sum())
    accum.push(var[4].copy())
    assert sc.identical(accum.value, var[2:5].sum())


def test_rolling_accumulator_sums_over_window_with_preprocess() -> None:
    accum = streaming.RollingAccumulator(window=3, preprocess=lambda x: x**0.5)
    var = sc.linspace(dim='x', start=0, stop=1, num=10)
    accum.push(var[0].copy())
    assert sc.identical(accum.value, var[0] ** 0.5)
    accum.push(var[1].copy())
    assert sc.identical(accum.value, (var[0:2] ** 0.5).sum())
    accum.push(var[2].copy())
    assert sc.identical(accum.value, (var[0:3] ** 0.5).sum())
    accum.push(var[3].copy())
    assert sc.identical(accum.value, (var[1:4] ** 0.5).sum())
    accum.push(var[4].copy())
    assert sc.identical(accum.value, (var[2:5] ** 0.5).sum())


def test_rolling_accumulator_works_if_output_value_is_modified() -> None:
    accum = streaming.RollingAccumulator(window=3)
    var = sc.linspace(dim='x', start=0, stop=1, num=10)
    for i in range(10):
        accum.push(var[i].copy())
    value = accum.value
    value += 1.0
    assert sc.identical(accum.value, var[7:10].sum())


def test_rolling_accumulator_does_not_modify_pushed_values() -> None:
    accum = streaming.RollingAccumulator(window=3)
    var = sc.linspace(dim='x', start=0, stop=1, num=10)
    original = var.copy()
    for i in range(10):
        accum.push(var[i])
    assert sc.identical(var, original)


def test_rolling_accumulator_clear() -> None:
    accum = streaming.RollingAccumulator(window=3)
    var = sc.linspace(dim='x', start=0, stop=1, num=10)
    for i in range(5):
        accum.push(var[i].copy())
    assert sc.identical(accum.value, var[2:5].sum())
    accum.clear()
    assert accum.is_empty
    with pytest.raises(ValueError, match="Cannot get value from empty accumulator"):
        _ = accum.value


def test_eternal_accumulator_is_empty() -> None:
    accum = streaming.EternalAccumulator()
    assert accum.is_empty
    with pytest.raises(ValueError, match="Cannot get value from empty accumulator"):
        _ = accum.value

    var = sc.linspace(dim='x', start=0, stop=1, num=10)
    accum.push(var[0].copy())
    assert not accum.is_empty
    assert sc.identical(accum.value, var[0])

    accum.clear()
    assert accum.is_empty
    with pytest.raises(ValueError, match="Cannot get value from empty accumulator"):
        _ = accum.value


def test_rolling_accumulator_is_empty() -> None:
    accum = streaming.RollingAccumulator(window=3)
    assert accum.is_empty
    with pytest.raises(ValueError, match="Cannot get value from empty accumulator"):
        _ = accum.value

    var = sc.linspace(dim='x', start=0, stop=1, num=10)
    accum.push(var[0].copy())
    assert not accum.is_empty
    assert sc.identical(accum.value, var[0])

    accum.clear()
    assert accum.is_empty
    with pytest.raises(ValueError, match="Cannot get value from empty accumulator"):
        _ = accum.value


def test_min_accumulator() -> None:
    accum = streaming.MinAccumulator()
    var = sc.array(dims=['x'], values=[1.0, 2.0, 3.0, 2.0, 1.0])
    for scalar_var in var:
        accum.push(scalar_var)
    assert sc.identical(accum.value, sc.min(var))
    accum.clear()
    assert accum.is_empty
    with pytest.raises(ValueError, match="Cannot get value from empty accumulator"):
        _ = accum.value


@pytest.mark.parametrize("Accum", [streaming.MinAccumulator, streaming.MaxAccumulator])
def test_accumulator_non_scalar_raises(Accum) -> None:
    accum = Accum()
    var = sc.array(dims=['x'], values=[1.0, 2.0, 3.0, 2.0, 1.0])
    accum.push(var)  # First push does not raise
    with pytest.raises(sc.DimensionError, match="Expected 0 dimensions"):
        accum.push(var)


def test_max_accumulator() -> None:
    accum = streaming.MaxAccumulator()
    var = sc.array(dims=['x'], values=[1.0, 2.0, 3.0, 2.0, 1.0])
    for scalar_var in var:
        accum.push(scalar_var)
    assert sc.identical(accum.value, sc.max(var))
    accum.clear()
    assert accum.is_empty
    with pytest.raises(ValueError, match="Cannot get value from empty accumulator"):
        _ = accum.value


DynamicA = NewType('DynamicA', float)
DynamicB = NewType('DynamicB', float)
DynamicC = NewType('DynamicC', float)
StaticA = NewType('StaticA', float)
AccumA = NewType('AccumA', float)
AccumB = NewType('AccumB', float)
AccumC = NewType('AccumC', float)
Target = NewType('Target', float)


def make_static_a() -> StaticA:
    make_static_a.call_count += 1
    return StaticA(2.0)


make_static_a.call_count = 0


def make_accum_a(value: DynamicA, static: StaticA) -> AccumA:
    return AccumA(value * static)


def make_accum_b(value: DynamicB) -> AccumB:
    return AccumB(value)


def make_accum_c(value: DynamicC) -> AccumC:
    return AccumC(value)


def make_target(accum_a: AccumA, accum_b: AccumB) -> Target:
    return Target(accum_a / accum_b)


def test_StreamProcessor_with_empty_workflow_and_targets() -> None:
    workflow = sciline.Pipeline()
    streaming_wf = streaming.StreamProcessor(
        base_workflow=workflow,
        dynamic_keys=(int,),
        target_keys=(),
        accumulators=(),
    )
    streaming_wf.accumulate({int: 1})
    result = streaming_wf.finalize()
    assert result == {}


def test_StreamProcessor_overall_behavior() -> None:
    base_workflow = sciline.Pipeline(
        (make_static_a, make_accum_a, make_accum_b, make_target)
    )
    orig_workflow = base_workflow.copy()

    streaming_wf = streaming.StreamProcessor(
        base_workflow=base_workflow,
        dynamic_keys=(DynamicA, DynamicB),
        target_keys=(Target,),
        accumulators=(AccumA, AccumB),
    )
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(1), DynamicB: sc.scalar(4)})
    assert sc.identical(result[Target], sc.scalar(2 * 1.0 / 4.0))
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(2), DynamicB: sc.scalar(5)})
    assert sc.identical(result[Target], sc.scalar(2 * 3.0 / 9.0))
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(3), DynamicB: sc.scalar(6)})
    assert sc.identical(result[Target], sc.scalar(2 * 6.0 / 15.0))
    assert make_static_a.call_count == 1

    # Consistency check: Run the original workflow with the same inputs, all at once
    orig_workflow[DynamicA] = sc.scalar(1 + 2 + 3)
    orig_workflow[DynamicB] = sc.scalar(4 + 5 + 6)
    expected = orig_workflow.compute(Target)
    assert sc.identical(expected, result[Target])


def test_StreamProcessor_uses_custom_accumulator() -> None:
    class Always42Accumulator(streaming.Accumulator[sc.Variable]):
        def _do_push(self, value: sc.Variable) -> None:
            pass

        @property
        def is_empty(self) -> bool:
            return False

        def _get_value(self) -> sc.Variable:
            return sc.scalar(42)

        def clear(self) -> None:
            # Nothing to clear
            pass

    base_workflow = sciline.Pipeline(
        (make_static_a, make_accum_a, make_accum_b, make_target)
    )

    streaming_wf = streaming.StreamProcessor(
        base_workflow=base_workflow,
        dynamic_keys=(DynamicA, DynamicB),
        target_keys=(Target,),
        accumulators={
            AccumA: streaming.EternalAccumulator(),
            AccumB: Always42Accumulator(),
        },
    )
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(1), DynamicB: sc.scalar(4)})
    assert sc.identical(result[Target], sc.scalar(2 * 1.0 / 42.0))
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(2), DynamicB: sc.scalar(5)})
    assert sc.identical(result[Target], sc.scalar(2 * 3.0 / 42.0))
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(3), DynamicB: sc.scalar(6)})
    assert sc.identical(result[Target], sc.scalar(2 * 6.0 / 42.0))


def test_StreamProcessor_does_not_compute_unused_static_nodes() -> None:
    def a_independent_target(accum_b: AccumB) -> Target:
        return Target(1.5 * accum_b)

    def derived_static_a(x: float) -> StaticA:
        derived_static_a.call_count += 1
        return StaticA(2.0 * x)

    derived_static_a.call_count = 0

    base_workflow = sciline.Pipeline(
        (derived_static_a, make_accum_a, make_accum_b, a_independent_target)
    )

    streaming_wf = streaming.StreamProcessor(
        base_workflow=base_workflow,
        dynamic_keys=(DynamicA, DynamicB),
        target_keys=(Target,),
        accumulators=(AccumA, AccumB),
    )
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(1), DynamicB: sc.scalar(4)})
    assert sc.identical(result[Target], sc.scalar(1.5 * 4.0))
    assert derived_static_a.call_count == 0


def test_StreamProcess_with_zero_accumulators_for_buffered_workflow_calls() -> None:
    base_workflow = sciline.Pipeline(
        (make_static_a, make_accum_a, make_accum_b, make_target)
    )
    make_static_a.call_count = 0

    streaming_wf = streaming.StreamProcessor(
        base_workflow=base_workflow,
        dynamic_keys=(DynamicA, DynamicB),
        target_keys=(Target,),
        accumulators=(),
        allow_bypass=True,
    )
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(1), DynamicB: sc.scalar(4)})
    assert sc.identical(result[Target], sc.scalar(2 * 1.0 / 4.0))
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(2), DynamicB: sc.scalar(5)})
    assert sc.identical(result[Target], sc.scalar(2 * 2.0 / 5.0))
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(3), DynamicB: sc.scalar(6)})
    assert sc.identical(result[Target], sc.scalar(2 * 3.0 / 6.0))
    assert make_static_a.call_count == 1


def test_StreamProcessor_with_bypass() -> None:
    def _make_static_a() -> StaticA:
        _make_static_a.call_count += 1
        return StaticA(2.0)

    _make_static_a.call_count = 0

    base_workflow = sciline.Pipeline(
        (_make_static_a, make_accum_a, make_accum_b, make_target)
    )
    orig_workflow = base_workflow.copy()

    streaming_wf = streaming.StreamProcessor(
        base_workflow=base_workflow,
        dynamic_keys=(DynamicA, DynamicB),
        target_keys=(Target,),
        accumulators=(AccumA,),  # Note: No AccumB
        allow_bypass=True,
    )
    streaming_wf.accumulate({DynamicA: sc.scalar(1), DynamicB: sc.scalar(4)})
    result = streaming_wf.finalize()
    assert sc.identical(result[Target], sc.scalar(2 * 1.0 / 4.0))
    streaming_wf.accumulate({DynamicA: sc.scalar(2), DynamicB: sc.scalar(5)})
    result = streaming_wf.finalize()
    # Note denominator is 5, not 9
    assert sc.identical(result[Target], sc.scalar(2 * 3.0 / 5.0))
    streaming_wf.accumulate({DynamicA: sc.scalar(3), DynamicB: sc.scalar(6)})
    result = streaming_wf.finalize()
    # Note denominator is 6, not 15
    assert sc.identical(result[Target], sc.scalar(2 * 6.0 / 6.0))
    assert _make_static_a.call_count == 1

    # Consistency check: Run the original workflow with the same inputs, all at once
    orig_workflow[DynamicA] = sc.scalar(1 + 2 + 3)
    orig_workflow[DynamicB] = sc.scalar(6)
    expected = orig_workflow.compute(Target)
    assert sc.identical(expected, result[Target])


def test_StreamProcessor_without_bypass_raises() -> None:
    def _make_static_a() -> StaticA:
        _make_static_a.call_count += 1
        return StaticA(2.0)

    _make_static_a.call_count = 0

    base_workflow = sciline.Pipeline(
        (_make_static_a, make_accum_a, make_accum_b, make_target)
    )

    streaming_wf = streaming.StreamProcessor(
        base_workflow=base_workflow,
        dynamic_keys=(DynamicA, DynamicB),
        target_keys=(Target,),
        accumulators=(AccumA,),  # Note: No AccumB
    )
    streaming_wf.accumulate({DynamicA: 1, DynamicB: 4})
    # Sciline passes `None` to the provider that needs AccumB.
    with pytest.raises(TypeError, match='unsupported operand type'):
        _ = streaming_wf.finalize()


def test_StreamProcessor_calls_providers_after_accumulators_only_when_finalizing() -> (
    None
):
    def _make_target(accum_a: AccumA, accum_b: AccumB) -> Target:
        _make_target.call_count += 1
        return Target(accum_a / accum_b)

    _make_target.call_count = 0

    base_workflow = sciline.Pipeline(
        (make_accum_a, make_accum_b, _make_target), params={StaticA: 2.0}
    )

    streaming_wf = streaming.StreamProcessor(
        base_workflow=base_workflow,
        dynamic_keys=(DynamicA, DynamicB),
        target_keys=(Target,),
        accumulators=(AccumA, AccumB),
    )
    streaming_wf.accumulate({DynamicA: sc.scalar(1), DynamicB: sc.scalar(4)})
    streaming_wf.accumulate({DynamicA: sc.scalar(2), DynamicB: sc.scalar(5)})
    assert _make_target.call_count == 0
    result = streaming_wf.finalize()
    assert sc.identical(result[Target], sc.scalar(2 * 3.0 / 9.0))
    assert _make_target.call_count == 1
    streaming_wf.accumulate({DynamicA: sc.scalar(3), DynamicB: sc.scalar(6)})
    assert _make_target.call_count == 1
    result = streaming_wf.finalize()
    assert sc.identical(result[Target], sc.scalar(2 * 6.0 / 15.0))
    assert _make_target.call_count == 2
    result = streaming_wf.finalize()
    assert sc.identical(result[Target], sc.scalar(2 * 6.0 / 15.0))
    # Outputs are not cached.
    assert _make_target.call_count == 3


def test_StreamProcessor_does_not_reuse_dynamic_keys() -> None:
    base_workflow = sciline.Pipeline(
        (make_accum_a, make_accum_b, make_target), params={StaticA: 2.0}
    )
    orig_workflow = base_workflow.copy()

    streaming_wf = streaming.StreamProcessor(
        base_workflow=base_workflow,
        dynamic_keys=(DynamicA, DynamicB),
        target_keys=(Target,),
        accumulators=(AccumA, AccumB),
    )
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(1), DynamicB: sc.scalar(4)})
    assert sc.identical(result[Target], sc.scalar(2 * 1.0 / 4.0))
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(2)})  # Only A
    assert not sc.identical(result[Target], sc.scalar(2 * 3.0 / 8.0))
    assert sc.identical(result[Target], sc.scalar(2 * 3.0 / 4.0))
    result = streaming_wf.add_chunk({DynamicB: sc.scalar(5)})  # Only B
    assert sc.identical(result[Target], sc.scalar(2 * 3.0 / 9.0))
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(3), DynamicB: sc.scalar(6)})
    assert sc.identical(result[Target], sc.scalar(2 * 6.0 / 15.0))

    # Consistency check: Run the original workflow with the same inputs, all at once
    orig_workflow[DynamicA] = sc.scalar(1 + 2 + 3)
    orig_workflow[DynamicB] = sc.scalar(4 + 5 + 6)
    expected = orig_workflow.compute(Target)
    assert sc.identical(expected, result[Target])


def test_StreamProcessor_raises_given_partial_update_for_accumulator() -> None:
    base_workflow = sciline.Pipeline(
        (make_accum_a, make_accum_b, make_accum_c, make_target), params={StaticA: 2.0}
    )
    streaming_wf = streaming.StreamProcessor(
        base_workflow=base_workflow,
        dynamic_keys=(DynamicA, DynamicB, DynamicC),
        target_keys=(Target, AccumC),
        accumulators=(Target, AccumC),  # Target depends on both A and B
    )
    # We can update either (A, B) and/or C...
    streaming_wf.accumulate({DynamicA: sc.scalar(1), DynamicB: sc.scalar(4)})
    streaming_wf.accumulate({DynamicC: sc.scalar(11)})
    result = streaming_wf.finalize()
    assert sc.identical(result[Target], sc.scalar(2 * 1.0 / 4.0))
    assert sc.identical(result[AccumC], sc.scalar(11))
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(2), DynamicB: sc.scalar(5)})
    assert sc.identical(result[Target], sc.scalar(2 * (1.0 / 4.0 + 2.0 / 5.0)))
    assert sc.identical(result[AccumC], sc.scalar(11))
    result = streaming_wf.add_chunk({DynamicC: sc.scalar(12)})
    assert sc.identical(result[Target], sc.scalar(2 * (1.0 / 4.0 + 2.0 / 5.0)))
    assert sc.identical(result[AccumC], sc.scalar(23))
    # ... but not just A or B
    with pytest.raises(
        ValueError,
        match=r'{tests.streaming_test.DynamicB} not provided in the current chunk',
    ):
        result = streaming_wf.add_chunk({DynamicA: sc.scalar(2)})  # Only A
    with pytest.raises(
        ValueError,
        match=r'{tests.streaming_test.DynamicA} not provided in the current chunk',
    ):
        result = streaming_wf.add_chunk({DynamicB: sc.scalar(5)})  # Only B


def test_StreamProcessor_raises_when_trying_to_update_non_dynamic_key() -> None:
    base_workflow = sciline.Pipeline(
        (make_static_a, make_accum_a, make_accum_b, make_target)
    )
    streaming_wf = streaming.StreamProcessor(
        base_workflow=base_workflow,
        dynamic_keys=(DynamicA, DynamicB),
        target_keys=(Target,),
        accumulators=(AccumA, AccumB),
    )

    # Regular update ok
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(1), DynamicB: sc.scalar(4)})
    assert sc.identical(result[Target], sc.scalar(2 * 1.0 / 4.0))

    # Non-dynamic input key
    with pytest.raises(
        ValueError,
        match=r'Got non-dynamic keys: {tests.streaming_test.StaticA}',
    ):
        result = streaming_wf.add_chunk({StaticA: sc.scalar(2)})
    # Intermediate key depending on dynamic key
    with pytest.raises(
        ValueError,
        match=r'Got non-dynamic keys: {tests.streaming_test.AccumA}',
    ):
        result = streaming_wf.add_chunk({AccumA: sc.scalar(2)})
    # Target key depending on dynamic key
    with pytest.raises(
        ValueError,
        match=r'Got non-dynamic keys: {tests.streaming_test.Target}',
    ):
        result = streaming_wf.add_chunk({Target: sc.scalar(2)})


def test_StreamProcessor_clear() -> None:
    base_workflow = sciline.Pipeline(
        (make_static_a, make_accum_a, make_accum_b, make_target)
    )

    # Reset call counter to ensure we can track it properly
    make_static_a.call_count = 0

    streaming_wf = streaming.StreamProcessor(
        base_workflow=base_workflow,
        dynamic_keys=(DynamicA, DynamicB),
        target_keys=(Target,),
        accumulators=(AccumA, AccumB),
    )
    # Add some data
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(1), DynamicB: sc.scalar(4)})
    assert sc.identical(result[Target], sc.scalar(2 * 1.0 / 4.0))
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(2), DynamicB: sc.scalar(5)})
    assert sc.identical(result[Target], sc.scalar(2 * 3.0 / 9.0))

    # Make sure static_a was called exactly once
    assert make_static_a.call_count == 1

    # Clear and verify we get back to initial state
    streaming_wf.clear()
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(1), DynamicB: sc.scalar(4)})
    assert sc.identical(result[Target], sc.scalar(2 * 1.0 / 4.0))

    # Static values should be preserved after clear, so call_count remains 1
    assert make_static_a.call_count == 1


def test_StreamProcessor_with_context() -> None:
    Streamed = NewType('Streamed', int)
    Context = NewType('Context', int)
    Static = NewType('Static', int)
    ProcessedContext = NewType('ProcessedContext', int)
    ProcessedStreamed = NewType('ProcessedStreamed', int)
    Output = NewType('Output', int)

    def make_static() -> Static:
        make_static.call_count += 1
        return Static(2)

    make_static.call_count = 0

    def process_context(context: Context, static: Static) -> ProcessedContext:
        process_context.call_count += 1
        return ProcessedContext(context * static)

    process_context.call_count = 0

    def process_streamed(
        streamed: Streamed, context: ProcessedContext
    ) -> ProcessedStreamed:
        return ProcessedStreamed(streamed + context)

    def finalize(streamed: ProcessedStreamed) -> Output:
        return Output(streamed)

    wf = sciline.Pipeline((make_static, process_context, process_streamed, finalize))
    streaming_wf = streaming.StreamProcessor(
        base_workflow=wf,
        dynamic_keys=(Streamed,),
        context_keys=(Context,),
        target_keys=(Output,),
        accumulators=(ProcessedStreamed,),
    )
    assert make_static.call_count == 1
    assert process_context.call_count == 0
    streaming_wf.set_context({Context: sc.scalar(3)})
    assert process_context.call_count == 1
    streaming_wf.accumulate({Streamed: sc.scalar(4)})
    assert sc.identical(streaming_wf.finalize()[Output], sc.scalar(2 * 3 + 4))
    assert make_static.call_count == 1
    assert process_context.call_count == 1
    streaming_wf.accumulate({Streamed: sc.scalar(4)})
    assert process_context.call_count == 1
    streaming_wf.set_context({Context: sc.scalar(1)})
    # Context changed but no chunk added, so new context is not visible in the result
    assert sc.identical(streaming_wf.finalize()[Output], sc.scalar(2 * (2 * 3 + 4)))
    # The third chunk sees the context value of "1" (instead of "3")
    streaming_wf.accumulate({Streamed: sc.scalar(4)})
    assert sc.identical(
        streaming_wf.finalize()[Output], sc.scalar(2 * (2 * 3 + 4) + 2 * 1 + 4)
    )
    assert make_static.call_count == 1
    assert process_context.call_count == 2


def test_StreamProcessor_with_context_updates_non_dynamic_targets() -> None:
    Streamed = NewType('Streamed', int)
    Context = NewType('Context', int)
    Static = NewType('Static', int)
    ProcessedContext = NewType('ProcessedContext', int)
    ProcessedStreamed = NewType('ProcessedStreamed', int)
    Output = NewType('Output', int)

    def make_static() -> Static:
        make_static.call_count += 1
        return Static(2)

    make_static.call_count = 0

    def process_context(context: Context, static: Static) -> ProcessedContext:
        process_context.call_count += 1
        return ProcessedContext(context * static)

    process_context.call_count = 0

    def process_streamed(streamed: Streamed) -> ProcessedStreamed:
        return ProcessedStreamed(streamed)

    def finalize(streamed: ProcessedStreamed) -> Output:
        return Output(streamed)

    wf = sciline.Pipeline((make_static, process_context, process_streamed, finalize))
    streaming_wf = streaming.StreamProcessor(
        base_workflow=wf,
        dynamic_keys=(Streamed,),
        context_keys=(Context,),
        target_keys=(ProcessedContext, Output),
        accumulators=(ProcessedStreamed,),
    )
    assert make_static.call_count == 1
    assert process_context.call_count == 0

    streaming_wf.set_context({Context: sc.scalar(3)})
    assert process_context.call_count == 1
    streaming_wf.accumulate({Streamed: sc.scalar(4)})

    results = streaming_wf.finalize()
    assert sc.identical(results[Output], sc.scalar(4))
    assert sc.identical(results[ProcessedContext], sc.scalar(2 * 3))
    assert make_static.call_count == 1
    assert process_context.call_count == 1

    streaming_wf.accumulate({Streamed: sc.scalar(4)})
    assert process_context.call_count == 1

    results = streaming_wf.finalize()
    assert process_context.call_count == 1

    streaming_wf.set_context({Context: sc.scalar(5)})
    results = streaming_wf.finalize()
    assert sc.identical(results[Output], sc.scalar(8))
    assert sc.identical(results[ProcessedContext], sc.scalar(2 * 5))
    assert process_context.call_count == 2


def test_StreamProcessor_raises_when_invalid_context_key_provided() -> None:
    Streamed = NewType('Streamed', int)
    Context = NewType('Context', int)
    NonContext = NewType('NonContext', int)
    Output = NewType('Output', int)

    def process(streamed: Streamed, context: Context) -> Output:
        return Output(streamed + context)

    wf = sciline.Pipeline((process,))
    streaming_wf = streaming.StreamProcessor(
        base_workflow=wf,
        dynamic_keys=(Streamed,),
        context_keys=(Context,),
        target_keys=(Output,),
        accumulators=(Output,),
    )

    streaming_wf.set_context({Context: sc.scalar(3)})
    # NonContext is not in context_keys
    with pytest.raises(ValueError, match=r"Key .* is not a context key"):
        streaming_wf.set_context({NonContext: sc.scalar(5)})


def test_StreamProcessor_context_only_recomputes_descendants() -> None:
    Streamed = NewType('Streamed', int)
    ContextInputA = NewType('ContextInputA', int)
    ContextInputB = NewType('ContextInputB', int)
    ProcessedA = NewType('ProcessedA', int)
    ProcessedB = NewType('ProcessedB', int)
    Output = NewType('Output', int)

    def process_a(context: ContextInputA) -> ProcessedA:
        process_a.call_count += 1
        return ProcessedA(context * 2)

    process_a.call_count = 0

    def process_b(context: ContextInputB) -> ProcessedB:
        process_b.call_count += 1
        return ProcessedB(context * 3)

    process_b.call_count = 0

    def combine(streamed: Streamed, a: ProcessedA, b: ProcessedB) -> Output:
        return Output(streamed + a + b)

    wf = sciline.Pipeline((process_a, process_b, combine))
    streaming_wf = streaming.StreamProcessor(
        base_workflow=wf,
        dynamic_keys=(Streamed,),
        context_keys=(ContextInputA, ContextInputB),
        target_keys=(Output,),
        accumulators=(Output,),
    )

    assert process_a.call_count == 0
    assert process_b.call_count == 0

    # Set both contexts initially
    streaming_wf.set_context({ContextInputA: sc.scalar(1), ContextInputB: sc.scalar(2)})
    assert process_a.call_count == 1
    assert process_b.call_count == 1

    # Update only ContextInputA
    streaming_wf.set_context({ContextInputA: sc.scalar(3)})
    assert process_a.call_count == 2
    assert process_b.call_count == 1  # Not called again

    # Update only ContextInputB
    streaming_wf.set_context({ContextInputB: sc.scalar(4)})
    assert process_a.call_count == 2  # Not called again
    assert process_b.call_count == 2


def test_StreamProcessor_context_retained_after_clear() -> None:
    Streamed = NewType('Streamed', int)
    Context = NewType('Context', int)
    ProcessedContext = NewType('ProcessedContext', int)
    Output = NewType('Output', int)

    def process_context(context: Context) -> ProcessedContext:
        process_context.call_count += 1
        return ProcessedContext(context * 2)

    process_context.call_count = 0

    def combine(streamed: Streamed, context: ProcessedContext) -> Output:
        return Output(streamed + context)

    wf = sciline.Pipeline((process_context, combine))
    streaming_wf = streaming.StreamProcessor(
        base_workflow=wf,
        dynamic_keys=(Streamed,),
        context_keys=(Context,),
        target_keys=(Output,),
        accumulators=(Output,),
    )

    streaming_wf.set_context({Context: sc.scalar(5)})
    assert process_context.call_count == 1

    streaming_wf.accumulate({Streamed: sc.scalar(3)})
    result = streaming_wf.finalize()
    assert sc.identical(result[Output], sc.scalar(3 + 5 * 2))

    streaming_wf.clear()
    # Context should be preserved after clear
    streaming_wf.accumulate({Streamed: sc.scalar(3)})
    result = streaming_wf.finalize()
    assert sc.identical(result[Output], sc.scalar(3 + 5 * 2))
    # process_context shouldn't have been called again
    assert process_context.call_count == 1


def test_StreamProcessor_empty_context_update() -> None:
    Streamed = NewType('Streamed', int)
    Context = NewType('Context', int)
    Output = NewType('Output', int)

    def process(streamed: Streamed, context: Context) -> Output:
        return Output(streamed + context)

    wf = sciline.Pipeline((process,))
    streaming_wf = streaming.StreamProcessor(
        base_workflow=wf,
        dynamic_keys=(Streamed,),
        context_keys=(Context,),
        target_keys=(Output,),
        accumulators=(Output,),
    )

    # Empty context update should not raise
    streaming_wf.set_context({})  # No-op

    streaming_wf.set_context({Context: sc.scalar(3)})
    streaming_wf.accumulate({Streamed: sc.scalar(4)})
    result = streaming_wf.finalize()
    assert sc.identical(result[Output], sc.scalar(7))


def test_StreamProcessor_with_overlapping_context_and_dynamic_keys_raises() -> None:
    Key = NewType('Key', int)  # Used as both context and dynamic
    Output = NewType('Output', int)

    def double(value: Key) -> Output:
        return Output(value * 2)

    wf = sciline.Pipeline((double,))

    # Using a key as both context and dynamic should raise
    with pytest.raises(ValueError, match="Keys cannot be both dynamic and context"):
        streaming.StreamProcessor(
            base_workflow=wf,
            dynamic_keys=(Key,),
            context_keys=(Key,),
            target_keys=(Output,),
            accumulators=(Output,),
        )


def test_StreamProcessor_with_unused_context_key() -> None:
    Streamed = NewType('Streamed', int)
    UsedContext = NewType('UsedContext', int)
    UnusedContext = NewType('UnusedContext', int)
    Output = NewType('Output', int)

    def process(streamed: Streamed, context: UsedContext) -> Output:
        return Output(streamed + context)

    wf = sciline.Pipeline((process,))
    streaming_wf = streaming.StreamProcessor(
        base_workflow=wf,
        dynamic_keys=(Streamed,),
        context_keys=(UsedContext, UnusedContext),  # UnusedContext is never used
        target_keys=(Output,),
        accumulators=(Output,),
    )

    # Setting only the used context should work fine
    streaming_wf.set_context({UsedContext: sc.scalar(3)})
    streaming_wf.accumulate({Streamed: sc.scalar(4)})
    result = streaming_wf.finalize()
    assert sc.identical(result[Output], sc.scalar(7))

    # Setting the unused context should also work without errors
    streaming_wf.set_context({UnusedContext: sc.scalar(99)})
    streaming_wf.accumulate({Streamed: sc.scalar(5)})
    result = streaming_wf.finalize()
    assert sc.identical(result[Output], sc.scalar(7 + 8))


def test_StreamProcessor_with_separate_context_nodes() -> None:
    """Test that non-interdependent context keys work fine."""
    Streamed = NewType('Streamed', int)
    ContextInputA = NewType('ContextInputA', int)
    ContextInputB = NewType('ContextInputB', int)
    Output = NewType('Output', int)

    def process(streamed: Streamed, a: ContextInputA, b: ContextInputB) -> Output:
        return Output(streamed + a + b)

    wf = sciline.Pipeline((process,))
    streaming_wf = streaming.StreamProcessor(
        base_workflow=wf,
        dynamic_keys=(Streamed,),
        context_keys=(ContextInputA, ContextInputB),  # Independent contexts
        target_keys=(Output,),
        accumulators=(Output,),
    )

    # Setting both contexts should work fine
    streaming_wf.set_context({ContextInputA: sc.scalar(3), ContextInputB: sc.scalar(5)})
    streaming_wf.accumulate({Streamed: sc.scalar(4)})
    result = streaming_wf.finalize()
    assert sc.identical(result[Output], sc.scalar(4 + 3 + 5))

    # Updating only one context should also work
    streaming_wf.set_context({ContextInputA: sc.scalar(7)})
    streaming_wf.accumulate({Streamed: sc.scalar(2)})
    result = streaming_wf.finalize()
    assert sc.identical(result[Output], sc.scalar(4 + 3 + 5 + 2 + 7 + 5))


def test_StreamProcessor_with_non_scalar_context() -> None:
    Streamed = NewType('Streamed', float)
    VectorContext = NewType('VectorContext', sc.Variable)
    Output = NewType('Output', float)

    def process(streamed: Streamed, context: VectorContext) -> Output:
        # Use the sum of the context vector
        return Output(streamed + sc.sum(context).value)

    wf = sciline.Pipeline((process,))
    streaming_wf = streaming.StreamProcessor(
        base_workflow=wf,
        dynamic_keys=(Streamed,),
        context_keys=(VectorContext,),
        target_keys=(Output,),
        accumulators=(Output,),
    )

    # Set a vector context
    vector_data = sc.array(dims=['x'], values=[1.0, 2.0, 3.0])
    streaming_wf.set_context({VectorContext: vector_data})
    streaming_wf.accumulate({Streamed: sc.scalar(4.0)})
    result = streaming_wf.finalize()
    assert sc.identical(result[Output], sc.scalar(4.0 + 6.0))  # 6.0 is sum of vector

    # Update vector context
    new_vector = sc.array(dims=['x'], values=[2.0, 3.0, 4.0])
    streaming_wf.set_context({VectorContext: new_vector})
    streaming_wf.accumulate({Streamed: sc.scalar(5.0)})
    result = streaming_wf.finalize()
    assert sc.identical(result[Output], sc.scalar(4.0 + 6.0 + 5.0 + 9.0))


def test_StreamProcessor_with_invalid_context_value() -> None:
    Streamed = NewType('Streamed', int)
    Context = NewType('Context', int)
    Output = NewType('Output', int)

    def process(streamed: Streamed, context: Context) -> Output:
        # This will fail if context is not an integer
        return Output(streamed + context)

    wf = sciline.Pipeline((process,))
    streaming_wf = streaming.StreamProcessor(
        base_workflow=wf,
        dynamic_keys=(Streamed,),
        context_keys=(Context,),
        target_keys=(Output,),
        accumulators=(Output,),
    )

    # Setting an invalid context type should trigger error when using it.
    streaming_wf.set_context({Context: "not_an_integer"})
    with pytest.raises(TypeError):
        streaming_wf.accumulate({Streamed: sc.scalar(4)})


def test_StreamProcessor_rejects_interdependent_dynamic_keys() -> None:
    Streamed1 = NewType('Streamed1', int)
    Streamed2 = NewType('Streamed2', int)
    Output = NewType('Output', int)

    def process_derived_dynamic(primary: Streamed1) -> Streamed2:
        return Streamed2(primary * 10)

    def process(value1: Streamed1, value2: Streamed2) -> Output:
        return Output(value1 + value2)

    wf = sciline.Pipeline((process_derived_dynamic, process))

    # Using interdependent dynamic keys should raise a clear error
    with pytest.raises(
        ValueError,
        match=r"Dynamic key .* depends on other dynamic/context keys: .*. "
        "This is not supported.",
    ):
        streaming.StreamProcessor(
            base_workflow=wf,
            dynamic_keys=(Streamed1, Streamed2),  # Streamed2 depends on Streamed1
            target_keys=(Output,),
            accumulators=(Output,),
        )


def test_StreamProcessor_rejects_interdependent_context_keys() -> None:
    Streamed = NewType('Streamed', int)
    PrimaryContext = NewType('PrimaryContext', int)
    DerivedContext = NewType('DerivedContext', int)
    Output = NewType('Output', int)

    def process_derived_context(primary: PrimaryContext) -> DerivedContext:
        return DerivedContext(primary * 10)

    def process(streamed: Streamed, derived: DerivedContext) -> Output:
        return Output(streamed + derived)

    wf = sciline.Pipeline((process_derived_context, process))

    # Context key depending on another context key should raise
    with pytest.raises(
        ValueError,
        match=r"Context key .* depends on other dynamic/context keys: .*. "
        "This is not supported.",
    ):
        streaming.StreamProcessor(
            base_workflow=wf,
            dynamic_keys=(Streamed,),
            context_keys=(
                PrimaryContext,
                DerivedContext,
            ),  # DerivedContext depends on PrimaryContext
            target_keys=(Output,),
            accumulators=(Output,),
        )


def test_StreamProcessor_rejects_dynamic_keys_depending_on_context_keys() -> None:
    Context = NewType('Context', int)
    Streamed = NewType('Streamed', int)
    Output = NewType('Output', int)

    def derive_dynamic(context: Context) -> Streamed:
        return Streamed(context * 2)

    def process(value: Streamed) -> Output:
        return Output(value * 10)

    wf = sciline.Pipeline((derive_dynamic, process))

    # Dynamic key depending on context key should raise
    with pytest.raises(
        ValueError,
        match=r"Dynamic key .* depends on other dynamic/context keys: .*. "
        "This is not supported.",
    ):
        streaming.StreamProcessor(
            base_workflow=wf,
            dynamic_keys=(Streamed,),
            context_keys=(Context,),
            target_keys=(Output,),
            accumulators=(Output,),
        )


def test_StreamProcessor_rejects_context_keys_depending_on_dynamic_keys() -> None:
    Streamed = NewType('Streamed', int)
    Context = NewType('Context', int)
    Output = NewType('Output', int)

    def derive_context(value: Streamed) -> Context:
        return Context(value * 2)

    def process(value: Streamed, context: Context) -> Output:
        return Output(value + context)

    wf = sciline.Pipeline((derive_context, process))

    # Context key depending on dynamic key should raise
    with pytest.raises(
        ValueError,
        match=r"Context key .* depends on other dynamic/context keys: .*. "
        "This is not supported.",
    ):
        streaming.StreamProcessor(
            base_workflow=wf,
            dynamic_keys=(Streamed,),
            context_keys=(Context,),
            target_keys=(Output,),
            accumulators=(Output,),
        )
