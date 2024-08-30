# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from typing import NewType

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


DynamicA = NewType('DynamicA', float)
DynamicB = NewType('DynamicB', float)
StaticA = NewType('StaticA', float)
AccumA = NewType('AccumA', float)
AccumB = NewType('AccumB', float)
Target = NewType('Target', float)


def make_static_a() -> StaticA:
    make_static_a.call_count += 1
    return StaticA(2.0)


make_static_a.call_count = 0


def make_accum_a(value: DynamicA, static: StaticA) -> AccumA:
    return AccumA(value * static)


def make_accum_b(value: DynamicB) -> AccumB:
    return AccumB(value)


def make_target(accum_a: AccumA, accum_b: AccumB) -> Target:
    return Target(accum_a / accum_b)


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
        def value(self) -> sc.Variable:
            return sc.scalar(42)

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
    )
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(1), DynamicB: sc.scalar(4)})
    assert sc.identical(result[Target], sc.scalar(2 * 1.0 / 4.0))
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(2), DynamicB: sc.scalar(5)})
    assert sc.identical(result[Target], sc.scalar(2 * 2.0 / 5.0))
    result = streaming_wf.add_chunk({DynamicA: sc.scalar(3), DynamicB: sc.scalar(6)})
    assert sc.identical(result[Target], sc.scalar(2 * 3.0 / 6.0))
    assert make_static_a.call_count == 1
