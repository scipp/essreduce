# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import scipp as sc

from ess.reduce import streaming


def test_eternal_accumulator_sums_everything():
    accum = streaming.EternalAccumulator()
    var = sc.linspace(dim='x', start=0, stop=1, num=10)
    for i in range(10):
        accum.push(var[i].copy())
    assert sc.identical(accum.value, sc.sum(var))


def test_eternal_accumulator_sums_everything_with_preprocess():
    accum = streaming.EternalAccumulator(preprocess=lambda x: x**0.5)
    var = sc.linspace(dim='x', start=0, stop=1, num=10)
    for i in range(10):
        accum.push(var[i].copy())
    assert sc.identical(accum.value, sc.sum(var**0.5))


def test_eternal_accumulator_works_if_output_value_is_modified():
    accum = streaming.EternalAccumulator()
    var = sc.linspace(dim='x', start=0, stop=1, num=10)
    for i in range(10):
        accum.push(var[i].copy())
    value = accum.value
    value += 1.0
    assert sc.identical(accum.value, sc.sum(var))


def test_eternal_accumulator_does_not_modify_pushed_values():
    accum = streaming.EternalAccumulator()
    var = sc.linspace(dim='x', start=0, stop=1, num=10)
    original = var.copy()
    for i in range(10):
        accum.push(var[i])
    assert sc.identical(var, original)


def test_rolling_accumulator_sums_over_window():
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


def test_rolling_accumulator_sums_over_window_with_preprocess():
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


def test_rolling_accumulator_works_if_output_value_is_modified():
    accum = streaming.RollingAccumulator(window=3)
    var = sc.linspace(dim='x', start=0, stop=1, num=10)
    for i in range(10):
        accum.push(var[i].copy())
    value = accum.value
    value += 1.0
    assert sc.identical(accum.value, var[7:10].sum())


def test_rolling_accumulator_does_not_modify_pushed_values():
    accum = streaming.RollingAccumulator(window=3)
    var = sc.linspace(dim='x', start=0, stop=1, num=10)
    original = var.copy()
    for i in range(10):
        accum.push(var[i])
    assert sc.identical(var, original)
