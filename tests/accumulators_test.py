# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pytest
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


def test_mean_accumulator_calculates_mean() -> None:
    accum = streaming.MeanAccumulator()
    var = sc.linspace(dim='x', start=0, stop=10, num=5)
    for i in range(5):
        accum.push(var[i].copy())
    assert sc.identical(accum.value, sc.mean(var))


def test_mean_accumulator_with_preprocess() -> None:
    accum = streaming.MeanAccumulator(preprocess=lambda x: x**2)
    var = sc.linspace(dim='x', start=0, stop=10, num=5)
    for i in range(5):
        accum.push(var[i].copy())
    assert sc.identical(accum.value, sc.mean(var**2))


def test_mean_accumulator_works_if_output_value_is_modified() -> None:
    accum = streaming.MeanAccumulator()
    var = sc.linspace(dim='x', start=0, stop=10, num=5)
    for i in range(5):
        accum.push(var[i].copy())
    value = accum.value
    value += 1.0
    assert sc.identical(accum.value, sc.mean(var))


def test_mean_accumulator_does_not_modify_pushed_values() -> None:
    accum = streaming.MeanAccumulator()
    var = sc.linspace(dim='x', start=0, stop=10, num=5)
    original = var.copy()
    for i in range(5):
        accum.push(var[i])
    assert sc.identical(var, original)


def test_mean_accumulator_clear() -> None:
    accum = streaming.MeanAccumulator()
    var = sc.linspace(dim='x', start=0, stop=10, num=5)
    for i in range(5):
        accum.push(var[i].copy())
    assert sc.identical(accum.value, sc.mean(var))
    accum.clear()
    assert accum.is_empty
    with pytest.raises(ValueError, match="Cannot get value from empty accumulator"):
        _ = accum.value


def test_mean_accumulator_is_empty() -> None:
    accum = streaming.MeanAccumulator()
    assert accum.is_empty
    with pytest.raises(ValueError, match="Cannot get value from empty accumulator"):
        _ = accum.value

    var = sc.linspace(dim='x', start=0, stop=10, num=5)
    accum.push(var[0].copy())
    assert not accum.is_empty
    assert sc.identical(accum.value, var[0])


def test_mean_accumulator_incremental_mean_calculation() -> None:
    accum = streaming.MeanAccumulator()
    var = sc.array(dims=['x'], values=[2.0, 4.0, 6.0, 8.0])

    # Push values one by one and verify mean is calculated correctly each time
    accum.push(var[0])
    assert sc.identical(accum.value, var[0])

    accum.push(var[1])
    assert sc.identical(accum.value, sc.mean(var[0:2]))

    accum.push(var[2])
    assert sc.identical(accum.value, sc.mean(var[0:3]))

    accum.push(var[3])
    assert sc.identical(accum.value, sc.mean(var))


def test_mean_accumulator_after_clear() -> None:
    accum = streaming.MeanAccumulator()
    var = sc.array(dims=['x'], values=[2.0, 4.0, 6.0])

    # First round of accumulation
    for i in range(3):
        accum.push(var[i].copy())
    assert sc.identical(accum.value, sc.mean(var))

    # Clear and verify empty state
    accum.clear()
    assert accum.is_empty

    # Second round with new values
    new_var = sc.array(dims=['x'], values=[10.0, 20.0])
    for i in range(2):
        accum.push(new_var[i].copy())

    # Verify mean is calculated correctly with only the new values
    assert sc.identical(accum.value, sc.mean(new_var))
