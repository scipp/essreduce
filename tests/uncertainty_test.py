# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc
from scipp.testing import assert_identical

from ess.reduce import uncertainty as unc


def test_fail_mode_always_returns_original():
    broadcaster = unc.broadcasters[unc.UncertaintyBroadcastMode.fail]
    var = sc.ones(dims=['x', 'y'], shape=[2, 3], with_variances=True)
    assert broadcaster(var, {'x': 2}) is var
    assert broadcaster(var, {'y': 3}) is var
    assert broadcaster(var, {'x': 2, 'y': 3}) is var
    assert broadcaster(var, {'z': 5}) is var


@pytest.mark.parametrize(
    'mode',
    [unc.UncertaintyBroadcastMode.drop, unc.UncertaintyBroadcastMode.upper_bound],
)
def test_broadcaster_returns_original_if_no_new_dims(mode):
    broadcaster = unc.broadcasters[mode]
    var = sc.ones(dims=['x', 'y'], shape=[2, 3], with_variances=True)
    assert broadcaster(var, {'x': 2}) is var
    assert broadcaster(var, {'y': 3}) is var
    assert broadcaster(var, {'x': 2, 'y': 3}) is var


@pytest.mark.parametrize(
    'mode',
    [unc.UncertaintyBroadcastMode.drop, unc.UncertaintyBroadcastMode.upper_bound],
)
def test_broadcaster_original_if_no_variances(mode):
    broadcaster = unc.broadcasters[mode]
    var = sc.ones(dims=['x'], shape=[2], with_variances=False)
    assert broadcaster(var, {'y': 3}) is var


def test_broadcast_scales_variances_by_new_subspace_volume():
    x = sc.linspace('x', 0.0, 1.0, 2)
    y = sc.linspace('y', 0.0, 2.0, 3)
    values = x * y
    var = values.copy()
    var.variances = var.values
    expected = sc.ones(dims=['z'], shape=[1]) * values
    expected.variances = 1 * expected.values
    assert_identical(unc.broadcast_with_upper_bound_variances(var, {'z': 1}), expected)
    expected = sc.ones(dims=['z'], shape=[2]) * values
    expected.variances = 2 * expected.values
    assert_identical(unc.broadcast_with_upper_bound_variances(var, {'z': 2}), expected)
    expected = sc.ones(dims=['z'], shape=[2]) * values
    expected.variances = 2 * expected.values
    assert_identical(
        unc.broadcast_with_upper_bound_variances(var, {'y': 3, 'z': 2}),
        expected.transpose(['y', 'z', 'x']),
    )


def test_drop_mode_drops_if_new_dims():
    broadcaster = unc.broadcasters[unc.UncertaintyBroadcastMode.drop]
    var = sc.ones(dims=['x', 'y'], shape=[2, 3], with_variances=True)
    assert_identical(broadcaster(var, {'z': 1}), sc.values(var))
