# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical

from ess.reduce import uncertainty as unc


def test_fail_mode_always_returns_original():
    broadcaster = unc.broadcasters[unc.UncertaintyBroadcastMode.fail]
    var = sc.ones(dims=['x', 'y'], shape=[2, 3], with_variances=True)
    assert broadcaster(var, prototype=sc.zeros(sizes={'x': 2})) is var
    assert broadcaster(var, prototype=sc.zeros(sizes={'y': 3})) is var
    assert broadcaster(var, prototype=sc.zeros(sizes={'x': 2, 'y': 3})) is var
    assert broadcaster(var, prototype=sc.zeros(sizes={'z': 5})) is var


@pytest.mark.parametrize(
    'mode',
    [unc.UncertaintyBroadcastMode.drop, unc.UncertaintyBroadcastMode.upper_bound],
)
def test_broadcaster_returns_original_if_no_new_dims(mode):
    broadcaster = unc.broadcasters[mode]
    var = sc.ones(dims=['x', 'y'], shape=[2, 3], with_variances=True)
    assert broadcaster(var, prototype=sc.zeros(sizes={'x': 2})) is var
    assert broadcaster(var, prototype=sc.zeros(sizes={'y': 3})) is var
    assert broadcaster(var, prototype=sc.zeros(sizes={'x': 2, 'y': 3})) is var


@pytest.mark.parametrize(
    'mode',
    [unc.UncertaintyBroadcastMode.drop, unc.UncertaintyBroadcastMode.upper_bound],
)
def test_broadcaster_original_if_no_variances(mode):
    broadcaster = unc.broadcasters[mode]
    var = sc.ones(dims=['x'], shape=[2], with_variances=False)
    assert broadcaster(var, prototype=sc.zeros(sizes={'y': 3})) is var


def test_broadcast_scales_variances_by_new_subspace_volume():
    x = sc.linspace('x', 0.0, 1.0, 2)
    y = sc.linspace('y', 0.0, 2.0, 3)
    values = x * y
    var = values.copy()
    var.variances = var.values
    expected = sc.ones(dims=['z'], shape=[1]) * values
    expected.variances = 1 * expected.values
    assert_identical(
        unc.broadcast_with_upper_bound_variances(
            var, prototype=sc.zeros(sizes={'z': 1})
        ),
        expected,
    )
    expected = sc.ones(dims=['z'], shape=[2]) * values
    expected.variances = 2 * expected.values
    assert_identical(
        unc.broadcast_with_upper_bound_variances(
            var, prototype=sc.zeros(sizes={'z': 2})
        ),
        expected,
    )
    expected = sc.ones(dims=['z'], shape=[2]) * values
    expected.variances = 2 * expected.values
    assert_identical(
        unc.broadcast_with_upper_bound_variances(
            var, prototype=sc.zeros(sizes={'y': 3, 'z': 2})
        ),
        expected.transpose(['y', 'z', 'x']),
    )


def test_drop_mode_drops_if_new_dims():
    broadcaster = unc.broadcasters[unc.UncertaintyBroadcastMode.drop]
    var = sc.ones(dims=['x', 'y'], shape=[2, 3], with_variances=True)
    assert_identical(
        broadcaster(var, prototype=sc.zeros(sizes={'z': 1})), sc.values(var)
    )


def test_broadcast_with_mask_along_single_broadcast_dim_does_not_count_masked():
    x = sc.ones(dims=['x'], shape=[2], with_variances=True)
    y = sc.DataArray(
        sc.ones(dims=['y'], shape=[3]),
        masks={'y': sc.array(dims=['y'], values=[False, True, False])},
    )
    xy = unc.broadcast_with_upper_bound_variances(x, prototype=y)
    expected = sc.ones(dims=['y', 'x'], shape=[3, 2], with_variances=True)
    expected.variances *= 2
    expected['y', 1].variances = [np.inf, np.inf]
    assert_identical(xy, expected)


def test_broadcast_with_mask_along_existing_dim_has_no_effect():
    x = sc.ones(dims=['x'], shape=[2], with_variances=True)
    y = sc.DataArray(
        sc.ones(dims=['y'], shape=[3]),
        masks={'x': sc.array(dims=['x'], values=[True, False])},
    )
    xy = unc.broadcast_with_upper_bound_variances(x, prototype=y)
    expected = sc.ones(dims=['y', 'x'], shape=[3, 2], with_variances=True)
    expected.variances *= 3
    assert_identical(xy, expected)


def test_broadcast_2d_with_mask_along_single_broadcast_dim_does_not_count_masked():
    x = sc.ones(dims=['x'], shape=[2], with_variances=True)
    y = sc.DataArray(
        sc.ones(dims=['z', 'y'], shape=[2, 3]),
        masks={'y': sc.array(dims=['y'], values=[False, True, False])},
    )
    xy = unc.broadcast_with_upper_bound_variances(x, prototype=y)
    expected = sc.ones(dims=['z', 'y', 'x'], shape=[2, 3, 2], with_variances=True)
    expected.variances *= 4
    expected['y', 1].variances = [[np.inf, np.inf], [np.inf, np.inf]]
    assert_identical(xy, expected)


def test_broadcast_into_orthogonal_2d_mask_does_not_count_masked():
    x = sc.ones(dims=['x'], shape=[2], with_variances=True)
    y = sc.DataArray(
        sc.ones(dims=['z', 'y'], shape=[2, 3]),
        masks={
            'zy': sc.array(
                dims=['z', 'y'], values=[[False, False, False], [False, True, False]]
            )
        },
    )
    xy = unc.broadcast_with_upper_bound_variances(x, prototype=y)
    expected = sc.ones(dims=['z', 'y', 'x'], shape=[2, 3, 2], with_variances=True)
    expected.variances *= 5
    expected['z', 1]['y', 1].variances = [np.inf, np.inf]
    assert_identical(xy, expected)


def test_broadcast_into_nonorthogonal_2d_mask_ignores_mask():
    x = sc.ones(dims=['x'], shape=[2], with_variances=True)
    # There is no simple way of handling this, so we just ignore the mask.
    # This is ok since it will simply mean the upper bound is larger than strictly
    # necessary.
    y = sc.DataArray(
        sc.ones(dims=['x', 'y'], shape=[2, 3]),
        masks={
            'xy': sc.array(
                dims=['x', 'y'], values=[[False, False, False], [False, True, False]]
            )
        },
    )
    xy = unc.broadcast_with_upper_bound_variances(x, prototype=y)
    expected = sc.ones(dims=['x', 'y'], shape=[2, 3], with_variances=True)
    expected.variances *= 3
    assert_identical(xy, expected)


def test_broadcast_into_nonorthogonal_2d_mask_reducible_mask_counts_masked():
    x = sc.ones(dims=['x'], shape=[2], with_variances=True)
    # There is no simple way of handling this, so we just ignore the mask.
    # This is ok since it will simply mean the upper bound is larger than strictly
    # necessary.
    y = sc.DataArray(
        sc.ones(dims=['x', 'y'], shape=[2, 3]),
        masks={
            'xy': sc.array(
                dims=['x', 'y'], values=[[False, True, False], [False, True, False]]
            )
        },
    )
    xy = unc.broadcast_with_upper_bound_variances(x, prototype=y)
    expected = sc.ones(dims=['x', 'y'], shape=[2, 3], with_variances=True)
    expected.variances *= 2
    expected['y', 1].variances = [np.inf, np.inf]
    assert_identical(xy, expected)


def test_upper_bound_broadcast_raises_if_input_is_binned():
    x = sc.linspace('x', 0.0, 1.0, 10).bin(x=1).squeeze()
    x.value.variances = x.value.values
    y = sc.linspace('y', 0.0, 1.0, 10)
    with pytest.raises(ValueError, match="Cannot broadcast binned data."):
        unc.broadcast_with_upper_bound_variances(x, prototype=y)


def test_upper_bound_event_broadcast_raises_if_binning_mismatching():
    prototype = sc.linspace('x', 0.0, 1.0, 10).bin(x=3).squeeze()
    data = sc.DataArray(
        sc.ones(dims=['x'], shape=[2], with_variances=True),
        coords={'x': sc.linspace('x', 0.0, 1.0, 3)},
    )
    with pytest.raises(
        ValueError, match="Mismatching binning not supported in broadcast."
    ):
        unc.broadcast_with_upper_bound_variances(data, prototype=prototype)


def test_upper_bound_event_broadcast_counts_events():
    content = sc.ones(dims=['event'], shape=[10])
    # sizes = [0,1,2,4,3]
    begin = sc.array(dims=['x'], values=[0, 0, 1, 3, 7], unit=None)
    prototype = sc.bins(data=content, dim='event', begin=begin)
    # We are not supporting the case of prototype missing dims of the data.
    prototype = sc.concat([prototype, prototype], 'y')
    y = sc.array(dims=['y'], values=[1.0, 2.0], variances=[1.0, 2.0])
    upper_bound_broadcast = unc.broadcast_with_upper_bound_variances(
        y, prototype=prototype
    )

    expected = prototype.copy().bins.constituents
    expected['data'].values[:10] = 1.0
    expected['data'].values[10:] = 2.0
    # There are 5 bins along x, but 10 events, so variance scale factor is 10.
    expected['data'].variances = expected['data'].values * 10
    expected = sc.bins(**expected)

    assert_identical(upper_bound_broadcast, expected)
    # The point of broadcast_with_upper_bound_variances is that we can afterwards
    # perform the following operation with getting a variance broadcast error.
    # Did it work?
    _ = prototype * upper_bound_broadcast


def test_upper_bound_event_broadcast_event_count_excludes_masked():
    content = sc.ones(dims=['event'], shape=[10])
    # sizes = [0,1,2,4,3]
    begin = sc.array(dims=['x'], values=[0, 0, 1, 3, 7], unit=None)
    prototype = sc.bins(data=content, dim='event', begin=begin)
    # We are not supporting the case of prototype missing dims of the data.
    prototype = sc.DataArray(sc.concat([prototype, prototype], 'y'))
    prototype.masks['x'] = sc.array(
        dims=['x'], values=[False, True, False, False, True]
    )
    y = sc.array(dims=['y'], values=[1.0, 2.0], variances=[1.0, 2.0])
    upper_bound_broadcast = unc.broadcast_with_upper_bound_variances(
        y, prototype=prototype
    )

    expected = prototype.copy().bins.constituents
    expected['data'].values[:10] = 1.0
    expected['data'].values[10:] = 2.0
    # There are 5 bins along x, but 10 events (4 masked), so variance scale factor is 6.
    expected['data'].variances = expected['data'].values * 6
    expected['data']['event', 0:1].variances = [np.inf]
    expected['data']['event', 7:10].variances = [np.inf, np.inf, np.inf]
    expected['data']['event', 10:11].variances = [np.inf]
    expected['data']['event', 17:20].variances = [np.inf, np.inf, np.inf]
    expected = sc.bins(**expected)

    assert_identical(upper_bound_broadcast, expected)
    # The point of broadcast_with_upper_bound_variances is that we can afterwards
    # perform the following operation with getting a variance broadcast error.
    # Did it work?
    _ = prototype * upper_bound_broadcast
    assert sc.all(sc.isinf(sc.variances(upper_bound_broadcast['x', 1])))
    assert sc.all(sc.isinf(sc.variances(upper_bound_broadcast['x', 4])))


def test_upper_bound_event_broadcast_only_bin_broadcast():
    content = sc.ones(dims=['event'], shape=[10])
    begin = sc.array(dims=['x'], values=[0, 3, 6], unit=None)
    prototype = sc.DataArray(sc.bins(data=content, dim='event', begin=begin))
    prototype.masks['x'] = sc.array(dims=['x'], values=[False, True, False])
    x = sc.array(dims=['x'], values=[1.0, 2.0, 3.0], variances=[1.0, 2.0, 3.0])
    # Data has same dims as prototype, broadcast is only into bins
    assert x.sizes == prototype.sizes
    upper_bound_broadcast = unc.broadcast_with_upper_bound_variances(
        x, prototype=prototype
    )

    expected = prototype.copy().bins.constituents
    expected['data'].values[:3] = 1.0
    expected['data'].values[3:6] = 2.0
    expected['data'].values[6:] = 3.0
    expected['data'].variances = expected['data'].values
    expected['data']['event', 0:3].variances = [3.0, 3.0, 3.0]
    # The x=1 bin is masked, but it is not being broadcasted, so no inf.
    expected['data']['event', 3:6].variances = [6.0, 6.0, 6.0]
    expected['data']['event', 6:].variances = [12.0, 12.0, 12.0, 12.0]
    expected = sc.bins(**expected)

    assert_identical(upper_bound_broadcast, expected)
    # The point of broadcast_with_upper_bound_variances is that we can afterwards
    # perform the following operation with getting a variance broadcast error.
    # Did it work?
    _ = prototype * upper_bound_broadcast
