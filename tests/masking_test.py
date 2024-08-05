# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc

from ess.reduce.masking import mask_range


def test_mask_range_dense_data():
    x = sc.arange('x', 5.0, unit='m')
    da = sc.DataArray(data=x, coords={'x': x})
    mask = sc.DataArray(
        data=sc.array(dims=['x'], values=[True]),
        coords={
            'x': sc.array(dims=['x'], values=[1.9, 3.001], unit='m'),
        },
    )
    masked = mask_range(da, mask=mask, name='mymask')
    assert masked.masks['mymask'].values.tolist() == [False, False, True, True, False]


def test_mask_range_dense_data_bin_edges():
    x = sc.arange('x', 6.0, unit='m')
    da = sc.DataArray(data=x[:-1], coords={'x': x})
    mask = sc.DataArray(
        data=sc.array(dims=['x'], values=[True]),
        coords={
            'x': sc.array(dims=['x'], values=[1.9, 3.001], unit='m'),
        },
    )
    masked = mask_range(da, mask=mask, name='mymask')
    assert masked.masks['mymask'].values.tolist() == [False, True, True, True, False]


def test_mask_range_dense_data_two_ranges():
    x = sc.arange('x', 11.0, unit='m')
    da = sc.DataArray(data=x[:-1], coords={'x': x})
    mask = sc.DataArray(
        data=sc.array(dims=['x'], values=[True, False, True]),
        coords={
            'x': sc.array(dims=['x'], values=[1.9, 3.001, 6.5, 8.8], unit='m'),
        },
    )
    masked = mask_range(da, mask=mask, name='mymask')
    assert masked.masks['mymask'].values.tolist() == [
        False,
        True,
        True,
        True,
        False,
        False,
        True,
        True,
        True,
        False,
    ]


def test_mask_range_binned_data_no_prior_binning():
    x = sc.arange('x', 5.0, unit='m')
    da = sc.DataArray(data=x, coords={'x': x, 'y': x + x.max()})
    binned = da.bin(y=2)
    mask = sc.DataArray(
        data=sc.array(dims=['x'], values=[True]),
        coords={
            'x': sc.array(dims=['x'], values=[1.9, 3.001], unit='m'),
        },
    )
    masked = mask_range(binned, mask=mask, name='mymask')
    # This should make 3 bins that span the entire data range, and where the middle
    # one is masked.
    assert masked.masks['mymask'].values.tolist() == [False, True, False]
    binned_full_range = binned.bin(x=1)
    assert sc.identical(masked.coords['x'][0], binned_full_range.coords['x'][0])
    assert sc.identical(masked.coords['x'][1:3], mask.coords['x'])
    assert sc.identical(masked.coords['x'][3], binned_full_range.coords['x'][1])


def test_mask_range_binned_data_has_prior_single_bin():
    x = sc.arange('x', 5.0, unit='m')
    da = sc.DataArray(data=x, coords={'x': x, 'y': x + x.max()})
    binned = da.bin(x=1)
    mask = sc.DataArray(
        data=sc.array(dims=['x'], values=[True]),
        coords={
            'x': sc.array(dims=['x'], values=[1.5, 3.001], unit='m'),
        },
    )
    masked = mask_range(binned, mask=mask, name='mymask')
    assert masked.masks['mymask'].values.tolist() == [False, True, False]
    assert np.allclose(masked.coords['x'].values, [0.0, 1.5, 3.001, 4.0])


def test_mask_range_binned_data_has_prior_multiple_bins():
    x = sc.arange('x', 5.0, unit='m')
    da = sc.DataArray(data=x, coords={'x': x, 'y': x + x.max()})
    binned = da.bin(x=2)
    mask = sc.DataArray(
        data=sc.array(dims=['x'], values=[True]),
        coords={
            'x': sc.array(dims=['x'], values=[1.5, 3.001], unit='m'),
        },
    )
    masked = mask_range(binned, mask=mask, name='mymask')
    assert masked.masks['mymask'].values.tolist() == [False, True, True, False]
    assert np.allclose(masked.coords['x'].values, [0.0, 1.5, 2.0, 3.001, 4.0])


def test_mask_range_binned_data_has_already_same_edge_as_mask():
    x = sc.arange('x', 5.0, unit='m')
    da = sc.DataArray(data=x, coords={'x': x, 'y': x + x.max()})
    binned = da.bin(x=2)
    mask = sc.DataArray(
        data=sc.array(dims=['x'], values=[True]),
        coords={
            'x': sc.array(
                dims=['x'], values=[binned.coords['x'][1].value, 3.001], unit='m'
            ),
        },
    )
    masked = mask_range(binned, mask=mask, name='mymask')
    assert masked.masks['mymask'].values.tolist() == [False, True, False]
    assert np.allclose(masked.coords['x'].values, [0.0, 2.0, 3.001, 4.0])


def test_mask_range_with_midpoints_coord_raises():
    x = sc.arange('x', 5.0, unit='m')
    da = sc.DataArray(data=x, coords={'x': x})
    mask = sc.DataArray(
        data=sc.array(dims=['x'], values=[True, False]),
        coords={
            'x': sc.array(dims=['x'], values=[1.9, 3.001], unit='m'),
        },
    )
    with pytest.raises(
        sc.DimensionError, match='Coordinate x must be bin-edges to mask a range'
    ):
        _ = mask_range(da, mask=mask, name='mymask')


def test_mask_range_on_dense_data_with_two_dimensional_coord_raises():
    xy = sc.arange('x', 20.0, unit='m').fold(dim='x', sizes={'y': 4, 'x': 5})
    da = sc.DataArray(data=xy, coords={'x': xy})
    mask = sc.DataArray(
        data=sc.array(dims=['x'], values=[True]),
        coords={
            'x': sc.array(dims=['x'], values=[1.9, 3.001], unit='m'),
        },
    )
    with pytest.raises(
        sc.DimensionError,
        match='Cannot mask range on data with multi-dimensional coordinate',
    ):
        _ = mask_range(da, mask=mask, name='mymask')


def test_mask_range_on_binned_data_with_two_dimensional_coord_raises():
    x = sc.arange('x', 10.0, unit='m')
    da = sc.DataArray(data=x, coords={'x': x, 'y': x * 1.5})
    binned = da.bin(x=2, y=2)
    binned.coords['x'] = sc.broadcast(binned.coords['x'], sizes={'x': 3, 'y': 2})
    mask = sc.DataArray(
        data=sc.array(dims=['x'], values=[True]),
        coords={
            'x': sc.array(dims=['x'], values=[1.9, 3.001], unit='m'),
        },
    )
    with pytest.raises(
        sc.DimensionError,
        match='Cannot mask range on data with multi-dimensional coordinate',
    ):
        _ = mask_range(binned, mask=mask, name='mymask')


def test_mask_range_on_data_with_existing_mask_of_same_name_raises():
    x = sc.arange('x', 10.0, unit='m')
    da = sc.DataArray(
        data=x,
        coords={'x': x, 'y': x * 1.5},
        masks={'mymask': x > sc.scalar(5.0, unit='m')},
    )
    mask = sc.DataArray(
        data=sc.array(dims=['x'], values=[True]),
        coords={
            'x': sc.array(dims=['x'], values=[1.9, 3.001], unit='m'),
        },
    )
    with pytest.raises(
        ValueError,
        match='Mask mymask already exists in data array and would be overwritten',
    ):
        _ = mask_range(da, mask=mask, name='mymask')
