# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical

from ess.reduce.live.raw import LogicalView


@pytest.fixture
def one_d_data():
    """Create a 1D data array for testing."""
    return sc.DataArray(
        sc.array(dims=['x'], values=np.arange(10), unit='counts'),
        coords={'x': sc.arange('x', 10, unit='m')},
    )


@pytest.fixture
def multi_d_data():
    """Create a multi-dimensional data array for testing."""
    values = np.reshape(np.arange(24), (2, 3, 4))
    return sc.DataArray(
        sc.array(dims=['x', 'y', 'z'], values=values, unit='counts'),
        coords={
            'x': sc.arange('x', 2, unit='m'),
            'y': sc.arange('y', 3, unit='m'),
            'z': sc.arange('z', 4, unit='m'),
        },
    )


@pytest.fixture
def foldable_data_24():
    """Create a data array that needs to be folded."""
    values = np.arange(24)
    return sc.DataArray(
        sc.array(dims=['detector'], values=values, unit='counts'),
        coords={'detector': sc.arange('detector', 24, unit=None)},
    )


def test_logical_view_default_settings():
    """Test that LogicalView with default settings returns a copy of the input."""
    da = sc.DataArray(
        sc.array(dims=['x'], values=np.arange(5), unit='counts'),
        coords={'x': sc.arange('x', 5, unit='m')},
    )
    view = LogicalView()
    result = view(da)

    assert_identical(result, da)
    # Check it's a copy, not the same object
    assert result is not da


def test_fold_dimension(foldable_data_24):
    """Test folding a single dimension into multiple dimensions."""
    view = LogicalView(fold={'x': 2, 'y': 3, 'z': 4})
    result = view(foldable_data_24)

    expected = sc.DataArray(
        sc.array(
            dims=['x', 'y', 'z'],
            values=foldable_data_24.values.reshape(2, 3, 4),
            unit='counts',
        ),
        coords={},
    )
    assert_identical(result.data, expected.data)
    assert result.dims == ('x', 'y', 'z')


def test_transpose_dimensions(multi_d_data):
    """Test transposing dimensions."""
    view = LogicalView(transpose=('z', 'y', 'x'))
    result = view(multi_d_data)

    expected = multi_d_data.transpose(['z', 'y', 'x'])
    assert_identical(result, expected)


def test_select_single_index(multi_d_data):
    """Test selecting a single index from a dimension."""
    view = LogicalView(select={'x': 1})
    result = view(multi_d_data)

    expected = multi_d_data['x', 1]
    assert_identical(result, expected)


def test_select_slice(multi_d_data):
    """Test selecting a slice from a dimension."""
    view = LogicalView(select={'y': slice(1, 3)})
    result = view(multi_d_data)

    expected = multi_d_data['y', 1:3]
    assert_identical(result, expected)


def test_sum_single_dimension(multi_d_data):
    """Test summing over a single dimension."""
    view = LogicalView(sum='x')
    result = view(multi_d_data)

    expected = multi_d_data.sum('x')
    assert_identical(result, expected)


def test_sum_multiple_dimensions(multi_d_data):
    """Test summing over multiple dimensions."""
    view = LogicalView(sum=('x', 'y'))
    result = view(multi_d_data)

    expected = multi_d_data.sum(['x', 'y'])
    assert_identical(result, expected)


def test_sum_all_dimensions(multi_d_data):
    """Test summing over all dimensions."""
    view = LogicalView(sum=None)
    result = view(multi_d_data)

    expected = multi_d_data.sum()
    assert_identical(result, expected)


def test_flatten_dimensions(multi_d_data):
    """Test flattening dimensions."""
    view = LogicalView(flatten={'xy': ['x', 'y']})
    result = view(multi_d_data)

    expected = multi_d_data.flatten(['x', 'y'], to='xy')
    assert_identical(result, expected)


def test_flatten_multiple_groups():
    """Test flattening multiple groups of dimensions."""
    # Create a 4D array
    values = np.reshape(np.arange(60), (2, 3, 5, 2))
    da = sc.DataArray(
        sc.array(dims=['a', 'b', 'c', 'd'], values=values, unit='counts'),
        coords={
            'a': sc.arange('a', 2),
            'b': sc.arange('b', 3),
            'c': sc.arange('c', 5),
            'd': sc.arange('d', 2),
        },
    )

    view = LogicalView(flatten={'ab': ['a', 'b'], 'cd': ['c', 'd']})
    result = view(da)

    expected = da.flatten(['a', 'b'], to='ab').flatten(['c', 'd'], to='cd')
    assert_identical(result, expected)


def test_combination_of_operations(multi_d_data):
    """Test a combination of multiple operations."""
    view = LogicalView(
        select={'z': 0}, transpose=('y', 'x'), sum='y', flatten={'flat': ['x']}
    )
    result = view(multi_d_data)

    expected = (
        multi_d_data['z', 0].transpose(['y', 'x']).sum('y').flatten(['x'], to='flat')
    )
    assert_identical(result, expected)


def test_fold_and_select(foldable_data_24):
    """Test folding and then selecting."""
    view = LogicalView(fold={'x': 4, 'y': 6}, select={'x': 2})
    result = view(foldable_data_24)

    values = foldable_data_24.values.reshape(4, 6)
    expected = sc.DataArray(
        sc.array(dims=['y'], values=values[2, :], unit='counts'),
        coords={},
    )
    assert_identical(result.data, expected.data)


def test_fold_and_transpose(foldable_data_24):
    """Test folding and then transposing."""
    view = LogicalView(fold={'x': 3, 'y': 8}, transpose=('y', 'x'))
    result = view(foldable_data_24)

    values = foldable_data_24.values.reshape(3, 8).transpose(1, 0)
    expected = sc.DataArray(
        sc.array(dims=['y', 'x'], values=values, unit='counts'),
        coords={},
    )
    assert_identical(result.data, expected.data)


def test_transpose_and_sum(multi_d_data):
    """Test transposing and then summing."""
    view = LogicalView(transpose=('z', 'y', 'x'), sum='z')
    result = view(multi_d_data)

    expected = multi_d_data.transpose(['z', 'y', 'x']).sum('z')
    assert_identical(result, expected)


def test_fold_transpose_select_sum_flatten(foldable_data_24):
    """Test a complex combination of all operations."""
    view = LogicalView(
        fold={'x': 2, 'y': 3, 'z': 4},
        transpose=('z', 'y', 'x'),
        select={'z': slice(1, 3)},
        sum='y',
        flatten={'xz': ['z', 'x']},
    )
    result = view(foldable_data_24)

    # Manually apply all operations
    values = foldable_data_24.values.reshape(2, 3, 4).transpose(2, 1, 0)[1:3, :, :]
    values = values.sum(axis=1)
    values = values.reshape(-1)

    expected = sc.DataArray(
        sc.array(dims=['xz'], values=values, unit='counts'),
        coords={},
    )
    assert_identical(result.data, expected.data)
    assert result.dims == ('xz',)


def test_select_nonexistent_dimension(multi_d_data):
    """Test selecting from a non-existent dimension."""
    view = LogicalView(select={'nonexistent': 0})
    with pytest.raises(sc.DimensionError):
        view(multi_d_data)


def test_sum_nonexistent_dimension(multi_d_data):
    """Test summing over a non-existent dimension."""
    view = LogicalView(sum='nonexistent')
    with pytest.raises(sc.DimensionError):
        view(multi_d_data)


def test_flatten_nonexistent_dimensions(multi_d_data):
    """Test flattening non-existent dimensions."""
    view = LogicalView(flatten={'flat': ['nonexistent']})
    with pytest.raises(sc.DimensionError):
        view(multi_d_data)


def test_transpose_wrong_dimensions(multi_d_data):
    """Test transposing with wrong dimensions."""
    view = LogicalView(transpose=('x', 'y', 'nonexistent'))
    with pytest.raises(sc.DimensionError):
        view(multi_d_data)


def test_fold_incompatible_size(foldable_data_24):
    """Test folding with incompatible sizes."""
    # Total folded size would be 3*9=27, but we have 24 elements
    view = LogicalView(fold={'x': 3, 'y': 9})
    with pytest.raises(sc.DimensionError):
        view(foldable_data_24)


def test_fold_transpose_incompatible(foldable_data_24):
    """Test folding with dimensions that don't match transpose."""
    view = LogicalView(fold={'x': 2, 'y': 12}, transpose=('x', 'y', 'z'))
    with pytest.raises(sc.DimensionError):
        view(foldable_data_24)


def test_fold_non_flat_array(multi_d_data):
    """Test folding on an array that already has multiple dimensions."""
    view = LogicalView(fold={'new_dim': 2})
    # Expecting any error since we just know folding multi-D arrays isn't supported
    with pytest.raises((sc.DimensionError, ValueError)):
        view(multi_d_data)


def test_select_out_of_bounds(multi_d_data):
    """Test selecting an index that's out of bounds."""
    view = LogicalView(select={'x': 5})  # x has only 2 elements
    with pytest.raises(IndexError):
        view(multi_d_data)


def test_select_after_transpose(multi_d_data):
    """Test the actual operation order (select before transpose)."""
    # In LogicalView, select happens before transpose
    view1 = LogicalView(select={'z': 0}, transpose=('y', 'x'))
    result1 = view1(multi_d_data)

    # Correct expected value - select first, then transpose
    expected1 = multi_d_data['z', 0].transpose(['y', 'x'])
    assert_identical(result1, expected1)

    # Test a different case
    view2 = LogicalView(select={'y': 1}, transpose=('x', 'z'))
    result2 = view2(multi_d_data)

    # Operations happen in this order: select, then transpose
    expected2 = multi_d_data['y', 1].transpose(['x', 'z'])
    assert_identical(result2, expected2)


def test_empty_operations(multi_d_data):
    view = LogicalView(fold=None, transpose=None, select={}, sum=(), flatten={})
    result = view(multi_d_data)
    assert_identical(result, multi_d_data)
