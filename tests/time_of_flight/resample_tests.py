# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc

from ess.reduce.time_of_flight import resample


class TestFindStrictlyIncreasingSections:
    """Tests for find_strictly_increasing_sections function."""

    def test_given_monotonically_increasing_values_returns_single_section(self):
        var = sc.array(dims=['x'], values=[1, 2, 3, 4, 5])
        sections = resample.find_strictly_increasing_sections(var)
        assert sections == [slice(0, 5)]

    def test_given_multiple_increasing_sections_returns_corresponding_slices(self):
        var = sc.array(dims=['x'], values=[1, 3, 2, 4, 6, 5, 7, 9])
        sections = resample.find_strictly_increasing_sections(var)
        assert sections == [slice(0, 2), slice(2, 5), slice(5, 8)]

    def test_given_flat_sections_finds_strictly_increasing_parts_only(self):
        var = sc.array(dims=['x'], values=[1, 2, 2, 3, 4, 4, 5])
        sections = resample.find_strictly_increasing_sections(var)
        assert sections == [slice(0, 2), slice(2, 5), slice(5, 7)]

    def test_given_decreasing_values_returns_empty_list(self):
        var = sc.array(dims=['x'], values=[5, 4, 3, 2, 1])
        sections = resample.find_strictly_increasing_sections(var)
        assert sections == []

    def test_given_mixed_behavior_finds_only_increasing_sections(self):
        var = sc.array(dims=['x'], values=[1, 3, 5, 4, 3, 6, 7, 8])
        sections = resample.find_strictly_increasing_sections(var)
        assert sections == [slice(0, 3), slice(4, 8)]

    def test_given_empty_array_returns_empty_list(self):
        var = sc.array(dims=['x'], values=[])
        sections = resample.find_strictly_increasing_sections(var)
        assert sections == []

    def test_given_single_value_returns_empty_list(self):
        var = sc.array(dims=['x'], values=[42])
        sections = resample.find_strictly_increasing_sections(var)
        assert sections == []

    def test_given_two_increasing_values_returns_single_section(self):
        var = sc.array(dims=['x'], values=[1, 2])
        sections = resample.find_strictly_increasing_sections(var)
        assert sections == [slice(0, 2)]

    def test_given_two_equal_values_returns_empty_list(self):
        var = sc.array(dims=['x'], values=[2, 2])
        sections = resample.find_strictly_increasing_sections(var)
        assert sections == []

    def test_given_nan_in_middle_stops_section_at_nan(self):
        var = sc.array(dims=['x'], values=[1, 2, 3, np.nan, 5, 6])
        sections = resample.find_strictly_increasing_sections(var)
        assert sections == [slice(0, 3), slice(4, 6)]

    def test_given_nan_at_beginning_skips_nan(self):
        var = sc.array(dims=['x'], values=[np.nan, 2, 3, 4])
        sections = resample.find_strictly_increasing_sections(var)
        assert sections == [slice(1, 4)]

    def test_given_nan_at_end_stops_section_before_nan(self):
        var = sc.array(dims=['x'], values=[1, 2, 3, np.nan])
        sections = resample.find_strictly_increasing_sections(var)
        assert sections == [slice(0, 3)]

    def test_given_multiple_nans_finds_valid_sections_between(self):
        var = sc.array(dims=['x'], values=[1, np.nan, 3, 4, np.nan, 6, 7])
        sections = resample.find_strictly_increasing_sections(var)
        assert sections == [slice(2, 4), slice(5, 7)]

    def test_given_all_nans_returns_empty_list(self):
        var = sc.array(dims=['x'], values=[np.nan, np.nan, np.nan])
        sections = resample.find_strictly_increasing_sections(var)
        assert sections == []

    def test_given_increasing_section_at_beginning_returns_correct_slice(self):
        var = sc.array(dims=['x'], values=[1, 2, 3, 3, 2, 1])
        sections = resample.find_strictly_increasing_sections(var)
        assert sections == [slice(0, 3)]

    def test_given_increasing_section_in_middle_returns_correct_slice(self):
        var = sc.array(dims=['x'], values=[3, 2, 1, 2, 3, 4, 3, 2])
        sections = resample.find_strictly_increasing_sections(var)
        assert sections == [slice(2, 6)]

    def test_given_increasing_section_at_end_returns_correct_slice(self):
        var = sc.array(dims=['x'], values=[3, 3, 2, 3, 4, 5])
        sections = resample.find_strictly_increasing_sections(var)
        assert sections == [slice(2, 6)]


class TestGetMinMax:
    """Tests for get_min_max function."""

    def test_basic_functionality(self):
        var = sc.array(dims=['x'], values=[1, 2, 3, 2, 3, 4, 5, 3, 4, 5, 6])
        slices = [slice(0, 3), slice(3, 7), slice(7, 11)]
        min_val, max_val = resample.get_min_max(var, dim='x', slices=slices)
        assert min_val.value == 1
        assert max_val.value == 6

    def test_with_units(self):
        var = sc.array(dims=['x'], values=[1.0, 2.0, 3.0, 2.0, 3.0, 4.0], unit='m')
        slices = [slice(0, 3), slice(3, 6)]
        min_val, max_val = resample.get_min_max(var, dim='x', slices=slices)
        assert min_val.value == 1.0
        assert max_val.value == 4.0
        assert min_val.unit == sc.Unit('m')
        assert max_val.unit == sc.Unit('m')

    def test_with_single_slice(self):
        var = sc.array(dims=['x'], values=[1, 2, 3, 4, 5])
        slices = [slice(0, 5)]
        min_val, max_val = resample.get_min_max(var, dim='x', slices=slices)
        assert min_val.value == 1
        assert max_val.value == 5

    def test_with_non_contiguous_slices(self):
        var = sc.array(dims=['x'], values=[1, 2, 3, 4, 5, 0, 1, 2])
        slices = [slice(0, 5), slice(5, 8)]
        min_val, max_val = resample.get_min_max(var, dim='x', slices=slices)
        assert min_val.value == 0  # The min value is at the start of the second slice
        assert max_val.value == 5  # The max value is at the end of the first slice

    def test_with_overlapping_slices(self):
        var = sc.array(dims=['x'], values=[1, 3, 5, 7, 9])
        slices = [slice(0, 3), slice(1, 4), slice(2, 5)]
        min_val, max_val = resample.get_min_max(var, dim='x', slices=slices)
        assert min_val.value == 1  # The min value is at the start of the first slice
        assert max_val.value == 9  # The max value is at the end of the third slice

    def test_with_different_dimension_name(self):
        var = sc.array(dims=['time'], values=[10, 20, 30, 40, 50])
        slices = [slice(1, 3), slice(3, 5)]
        min_val, max_val = resample.get_min_max(var, dim='time', slices=slices)
        assert min_val.value == 20  # The min value is at the start of the first slice
        assert max_val.value == 50  # The max value is at the end of the second slice

    def test_with_float_data(self):
        var = sc.array(dims=['x'], values=[1.1, 2.2, 3.3, 4.4, 5.5])
        slices = [slice(0, 3), slice(3, 5)]
        min_val, max_val = resample.get_min_max(var, dim='x', slices=slices)
        assert min_val.value == 1.1
        assert max_val.value == 5.5

    def test_with_mixed_positive_and_negative_values(self):
        var = sc.array(dims=['x'], values=[-5, -3, -1, 0, 2, 4])
        slices = [slice(0, 3), slice(3, 6)]
        min_val, max_val = resample.get_min_max(var, dim='x', slices=slices)
        assert min_val.value == -5
        assert max_val.value == 4

    def test_with_empty_slices_list_raises_error(self):
        var = sc.array(dims=['x'], values=[1, 2, 3, 4, 5])
        slices = []
        with pytest.raises(ValueError, match="No strictly increasing sections found."):
            resample.get_min_max(var, dim='x', slices=slices)

    def test_integration_with_find_strictly_increasing_sections(self):
        var = sc.array(dims=['x'], values=[1, 2, 3, 2, 3, 4, 1, 2, 3])
        slices = resample.find_strictly_increasing_sections(var)
        min_val, max_val = resample.get_min_max(var, dim='x', slices=slices)
        assert min_val.value == 1
        assert max_val.value == 4


class TestMakeRegularGrid:
    """Tests for make_regular_grid function."""

    def test_basic_functionality(self):
        var = sc.array(dims=['x'], values=[1, 2, 3, 2, 3, 4])
        slices = [slice(0, 3), slice(3, 6)]
        grid = resample.make_regular_grid(var, dim='x', slices=slices)
        assert sc.identical(grid, sc.array(dims=['x'], values=[1, 2, 3, 4]))

    def test_with_units(self):
        var = sc.array(dims=['x'], values=[1.0, 2.0, 3.0, 1.5, 2.5, 3.5], unit='m')
        slices = [slice(0, 3), slice(3, 6)]
        grid = resample.make_regular_grid(var, dim='x', slices=slices)
        expected = sc.array(dims=['x'], values=[1.0, 2.0, 3.0, 4.0], unit='m')
        assert sc.identical(grid, expected)
        assert grid.unit == sc.Unit('m')

    def test_with_different_step_sizes_uses_minimum_section(self):
        # First section has step size 1, second has step size 0.5
        var = sc.array(dims=['x'], values=[1, 2, 3, 3.5, 4, 4.5])
        slices = [slice(0, 3), slice(3, 6)]
        grid = resample.make_regular_grid(var, dim='x', slices=slices)
        # Should use step size 1 from the first section (where min value is located)
        expected = sc.array(dims=['x'], values=[1.0, 2, 3, 4, 5])
        assert sc.identical(grid, expected)

    def test_with_different_dimension_name(self):
        var = sc.array(dims=['time'], values=[10, 20, 30, 5, 15, 25])
        slices = [slice(0, 3), slice(3, 6)]
        grid = resample.make_regular_grid(var, dim='time', slices=slices)
        expected = sc.array(dims=['time'], values=[5, 15, 25, 35])
        assert sc.identical(grid, expected)

    def test_with_float_data(self):
        var = sc.array(dims=['x'], values=[1.1, 2.2, 3.3, 0.5, 1.6, 2.7])
        slices = [slice(0, 3), slice(3, 6)]
        grid = resample.make_regular_grid(var, dim='x', slices=slices)
        expected = sc.array(dims=['x'], values=[0.5, 1.6, 2.7, 3.8])
        assert sc.allclose(grid, expected)

    def test_with_mixed_positive_and_negative_values(self):
        var = sc.array(dims=['x'], values=[-5, -3, -1, 0, 2, 4])
        slices = [slice(0, 3), slice(3, 6)]
        grid = resample.make_regular_grid(var, dim='x', slices=slices)
        expected = sc.array(dims=['x'], values=[-5, -3, -1, 1, 3, 5])
        assert sc.identical(grid, expected)

    def test_with_single_slice(self):
        var = sc.array(dims=['x'], values=[1, 3, 5, 7, 9])
        slices = [slice(0, 5)]
        grid = resample.make_regular_grid(var, dim='x', slices=slices)
        expected = sc.array(dims=['x'], values=[1, 3, 5, 7, 9])
        assert sc.identical(grid, expected)

    def test_with_different_dtype(self):
        var = sc.array(dims=['x'], values=[1, 2, 3, 2, 3, 4], dtype=np.int64)
        slices = [slice(0, 3), slice(3, 6)]
        grid = resample.make_regular_grid(var, dim='x', slices=slices)
        expected = sc.array(dims=['x'], values=[1, 2, 3, 4], dtype=np.int64)
        assert sc.identical(grid, expected)
        assert grid.dtype == np.int64

    def test_empty_slices_list_raises_error(self):
        var = sc.array(dims=['x'], values=[1, 2, 3, 4, 5])
        slices = []
        with pytest.raises(ValueError, match="No strictly increasing sections found."):
            resample.make_regular_grid(var, dim='x', slices=slices)

    def test_integration_with_find_strictly_increasing_sections(self):
        var = sc.array(dims=['x'], values=[1, 3, 5, 2, 4, 6])
        slices = resample.find_strictly_increasing_sections(var)
        grid = resample.make_regular_grid(var, dim='x', slices=slices)
        expected = sc.array(dims=['x'], values=[1, 3, 5, 7])
        assert sc.identical(grid, expected)

    def test_ensures_last_bin_edge_is_included(self):
        var = sc.array(dims=['x'], values=[1, 2, 3, 2, 3, 4, 5])
        slices = [slice(0, 3), slice(3, 7)]
        grid = resample.make_regular_grid(var, dim='x', slices=slices)
        expected = sc.array(dims=['x'], values=[1, 2, 3, 4, 5])
        assert sc.identical(grid, expected)
        # Test that the maximum value from original data is included in the grid
        assert grid[-1].value == 5
