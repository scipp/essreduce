# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical

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

    def test_given_extended_flat_sections_finds_strictly_increasing_parts_only(self):
        var = sc.array(dims=['x'], values=[1, 2, 2, 2, 3, 4, 4, 5])
        sections = resample.find_strictly_increasing_sections(var)
        assert sections == [slice(0, 2), slice(3, 6), slice(6, 8)]

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


class TestRebinStrictlyIncreasing:
    """Tests for rebin_strictly_increasing function."""

    def test_basic_functionality(self):
        # Create a data array with a simple time-of-flight coordinate that has two
        # strictly increasing sections
        tof = sc.array(dims=['tof'], values=[1, 2, 3, 2, 3, 4, 5])
        # 666 is in the "unphysical" negative-size bin and should be dropped.
        data = sc.array(dims=['tof'], values=[10, 20, 666, 25, 35, 11])
        da = sc.DataArray(data=data, coords={'tof': tof})

        result = resample.rebin_strictly_increasing(da, 'tof')

        # Check the rebinned result has a regular grid from 1 to 5
        expected_tof = sc.array(dims=['tof'], values=[1.0, 2, 3, 4, 5])
        assert sc.identical(result.coords['tof'], expected_tof)

        # Check the data values are properly rebinned and combined
        expected_data = sc.array(dims=['tof'], values=[10.0, 20 + 25, 35, 11])
        assert_identical(result.data, expected_data)

    def test_with_different_step_sizes(self):
        # First section has step size 1, second has step size 0.5
        tof = sc.array(dims=['tof'], values=[1, 2, 4, 3.5, 4, 4.5, 5])
        data = sc.array(dims=['tof'], values=[10, 20, 15, 25, 35, 45])
        da = sc.DataArray(data=data, coords={'tof': tof})

        result = resample.rebin_strictly_increasing(da, 'tof')

        # Should use step size 1 from the first section (where min value is found)
        expected_tof = sc.array(dims=['tof'], values=[1.0, 2, 3, 4, 5])
        assert_identical(result.coords['tof'], expected_tof)

    def test_with_units(self):
        tof = sc.array(dims=['tof'], values=[1.0, 2.0, 3.0, 2.0, 3.0, 4.0], unit='ms')
        data = sc.array(dims=['tof'], values=[10, 20, 15, 25, 35], unit='counts')
        da = sc.DataArray(data=data, coords={'tof': tof})

        result = resample.rebin_strictly_increasing(da, 'tof')

        # Check units are preserved
        assert result.coords['tof'].unit == sc.Unit('ms')
        assert result.data.unit == sc.Unit('counts')

        # Check values
        expected_tof = sc.array(dims=['tof'], values=[1.0, 2.0, 3.0, 4.0], unit='ms')
        assert sc.identical(result.coords['tof'], expected_tof)

    def test_with_single_increasing_section(self):
        tof = sc.array(dims=['tof'], values=[1, 2, 3, 4, 5, 6])
        data = sc.array(dims=['tof'], values=[10, 20, 30, 40, 50])
        da = sc.DataArray(data=data, coords={'tof': tof})

        result = resample.rebin_strictly_increasing(da, 'tof')

        # For a single increasing section, should return just that section
        assert sc.identical(result, da)

    def test_with_single_section_minimum_length(self):
        """Test with a single section of the minimum length (2 points)."""
        tof = sc.array(dims=['tof'], values=[1, 2])
        data = sc.array(dims=['tof'], values=[10])
        da = sc.DataArray(data=data, coords={'tof': tof})

        result = resample.rebin_strictly_increasing(da, 'tof')

        # Should return the original section without rebinning
        assert sc.identical(result, da)

    def test_with_single_section_integer_coords(self):
        """Test with a single section with integer coordinates."""
        tof = sc.array(dims=['tof'], values=[1, 2, 3, 4], dtype=np.int32)
        data = sc.array(dims=['tof'], values=[10, 20, 30])
        da = sc.DataArray(data=data, coords={'tof': tof})

        result = resample.rebin_strictly_increasing(da, 'tof')

        # Should return the original section, with converted coords
        assert sc.identical(result, da)
        # Check that we're maintaining the same values even if the dtype changed
        assert sc.allclose(result.coords['tof'], da.coords['tof'])

    def test_with_single_section_with_masks_and_extra_coords(self):
        """Test with a single section that has masks and attributes."""
        tof = sc.array(dims=['tof'], values=[1, 2, 3, 4, 5])
        data = sc.array(dims=['tof'], values=[10, 20, 30, 40])
        da = sc.DataArray(data=data, coords={'tof': tof})

        # Add mask and attribute
        mask = sc.array(dims=['tof'], values=[False, True, False, False])
        da.masks['quality'] = mask
        da.coords['test_attr'] = sc.scalar(42)

        result = resample.rebin_strictly_increasing(da, 'tof')

        # Should preserve the original data with masks and attributes
        assert sc.identical(result, da)
        assert 'quality' in result.masks
        assert sc.identical(result.masks['quality'], mask)
        assert 'test_attr' in result.coords
        assert sc.identical(result.coords['test_attr'], sc.scalar(42))

    def test_with_three_increasing_sections(self):
        tof = sc.array(dims=['tof'], values=[1, 2, 3, 1, 2, 3, 4, 2, 3, 4, 5])
        data = sc.array(dims=['tof'], values=[5, 10, 6, 12, 18, 8, 14, 21, 28, 35])
        da = sc.DataArray(data=data, coords={'tof': tof})

        result = resample.rebin_strictly_increasing(da, 'tof')

        expected_tof = sc.array(dims=['tof'], values=[1.0, 2, 3, 4, 5])
        assert_identical(result.coords['tof'], expected_tof)

        # Sum of all three sections properly rebinned
        expected_data = sc.array(
            dims=['tof'], values=[5.0 + 12, 10 + 18 + 21, 8 + 28, 35]
        )
        assert_identical(result.data, expected_data)

    def test_with_nan_values_in_coordinate(self):
        tof = sc.array(dims=['tof'], values=[1, 2, 3, np.nan, 5, 6, 7])
        data = sc.array(dims=['tof'], values=[10, 20, 40, 50, 60, 70])
        da = sc.DataArray(data=data, coords={'tof': tof})

        result = resample.rebin_strictly_increasing(da, 'tof')

        # Should have two increasing sections: [1,2,3] and [5,6,7]
        expected_tof = sc.array(dims=['tof'], values=[1.0, 2, 3, 4, 5, 6, 7])
        assert_identical(result.coords['tof'], expected_tof)

        # Data should be correctly rebinned, excluding the NaN point
        expected_data = sc.array(dims=['tof'], values=[10.0, 20, 0, 0, 60, 70])
        assert_identical(result.data, expected_data)

    def test_with_no_increasing_sections_raises_error(self):
        tof = sc.array(dims=['tof'], values=[5, 4, 3, 2, 1, 0])
        data = sc.array(dims=['tof'], values=[10, 20, 30, 40, 50])
        da = sc.DataArray(data=data, coords={'tof': tof})

        with pytest.raises(ValueError, match="No strictly increasing sections found."):
            resample.rebin_strictly_increasing(da, 'tof')

    def test_with_variances(self):
        tof = sc.array(dims=['tof'], values=[1, 2, 3, 2, 3, 4])
        values = [
            10.0,
            20.0,
            15.0,
            25.0,
            35.0,
        ]  # Using float for values to match variances
        variances = [1.0, 2.0, 1.5, 2.5, 3.5]  # Float variances
        data = sc.array(dims=['tof'], values=values, variances=variances)
        da = sc.DataArray(data=data, coords={'tof': tof})

        result = resample.rebin_strictly_increasing(da, 'tof')

        # Check that variances are properly propagated
        assert result.data.variances is not None
        expected_variances = sc.array(dims=['tof'], values=[1.0, 2.0 + 2.5, 3.5])
        assert_identical(sc.variances(result.data), expected_variances)

    def test_additional_coords_are_dropped(self):
        tof = sc.array(dims=['tof'], values=[1, 2, 3, 2, 3, 4, 5])
        data = sc.array(dims=['tof'], values=[10, 20, 15, 25, 35, 45])
        energy = sc.array(dims=['tof'], values=[1.1, 1.2, 1.3, 1.2, 1.4, 1.2])
        da = sc.DataArray(data=data, coords={'tof': tof, 'energy': energy})

        result = resample.rebin_strictly_increasing(da, 'tof')

        # Rebin cannot preserve coords
        assert 'energy' not in result.coords

    def test_masks_are_applied(self):
        tof = sc.array(dims=['tof'], values=[1, 2, 3, 2, 3, 4, 5])
        data = sc.array(dims=['tof'], values=[10, 20, 15, 25, 35, 45])
        da = sc.DataArray(data=data, coords={'tof': tof})

        baseline = resample.rebin_strictly_increasing(da, 'tof')

        # Add a mask
        mask = sc.array(dims=['tof'], values=[False, False, True, False, False, True])
        da.masks['quality'] = mask

        result = resample.rebin_strictly_increasing(da, 'tof')
        # Rebin applies masks
        assert 'quality' not in baseline.masks
        assert result.sum().value < baseline.sum().value

    def test_uses_coord_name_as_new_dimension_name(self):
        """Test with a dimension name different from the coordinate name."""
        # Create data array with dimension name 'd' but coordinate named 'tof'
        tof = sc.array(dims=['d'], values=[1, 2, 3, 2, 3, 4])
        data = sc.array(dims=['d'], values=[10, 20, 15, 25, 35])
        da = sc.DataArray(data=data, coords={'tof': tof})

        # The dimension name 'd' is different from the coordinate name 'tof'
        assert da.coords['tof'].dims[0] == 'd'
        assert 'tof' not in da.dims

        result = resample.rebin_strictly_increasing(da, 'tof')

        # Check that the dimension has been renamed to match the coordinate
        assert 'tof' in result.dims
        assert result.coords['tof'].dims[0] == 'tof'

        # Check that rebinning worked correctly
        expected_tof = sc.array(dims=['tof'], values=[1.0, 2.0, 3.0, 4.0])
        assert sc.identical(result.coords['tof'], expected_tof)

        # Check the data values are properly rebinned and combined
        expected_data = sc.array(dims=['tof'], values=[10.0, 20.0 + 25.0, 35.0])
        assert_identical(result.data, expected_data)
