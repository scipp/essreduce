# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc
import numpy as np

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
