# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
import scipp as sc

from ess.reduce.live import roi


@pytest.fixture
def binned_indices() -> sc.DataArray:
    table = sc.data.table_xyz(123)
    table.coords['index'] = (
        10 * table.coords['x'] + 10 * table.coords['y'] + 10 * table.coords['z']
    ).to(dtype='int32')
    binned = table.bin(x=4, y=5)
    return sc.DataArray(binned.bins.coords['index'], coords=binned.coords)


def test_select_indices_positional_indexing(binned_indices):
    selected = roi.select_indices_in_intervals(
        intervals=sc.DataGroup(x=(1, 3), y=(2, 4)), indices=binned_indices
    )
    assert selected.dim == 'index'
    assert selected.sizes[selected.dim] > 0
    assert selected.sizes[selected.dim] < binned_indices.bins.size().sum().value


def test_select_indices_label_based_indexing(binned_indices):
    selected = roi.select_indices_in_intervals(
        intervals=sc.DataGroup(x=(sc.scalar(0.3, unit='m'), sc.scalar(0.5, unit='m'))),
        indices=binned_indices,
    )
    assert selected.dim == 'index'
    assert selected.sizes[selected.dim] > 0
    assert selected.sizes[selected.dim] < binned_indices.bins.size().sum().value


def test_select_indices_label_based_indexing_reverse_order(binned_indices):
    selected = roi.select_indices_in_intervals(
        intervals=sc.DataGroup(x=(sc.scalar(0.5, unit='m'), sc.scalar(0.3, unit='m'))),
        indices=binned_indices,
    )
    assert selected.dim == 'index'
    assert selected.sizes[selected.dim] > 0
    assert selected.sizes[selected.dim] < binned_indices.bins.size().sum().value


def test_select_indices_fails_with_invalid_dimension():
    data = sc.data.table_xyz(10)
    with pytest.raises(sc.DimensionError):
        roi.select_indices_in_intervals(
            intervals=sc.DataGroup(invalid_dim=(1, 2)), indices=data
        )


def test_select_indices_fails_without_required_coords():
    data = sc.DataArray(sc.array(dims=['x'], values=[1, 2, 3]))
    with pytest.raises(sc.DimensionError, match='no coordinate for that dimension'):
        roi.select_indices_in_intervals(
            intervals=sc.DataGroup(
                x=(sc.scalar(1.0, unit='m'), sc.scalar(2.0, unit='m'))
            ),
            indices=data,
        )


def test_select_indices_works_with_empty_selection(binned_indices):
    selected = roi.select_indices_in_intervals(
        intervals=sc.DataGroup(x=(1, 1)), indices=binned_indices
    )
    assert selected.dim == 'index'
    assert selected.sizes[selected.dim] == 0


def test_apply_selection_empty_yields_empty_result():
    selection = sc.array(dims=['index'], values=[], unit=None, dtype='int32')
    data = sc.arange('detector_number', 12, dtype='int32')
    result, _ = roi.apply_selection(data, selection=selection)
    assert sc.identical(
        result, sc.array(dims=['detector_number'], values=[], dtype='int32')
    )


def test_apply_selection_single_value():
    selection = sc.array(dims=['index'], values=[3], unit=None, dtype='int32')
    data = sc.arange('detector_number', 5, dtype='int32')
    result, _ = roi.apply_selection(data, selection=selection)
    assert sc.identical(
        result, sc.array(dims=['detector_number'], values=[3], dtype='int32')
    )


def test_apply_selection_repeated_indices():
    selection = sc.array(dims=['index'], values=[1, 1, 2, 1], unit=None, dtype='int32')
    data = sc.arange('detector_number', 5)
    result, scale = roi.apply_selection(data, selection=selection)
    assert sc.identical(result, sc.array(dims=['detector_number'], values=[1, 2]))
    assert sc.identical(scale, sc.array(dims=['detector_number'], values=[3.0, 1]))


def test_apply_selection_with_data_array():
    selection = sc.array(dims=['index'], values=[0, 0, 2], unit=None, dtype='int32')
    data = sc.DataArray(
        data=sc.arange('detector_number', 3, dtype='float64'),
        coords={'pos': sc.arange('detector_number', 3, unit='m', dtype='float64')},
    )
    result, _ = roi.apply_selection(data, selection=selection)
    expected = sc.DataArray(
        data=sc.array(dims=['detector_number'], values=[0.0, 2.0], dtype='float64')
        * sc.array(dims=['detector_number'], values=[2, 1], dtype='float64'),
        coords={'pos': sc.array(dims=['detector_number'], values=[0.0, 2.0], unit='m')},
    )
    assert sc.identical(result, expected)


def test_apply_selection_with_units():
    selection = sc.array(dims=['index'], values=[1, 1, 1], unit=None, dtype='int32')
    data = sc.arange('detector_number', 3, unit='K', dtype='float64')
    result, _ = roi.apply_selection(data, selection=selection)
    assert sc.identical(
        result, sc.array(dims=['detector_number'], values=[1.0], unit='K')
    )


def test_apply_selection_non_contiguous_indices():
    selection = sc.array(dims=['index'], values=[0, 4, 4, 8], unit=None, dtype='int32')
    data = sc.arange('detector_number', 10)
    result, scale = roi.apply_selection(data, selection=selection)
    assert sc.identical(result, sc.array(dims=['detector_number'], values=[0, 4, 8]))
    assert sc.identical(scale, sc.array(dims=['detector_number'], values=[1.0, 2, 1]))


def test_apply_selection_ignores_selection_dim():
    selection = sc.array(dims=['ignored'], values=[1, 2], dtype='int32')
    data = sc.arange('detector_number', 5, dtype='int32')
    result, _ = roi.apply_selection(data, selection=selection)
    assert sc.identical(
        result, sc.array(dims=['detector_number'], values=[1, 2], dtype='int32')
    )


def test_apply_selection_fails_with_out_of_bounds_index():
    selection = sc.array(dims=['index'], values=[10], dtype='int32')
    data = sc.arange('detector_number', 5, dtype='int32')
    with pytest.raises(IndexError):
        roi.apply_selection(data, selection=selection)


def logical_view(da: sc.DataArray) -> sc.DataArray:
    return da.fold(da.dim, sizes={'x': 3, 'y': 4, 'z': 2})['z', 0]


@pytest.fixture
def roi_filter() -> roi.ROIFilter:
    indices = sc.ones(sizes={'detector_number': 24}, dtype='int32', unit=None)
    indices = sc.cumsum(indices, mode='exclusive')
    # indices after logical_view have shape (x, y) but values reference full (x, y, z)
    # detector space, so spatial_dims must be the full detector dims
    return roi.ROIFilter(logical_view(indices), spatial_dims=('x', 'y', 'z'))


def test_ROIFilter_defaults_to_empty_roi(roi_filter: roi.ROIFilter):
    data = sc.linspace('detector_number', 0.0, 1.0, num=24, unit='counts')
    result, _ = roi_filter.apply(data)
    assert sc.identical(
        result, sc.array(dims=['detector_number'], values=[], unit='counts')
    )


def test_ROIFilter_applies_roi(roi_filter: roi.ROIFilter):
    roi_filter.set_roi_from_intervals(sc.DataGroup(x=(1, 3), y=(2, 4)))
    data = sc.linspace('detector_number', 1.0, 24.0, num=24, unit='counts')
    result, _ = roi_filter.apply(data)
    assert sc.identical(
        result,
        sc.array(
            dims=['detector_number'], values=[13.0, 15.0, 21.0, 23.0], unit='counts'
        ),
    )


def test_ROIFilter_applies_roi_to_2d_data(roi_filter: roi.ROIFilter):
    roi_filter.set_roi_from_intervals(sc.DataGroup(x=(1, 3), y=(2, 4)))
    data = sc.linspace('detector_number', 1.0, 24.0, num=24, unit='counts').fold(
        dim='detector_number', sizes={'x': 3, 'y': 4, 'z': 2}
    )
    result, _ = roi_filter.apply(data)
    assert sc.identical(
        result,
        sc.array(
            dims=['detector_number'], values=[13.0, 15.0, 21.0, 23.0], unit='counts'
        ),
    )


@pytest.fixture
def roi_filter_2d() -> roi.ROIFilter:
    """ROI filter with 2D indices matching (x, y) spatial dims."""
    # indices shape (x=3, y=4), values 0..11
    indices = sc.arange('detector_number', 12, dtype='int32', unit=None).fold(
        dim='detector_number', sizes={'x': 3, 'y': 4}
    )
    return roi.ROIFilter(indices)


def test_ROIFilter_applies_roi_to_dense_data_preserves_time(
    roi_filter_2d: roi.ROIFilter,
):
    """Dense data (time, x, y) should preserve time dim, only flatten spatial dims."""
    roi_filter_2d.set_roi_from_intervals(sc.DataGroup(x=(1, 3), y=(2, 4)))
    # Dense data shape (time=5, x=3, y=4)
    data = sc.arange('flat', 60, dtype='float64', unit='counts').fold(
        dim='flat', sizes={'time': 5, 'x': 3, 'y': 4}
    )
    result, _ = roi_filter_2d.apply(data)
    # Should have shape (time=5, detector_number=4) - 4 pixels selected
    assert result.dims == ('time', 'detector_number')
    assert result.sizes == {'time': 5, 'detector_number': 4}
    # Selected indices are: x=1,y=2 -> 6, x=1,y=3 -> 7, x=2,y=2 -> 10, x=2,y=3 -> 11
    # For time=0: values at flat indices 6,7,10,11
    expected_time0 = sc.array(
        dims=['detector_number'], values=[6.0, 7.0, 10.0, 11.0], unit='counts'
    )
    assert sc.identical(result['time', 0], expected_time0)


def test_ROIFilter_dense_data_scale_is_broadcast_compatible(
    roi_filter_2d: roi.ROIFilter,
):
    """Scale factor should work with preserved time dimension."""
    roi_filter_2d.set_roi_from_intervals(sc.DataGroup(x=(0, 2), y=(0, 2)))
    data = sc.ones(sizes={'time': 3, 'x': 3, 'y': 4}, unit='counts')
    result, scale = roi_filter_2d.apply(data)
    # Should be able to multiply result * scale without broadcast issues
    weighted = result * scale
    assert weighted.dims == ('time', 'detector_number')


def test_ROIFilter_dense_data_empty_roi(roi_filter_2d: roi.ROIFilter):
    """Empty ROI should return empty result but preserve time dim."""
    # Default ROI is empty
    data = sc.ones(sizes={'time': 3, 'x': 3, 'y': 4}, unit='counts')
    result, _ = roi_filter_2d.apply(data)
    assert result.dims == ('time', 'detector_number')
    assert result.sizes == {'time': 3, 'detector_number': 0}


def test_ROIFilter_dense_data_full_roi(roi_filter_2d: roi.ROIFilter):
    """Selecting all pixels should return all data."""
    roi_filter_2d.set_roi_from_intervals(sc.DataGroup(x=(0, 3), y=(0, 4)))
    data = sc.arange('flat', 60, dtype='float64', unit='counts').fold(
        dim='flat', sizes={'time': 5, 'x': 3, 'y': 4}
    )
    result, _ = roi_filter_2d.apply(data)
    assert result.sizes == {'time': 5, 'detector_number': 12}


def test_ROIFilter_dense_data_single_pixel(roi_filter_2d: roi.ROIFilter):
    """Selecting a single pixel should work."""
    roi_filter_2d.set_roi_from_intervals(sc.DataGroup(x=(1, 2), y=(2, 3)))
    data = sc.arange('flat', 60, dtype='float64', unit='counts').fold(
        dim='flat', sizes={'time': 5, 'x': 3, 'y': 4}
    )
    result, _ = roi_filter_2d.apply(data)
    assert result.sizes == {'time': 5, 'detector_number': 1}
    # x=1, y=2 -> index 6, so time=0 value is 6.0
    assert result['time', 0].values[0] == 6.0
