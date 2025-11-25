# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc
from scipp.testing import assert_identical

from ess.reduce.live import raw


@pytest.mark.parametrize(
    'sigma',
    [sc.scalar(0.01, unit='m'), sc.scalar(1.0, unit='cm'), sc.scalar(10.0, unit='mm')],
)
def test_gaussian_position_noise_is_sigma_unit_independent(sigma: sc.Variable) -> None:
    sigma_m = sigma.to(unit='m')
    reference = raw.gaussian_position_noise(sigma=sigma_m)
    noise = raw.gaussian_position_noise(sigma=sigma)
    assert sc.identical(noise, reference)


def test_clear_counts_resets_counts_to_zero() -> None:
    detector_number = sc.array(dims=['pixel'], values=[1, 2, 3], unit=None)
    det = raw.Detector(detector_number)
    assert det.data.sum().value == 0
    det.add_counts([1, 2, 3, 2])
    assert det.data.sum().value == 4
    det.clear_counts()
    assert det.data.sum().value == 0


def test_Detector_bincount_drops_out_of_range_ids() -> None:
    detector_number = sc.array(dims=['pixel'], values=[1, 2, 3], unit=None)
    det = raw.Detector(detector_number)
    counts = det.bincount([1, 2, 0, -1, 3, 4])
    assert sc.identical(
        counts.data,
        sc.array(dims=['pixel'], values=[1, 1, 1], unit='counts', dtype='int32'),
    )


def test_Detector_bincount_raises_if_detector_number_not_sorted() -> None:
    detector_number = sc.array(dims=['pixel'], values=[1, 3, 2], unit=None)
    det = raw.Detector(detector_number)
    with pytest.raises(ValueError, match="sorted"):
        det.bincount([1])


def test_Detector_bincount_raises_if_detector_number_not_consecutive() -> None:
    detector_number = sc.array(dims=['pixel'], values=[1, 2, 4], unit=None)
    det = raw.Detector(detector_number)
    with pytest.raises(ValueError, match="consecutive"):
        det.bincount([1])


@pytest.mark.parametrize('window', [1, 2, 33])
def test_RollingDetectorView_max_window_property_returns_configured(
    window: int,
) -> None:
    detector_number = sc.array(dims=['pixel'], values=[1, 2, 3], unit=None)
    det = raw.RollingDetectorView(detector_number=detector_number, window=window)
    assert det.max_window == window


def test_RollingDetectorView_full_window() -> None:
    detector_number = sc.array(dims=['pixel'], values=[1, 2, 3], unit=None)
    det = raw.RollingDetectorView(detector_number=detector_number, window=2)
    rolling = det.get()
    assert rolling.sizes == {'pixel': 3}
    assert rolling.sum().value == 0
    det.add_counts([1, 2, 3, 2])
    assert det.get().sum().value == 4
    det.add_counts([1, 3, 2])
    assert det.get().sum().value == 7
    det.add_counts([1, 2])
    assert det.get().sum().value == 5
    det.add_counts([])
    assert det.get().sum().value == 2


def test_RollingDetectorView_cumulative_returns_everything_since_clear() -> None:
    detector_number = sc.array(dims=['pixel'], values=[1, 2, 3], unit=None)
    det = raw.RollingDetectorView(detector_number=detector_number, window=2)
    expected = sc.zeros_like(det.get())
    assert sc.identical(det.cumulative, expected)
    det.add_counts([1, 2, 3, 2])
    delta1 = det.get(window=1)
    expected += delta1
    assert sc.identical(det.cumulative, expected)
    det.add_counts([1, 3, 2])
    delta2 = det.get(window=1)
    expected += delta2
    assert sc.identical(det.cumulative, expected)
    assert sc.identical(det.cumulative, det.get())  # everything still in window
    det.add_counts([1, 2])
    delta3 = det.get(window=1)
    expected += delta3
    assert sc.identical(det.cumulative, expected)
    assert not sc.identical(det.cumulative, det.get())  # delta1 fell out of window
    det.add_counts([])
    assert sc.identical(det.cumulative, expected)
    # Reset
    det.clear_counts()
    expected = sc.zeros_like(expected)
    assert sc.identical(det.cumulative, expected)
    assert sc.identical(det.cumulative, expected)
    det.add_counts([1, 2, 3, 2])
    delta1 = det.get(window=1)
    expected += delta1
    assert sc.identical(det.cumulative, expected)


def test_RollingDetectorView_add_events_accepts_unsorted_detector_number() -> None:
    detector_number = sc.array(dims=['detector_number'], values=[1, 3, 2], unit=None)
    det = raw.RollingDetectorView(detector_number=detector_number, window=2)
    pixel = sc.array(dims=['event'], values=[1, 2, 3, 2], unit=None)
    events = sc.DataArray(sc.ones_like(pixel), coords={'detector_number': pixel})
    det.add_events(events.group(detector_number))
    assert det.get().sum().value == 4


def test_RollingDetectorView_add_events_accepts_non_consecutive_detector_number() -> (
    None
):
    detector_number = sc.array(dims=['detector_number'], values=[1, 2, 4], unit=None)
    det = raw.RollingDetectorView(detector_number=detector_number, window=2)
    pixel = sc.array(dims=['event'], values=[1, 2, 4, 2], unit=None)
    events = sc.DataArray(sc.ones_like(pixel), coords={'detector_number': pixel})
    det.add_events(events.group(detector_number))
    assert det.get().sum().value == 4


def test_RollingDetectorView_partial_window() -> None:
    detector_number = sc.array(dims=['pixel'], values=[1, 2, 3], unit=None)
    det = raw.RollingDetectorView(detector_number=detector_number, window=3)
    det.add_counts([1, 2, 3, 2])
    assert det.get(window=0).sum().value == 0
    assert det.get(window=1).sum().value == 4
    assert det.get(window=2).sum().value == 4
    assert det.get(window=3).sum().value == 4
    det.add_counts([1, 3, 2])
    assert det.get(window=0).sum().value == 0
    assert det.get(window=1).sum().value == 3
    assert det.get(window=2).sum().value == 7
    assert det.get(window=3).sum().value == 7
    det.add_counts([1, 2])
    assert det.get(window=0).sum().value == 0
    assert det.get(window=1).sum().value == 2
    assert det.get(window=2).sum().value == 5
    assert det.get(window=3).sum().value == 9
    det.add_counts([])
    assert det.get(window=0).sum().value == 0
    assert det.get(window=1).sum().value == 0
    assert det.get(window=2).sum().value == 2
    assert det.get(window=3).sum().value == 5
    det.add_counts([1, 2])
    assert det.get(window=0).sum().value == 0
    assert det.get(window=1).sum().value == 2
    assert det.get(window=2).sum().value == 2
    assert det.get(window=3).sum().value == 4


def test_RollingDetectorView_clear_counts() -> None:
    detector_number = sc.array(dims=['pixel'], values=[1, 2, 3], unit=None)
    det = raw.RollingDetectorView(detector_number=detector_number, window=3)
    det.add_counts([1, 2, 3, 2])
    assert det.get(window=0).sum().value == 0
    assert det.get(window=1).sum().value == 4
    assert det.get(window=2).sum().value == 4
    assert det.get(window=3).sum().value == 4
    assert det.get().sum().value == 4
    det.clear_counts()
    assert det.get(window=0).sum().value == 0
    assert det.get(window=1).sum().value == 0
    assert det.get(window=2).sum().value == 0
    assert det.get(window=3).sum().value == 0
    assert det.get().sum().value == 0


def test_RollingDetectorView_raises_if_subwindow_exceeds_window() -> None:
    detector_number = sc.array(dims=['pixel'], values=[1, 2, 3], unit=None)
    det = raw.RollingDetectorView(detector_number=detector_number, window=3)
    with pytest.raises(ValueError, match="Window size"):
        det.get(window=4)
    with pytest.raises(ValueError, match="Window size"):
        det.get(window=-1)


def test_RollingDetectorView_projects_counts() -> None:
    detector_number = sc.array(dims=['pixel'], values=[1, 2, 3], unit=None)
    # Dummy projection that just drops the first pixel
    det = raw.RollingDetectorView(
        detector_number=detector_number,
        window=3,
        projection=lambda da: da['pixel', 1:].rename(pixel='abc'),
    )
    expected = sc.DataArray(
        sc.array(dims=['pixel'], values=[0, 0], unit='counts', dtype='int32'),
        coords={'detector_number': detector_number[1:]},
    ).rename(pixel='abc')

    det.add_counts([1, 2, 3, 2])
    expected.values = [2, 1]
    assert sc.identical(det.get(), expected)

    det.add_counts([1, 3, 3, 1])
    expected.values = [2, 3]
    assert sc.identical(det.get(), expected)


def test_project_xy_with_given_zplane_scales() -> None:
    result = raw.project_xy(
        sc.vectors(
            dims=['point'],
            values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [6.0, 10.0, 12.0]],
            unit='m',
        ),
        zplane=sc.scalar(6.0, unit='m'),
    )
    assert sc.identical(
        result,
        sc.DataGroup(
            x=sc.array(dims=['point'], values=[2.0, 4.0, 3.0], unit='m'),
            y=sc.array(dims=['point'], values=[4.0, 5.0, 5.0], unit='m'),
            z=sc.scalar(6.0, unit='m'),
        ),
    )


def test_project_xy_defaults_to_scale_to_zmin() -> None:
    result = raw.project_xy(
        sc.vectors(
            dims=['point'],
            values=[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [6.0, 10.0, 12.0]],
            unit='m',
        )
    )
    assert sc.identical(
        result,
        sc.DataGroup(
            # Note same relative values as with zplane=6, just scaled.
            x=sc.array(dims=['point'], values=[1.0, 2.0, 1.5], unit='m'),
            y=sc.array(dims=['point'], values=[2.0, 2.5, 2.5], unit='m'),
            z=sc.scalar(3.0, unit='m'),
        ),
    )


def test_project_onto_cylinder_z() -> None:
    radius = sc.scalar(2.0, unit='m')
    # Input radii are 4 and 1 => scale by 1/2 and 2.
    result = raw.project_onto_cylinder_z(
        sc.vectors(dims=['point'], values=[[0.0, 4.0, 3.0], [1.0, 0.0, 6.0]], unit='m'),
        radius=radius,
    )
    assert sc.identical(result['r'], radius)
    assert sc.identical(
        result['z'], sc.array(dims=['point'], values=[1.5, 12.0], unit='m')
    )
    assert sc.identical(
        result['phi'], sc.array(dims=['point'], values=[90.0, 0.0], unit='deg')
    )
    assert sc.identical(
        result['arc_length'],
        sc.array(dims=['point'], values=[radius.value * np.pi * 0.5, 0.0], unit='m'),
    )


def make_grid_cube(
    nx: int = 5,
    ny: int = 5,
    nz: int = 5,
    center: tuple = (0.0, 0.0, 10.0),
    size: float = 1.0,
) -> sc.Variable:
    """Create a grid of points in a cube centered at specified position.

    Parameters
    ----------
    nx:
        Number of points along x-axis, by default 5
    ny:
        Number of points along y-axis, by default 5
    nz:
        Number of points along z-axis, by default 5
    center:
        (x, y, z) coordinates of cube center, by default (0.0, 0.0, 10.0)
    size:
        Side length of cube in meters, by default 1.0

    Returns
    -------
    :
        Scipp variable containing grid points with shape (nx * ny * nz, 3)

    Examples
    --------
    >>> grid = make_grid_cube(nx=3, ny=3, nz=3)
    >>> grid.shape
    (27, 3)
    """
    # Create coordinate arrays
    x = np.linspace(-size / 2, size / 2, nx) + center[0]
    y = np.linspace(-size / 2, size / 2, ny) + center[1]
    z = np.linspace(-size / 2, size / 2, nz) + center[2]

    # Create meshgrid
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')

    # Stack into points array
    points = np.stack([X.flatten(), Y.flatten(), Z.flatten()]).T

    return sc.vectors(dims=['point'], values=points, unit='m')


def test_histogrammer_input_indices() -> None:
    nx, ny, nz = 3, 3, 3
    coords = raw.project_xy(
        make_grid_cube(nx=nx, ny=ny, nz=nz, center=(0.0, 3.0, 10.0))
    )
    coords = sc.concat([coords], 'replica')

    resolution = {'x': 4, 'y': 5}
    histogrammer = raw.Histogrammer.from_coords(coords=coords, resolution=resolution)
    indices = histogrammer.input_indices()
    assert set(indices.coords) == {'x', 'y'}
    assert indices.bins.size().sum().value == nx * ny * nz
    assert indices.sizes == resolution


def test_ROIFilter_from_trivial_RollingDetectorView() -> None:
    detector_number = sc.array(
        dims=['x', 'y'], values=[[1, 2, 3], [4, 5, 6]], unit=None
    )
    view = raw.RollingDetectorView(detector_number=detector_number, window=2)
    roi_filter = view.make_roi_filter()
    data = detector_number.copy()
    data.unit = 'counts'
    flat = data.flatten(to='detector_number')

    result, scale = roi_filter.apply(data)
    # ROIFilter defaults to include nothing
    assert sc.identical(result, flat[0:0])
    assert sc.identical(scale, sc.zeros(dims=['detector_number'], shape=[0]))

    roi_filter.set_roi_from_intervals(sc.DataGroup(x=(1, 2)))
    result, scale = roi_filter.apply(data)
    assert sc.identical(result, flat[3:6])
    assert sc.identical(scale, sc.ones(dims=['detector_number'], shape=[3]))

    roi_filter.set_roi_from_intervals(sc.DataGroup(x=(1, 2), y=(1, 3)))
    result, scale = roi_filter.apply(data)
    assert sc.identical(result, flat[4:6])
    assert sc.identical(scale, sc.ones(dims=['detector_number'], shape=[2]))


def test_ROIFilter_from_RollingDetectorView_with_custom_projection() -> None:
    detector_number = sc.array(
        dims=['x', 'y', 'z'], values=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], unit=None
    )
    view = raw.RollingDetectorView(
        detector_number=detector_number, window=2, projection=lambda da: da['z', 0]
    )
    roi_filter = view.make_roi_filter()
    data = detector_number.copy()
    data.unit = 'counts'
    flat = data['z', 0].flatten(to='detector_number')

    result, scale = roi_filter.apply(data)
    # ROIFilter defaults to include nothing
    assert sc.identical(result, flat[0:0])
    assert sc.identical(scale, sc.zeros(dims=['detector_number'], shape=[0]))

    roi_filter.set_roi_from_intervals(sc.DataGroup(x=(1, 2)))
    result, scale = roi_filter.apply(data)
    assert sc.identical(result, flat[2:4])
    assert sc.identical(scale, sc.ones(dims=['detector_number'], shape=[2]))

    roi_filter.set_roi_from_intervals(sc.DataGroup(x=(1, 2), y=(1, 3)))
    result, scale = roi_filter.apply(data)
    assert sc.identical(result, flat[3:4])
    assert sc.identical(scale, sc.ones(dims=['detector_number'], shape=[1]))


def test_ROIFilter_from_RollingDetectorView_with_xy_projection() -> None:
    detector_number = sc.array(
        dims=['x', 'y', 'z'], values=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], unit=None
    )
    nx, ny, nz = 2, 2, 2
    coords = raw.project_xy(
        make_grid_cube(nx=nx, ny=ny, nz=nz, center=(0.0, 0.0, 10.0))
    )
    coords = sc.concat(
        [coords.fold(dim='point', sizes=detector_number.sizes)], 'replica'
    )

    resolution = {'x': 4, 'y': 4}
    histogrammer = raw.Histogrammer.from_coords(coords=coords, resolution=resolution)
    view = raw.RollingDetectorView(
        detector_number=detector_number, window=1, projection=histogrammer
    )
    roi_filter = view.make_roi_filter()
    data = detector_number.copy()
    data.unit = 'counts'
    flat = data.flatten(to='detector_number')

    result, scale = roi_filter.apply(data)
    # ROIFilter defaults to include nothing
    assert sc.identical(result, flat[0:0])
    assert sc.identical(scale, sc.zeros(dims=['detector_number'], shape=[0]))

    roi_filter.set_roi_from_intervals(sc.DataGroup(x=(0, 2)))
    result, scale = roi_filter.apply(data)
    assert sc.identical(scale, sc.ones(dims=['detector_number'], shape=[4]))
    assert sc.identical(result, flat[:4])

    roi_filter.set_roi_from_intervals(sc.DataGroup(x=(0, 2), y=(0, 2)))
    result, scale = roi_filter.apply(data)
    assert sc.identical(scale, sc.ones(dims=['detector_number'], shape=[2]))
    assert sc.identical(result, flat[0:2])


def test_Histogrammer_given_empty_bins_returns_nans_in_transform_weights() -> None:
    x = sc.array(dims=['replica', 'pixel'], values=[[0.0, 1.0]], unit='m')
    coords = sc.DataGroup(x=x, y=x, z=sc.zeros_like(x))
    resolution = {'y': 1, 'x': 4}
    view = raw.RollingDetectorView(
        detector_number=sc.array(dims=['pixel'], values=[1, 2], unit=None),
        window=1,
        projection=raw.Histogrammer.from_coords(coords=coords, resolution=resolution),
    )
    weights = view.transform_weights()
    assert_identical(
        weights.data,
        sc.array(dims=['y', 'x'], values=[[1.0, np.nan, np.nan, 1.0]], dtype='float32'),
    )


def test_Histogrammer_threshold_based_on_non_empty_bins_in_transform_weights() -> None:
    x = sc.array(dims=['replica', 'pixel'], values=[[0.0, 0.0, 1.0]], unit='m')
    coords = sc.DataGroup(x=x, y=x, z=sc.zeros_like(x))
    resolution = {'y': 1, 'x': 4}
    view = raw.RollingDetectorView(
        detector_number=sc.array(dims=['pixel'], values=[1, 2, 3], unit=None),
        window=1,
        projection=raw.Histogrammer.from_coords(coords=coords, resolution=resolution),
    )
    weights = view.transform_weights(threshold=0.9)
    assert_identical(
        weights.data,
        sc.array(
            dims=['y', 'x'], values=[[2.0, np.nan, np.nan, np.nan]], dtype='float32'
        ),
    )


def test_Histogrammer_replicas_do_not_affect_transformed_weights() -> None:
    resolution = {'y': 1, 'x': 4}
    x = sc.array(dims=['replica', 'pixel'], values=[[0.0, 0.0, 0.5, 1.0]], unit='m')
    coords = sc.DataGroup(x=x, y=x, z=sc.zeros_like(x))
    detector_number = sc.array(dims=['pixel'], values=[1, 2, 3, 4], unit=None)
    view = raw.RollingDetectorView(
        detector_number=detector_number,
        window=1,
        projection=raw.Histogrammer.from_coords(coords=coords, resolution=resolution),
    )
    reference = view.transform_weights()

    x = sc.concat([x, x], 'replica')
    coords = sc.DataGroup(x=x, y=x, z=sc.zeros_like(x))
    view = raw.RollingDetectorView(
        detector_number=detector_number,
        window=1,
        projection=raw.Histogrammer.from_coords(coords=coords, resolution=resolution),
    )
    result = view.transform_weights()

    assert_identical(reference, result)


def test_RollingDetectorView_normalization() -> None:
    resolution = {'y': 1, 'x': 4}
    x = sc.array(dims=['replica', 'pixel'], values=[[0.0, 0.0, 0.3, 1.0]], unit='m')
    coords = sc.DataGroup(x=x, y=x, z=sc.zeros_like(x))
    histogrammer = raw.Histogrammer.from_coords(coords=coords, resolution=resolution)
    detector_number = sc.array(dims=['pixel'], values=[1, 2, 3, 4], unit=None)
    view = raw.RollingDetectorView(
        detector_number=detector_number, window=1, projection=histogrammer
    )
    weights = view.transform_weights().data
    expected = sc.array(
        dims=['y', 'x'],
        values=[[0.0, 0.0, np.nan, 0.0]],
        unit='counts',
        dtype='float32',
    )
    assert_identical(view.get().data / weights, expected)
    assert_identical(view.get().data / weights, expected)
    expected.values = [[0.0, 0.0, 0.0, 0.0]]
    assert_identical(view.get().data, expected.to(dtype='int32'))

    # detector_number 1 and 2 are in the same bin, so their counts are summed,
    view.add_counts([1, 2, 3, 4])
    expected.values = [[1.0, 1.0, np.nan, 1.0]]
    assert_identical(view.get().data / weights, expected)
    assert_identical(view.get().data / weights, expected)
    expected.values = [[2.0, 1.0, 0.0, 1.0]]
    assert_identical(view.get().data, expected.to(dtype='int32'))

    view.add_counts([2, 4])
    expected.values = [[0.5, 0.0, np.nan, 1.0]]
    assert_identical(view.get().data / weights, expected)
    assert_identical(view.get().data / weights, expected)
    expected.values = [[1.0, 0.0, 0.0, 1.0]]
    assert_identical(view.get().data, expected.to(dtype='int32'))


def test_transform_weights_raises_given_bad_sizes() -> None:
    resolution = {'y': 1, 'x': 4}
    x = sc.array(dims=['replica', 'pixel'], values=[[0.0, 0.0, 0.3, 1.0]], unit='m')
    coords = sc.DataGroup(x=x, y=x, z=sc.zeros_like(x))
    histogrammer = raw.Histogrammer.from_coords(coords=coords, resolution=resolution)
    detector_number = sc.array(dims=['pixel'], values=[1, 2, 3, 4], unit=None)
    view = raw.RollingDetectorView(
        detector_number=detector_number, window=1, projection=histogrammer
    )
    with pytest.raises(sc.DimensionError):
        view.transform_weights(sc.ones(dims=['x'], shape=[4]))
    with pytest.raises(sc.DimensionError):
        view.transform_weights(sc.ones(dims=['pixel'], shape=[3]))


def test_transform_weights_raises_given_DataArray_with_bad_det_num() -> None:
    resolution = {'y': 1, 'x': 4}
    x = sc.array(dims=['replica', 'pixel'], values=[[0.0, 0.0, 1.0]], unit='m')
    coords = sc.DataGroup(x=x, y=x, z=sc.zeros_like(x))
    histogrammer = raw.Histogrammer.from_coords(coords=coords, resolution=resolution)
    detector_number = sc.array(dims=['pixel'], values=[1, 2, 3], unit=None)
    view = raw.RollingDetectorView(
        detector_number=detector_number, window=1, projection=histogrammer
    )
    weights = sc.DataArray(
        data=sc.ones(dims=['pixel'], shape=[3]),
        coords={
            'detector_number': sc.array(dims=['pixel'], values=[1, 2, 4], unit=None)
        },
    )
    with pytest.raises(sc.CoordError):
        view.transform_weights(weights)


class TestLogicalView:
    """Tests for LogicalView class."""

    def test_single_dim_downsampling(self) -> None:
        """Test basic 1D downsampling with transform + reduction."""

        # Transform: fold into 4 groups of 2
        def transform(da: sc.DataArray) -> sc.DataArray:
            return da.fold(
                dim='x_pixel_offset', sizes={'x_pixel_offset': 4, 'x_bin': 2}
            )

        view = raw.LogicalView(
            transform=transform,
            reduction_dim='x_bin',
        )

        # Create test data: each pixel has value equal to its index
        data = sc.DataArray(
            data=sc.arange('x_pixel_offset', 8, dtype='float64', unit='counts')
        )

        # Apply downsampling
        result = view(data)

        # Should sum pairs: [0+1, 2+3, 4+5, 6+7] = [1, 5, 9, 13]
        expected = sc.array(
            dims=['x_pixel_offset'],
            values=[1.0, 5.0, 9.0, 13.0],
            unit='counts',
        )
        assert sc.allclose(result.data, expected)
        assert result.sizes == {'x_pixel_offset': 4}

    def test_multi_dim_downsampling(self) -> None:
        """Test 2D downsampling similar to _resize_image example."""

        # Transform: fold both dimensions for 8x8 -> 4x4 (2x2 binning in each dimension)
        def transform(da: sc.DataArray) -> sc.DataArray:
            da = da.fold(dim='x_pixel_offset', sizes={'x_pixel_offset': 4, 'x_bin': 2})
            da = da.fold(dim='y_pixel_offset', sizes={'y_pixel_offset': 4, 'y_bin': 2})
            return da

        view = raw.LogicalView(
            transform=transform,
            reduction_dim=['x_bin', 'y_bin'],
        )

        # Create test data: constant value of 1 everywhere
        data = sc.DataArray(
            data=sc.ones(
                sizes={'x_pixel_offset': 8, 'y_pixel_offset': 8}, unit='counts'
            )
        )

        # Apply downsampling
        result = view(data)

        # Each output pixel should be sum of 2x2=4 input pixels
        expected = sc.full(
            dims=['x_pixel_offset', 'y_pixel_offset'],
            shape=[4, 4],
            value=4.0,
            unit='counts',
        )
        assert sc.allclose(result.data, expected)
        assert result.sizes == {'x_pixel_offset': 4, 'y_pixel_offset': 4}

    def test_input_indices_single_dim(self) -> None:
        """Test that input_indices creates correct binned mapping for 1D."""

        def transform(da: sc.DataArray) -> sc.DataArray:
            return da.fold(
                dim='x_pixel_offset', sizes={'x_pixel_offset': 4, 'x_bin': 2}
            )

        view = raw.LogicalView(
            transform=transform,
            reduction_dim='x_bin',
            input_sizes={'x_pixel_offset': 8},
        )

        # Get index mapping
        indices = view.input_indices()

        # Should be binned data with 4 bins, each containing 2 indices
        assert indices.sizes == {'x_pixel_offset': 4}
        assert indices.bins is not None

        # Check each bin contains the correct indices
        # Bin 0: [0, 1], Bin 1: [2, 3], Bin 2: [4, 5], Bin 3: [6, 7]
        # Extract all bin contents using the bins accessor
        bin_sizes = indices.bins.size()
        assert all(bin_sizes.values == 2)  # Each bin should have 2 indices

        # Check total count
        assert indices.bins.size().sum().value == 8

    def test_input_indices_multi_dim(self) -> None:
        """Test that input_indices creates correct binned mapping for 2D."""

        def transform(da: sc.DataArray) -> sc.DataArray:
            da = da.fold(dim='x_pixel_offset', sizes={'x_pixel_offset': 2, 'x_bin': 2})
            da = da.fold(dim='y_pixel_offset', sizes={'y_pixel_offset': 2, 'y_bin': 2})
            return da

        view = raw.LogicalView(
            transform=transform,
            reduction_dim=['x_bin', 'y_bin'],
            input_sizes={'x_pixel_offset': 4, 'y_pixel_offset': 4},
        )

        # Get index mapping
        indices = view.input_indices()

        # Should be binned data with 2x2 output bins
        assert indices.sizes == {'x_pixel_offset': 2, 'y_pixel_offset': 2}
        assert indices.bins is not None

        # Each bin should contain 2x2=4 indices from the flattened input
        bin_sizes = indices.bins.size()
        assert all(bin_sizes.values.ravel() == 4)  # Each bin should have 4 indices

        # Check total count: 4x4 input pixels -> 2x2 output bins
        assert indices.bins.size().sum().value == 16

    def test_with_varying_input_values(self) -> None:
        """Test that downsampling correctly sums varying input values."""

        def transform(da: sc.DataArray) -> sc.DataArray:
            return da.fold(
                dim='x_pixel_offset', sizes={'x_pixel_offset': 3, 'x_bin': 2}
            )

        view = raw.LogicalView(
            transform=transform,
            reduction_dim='x_bin',
        )

        # Create test data with specific values: [10, 20, 30, 40, 50, 60]
        data = sc.DataArray(
            data=sc.array(
                dims=['x_pixel_offset'],
                values=[10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
                unit='counts',
            )
        )

        result = view(data)

        # Should sum pairs: [10+20, 30+40, 50+60] = [30, 70, 110]
        expected = sc.array(
            dims=['x_pixel_offset'],
            values=[30.0, 70.0, 110.0],
            unit='counts',
        )
        assert sc.allclose(result.data, expected)

    def test_transform_without_reduction_slicing(self) -> None:
        """Test transform without reduction (slicing to select front layer)."""

        # Transform: fold to 3D volume, then slice front layer
        def transform(da: sc.DataArray) -> sc.DataArray:
            return da.fold(dim='voxel', sizes={'x': 2, 'y': 2, 'z': 3})['z', 0]

        view = raw.LogicalView(transform=transform)

        # Create test data: each voxel has value equal to its index
        data = sc.DataArray(data=sc.arange('voxel', 12, dtype='float64', unit='counts'))

        result = view(data)

        # fold orders: z is innermost, so z=0 gives every 3rd element starting at 0
        # indices: 0, 3, 6, 9
        assert result.sizes == {'x': 2, 'y': 2}
        expected = sc.array(
            dims=['x', 'y'],
            values=[[0.0, 3.0], [6.0, 9.0]],
            unit='counts',
        )
        assert sc.allclose(result.data, expected)

    def test_transform_without_reduction_reshape(self) -> None:
        """Test transform without reduction (pure reshape)."""

        # Transform: just fold without any reduction
        def transform(da: sc.DataArray) -> sc.DataArray:
            return da.fold(dim='pixel', sizes={'x': 3, 'y': 4})

        view = raw.LogicalView(transform=transform)

        data = sc.DataArray(data=sc.arange('pixel', 12, dtype='float64', unit='counts'))

        result = view(data)

        # Should just reshape, no reduction
        assert result.sizes == {'x': 3, 'y': 4}
        assert result.data.sum().value == 66.0  # Sum of 0..11

    def test_input_indices_without_reduction(self) -> None:
        """Test that input_indices returns dense indices when no reduction."""

        def transform(da: sc.DataArray) -> sc.DataArray:
            return da.fold(dim='voxel', sizes={'x': 2, 'y': 2, 'z': 3})['z', 0]

        view = raw.LogicalView(
            transform=transform,
            input_sizes={'voxel': 12},
        )

        indices = view.input_indices()

        # Should be dense (not binned) - 1:1 mapping
        assert indices.bins is None
        assert indices.sizes == {'x': 2, 'y': 2}

        # Indices should correspond to front layer of folded volume
        # fold orders: z is innermost, so z=0 gives every 3rd index starting at 0
        expected = sc.array(dims=['x', 'y'], values=[[0, 3], [6, 9]], unit=None)
        assert sc.identical(indices.data, expected)

    def test_input_indices_without_reduction_preserves_total_count(self) -> None:
        """Test that non-reducing input_indices has correct number of indices."""

        def transform(da: sc.DataArray) -> sc.DataArray:
            return da.fold(dim='pixel', sizes={'x': 4, 'y': 5})

        view = raw.LogicalView(
            transform=transform,
            input_sizes={'pixel': 20},
        )

        indices = view.input_indices()

        # Dense indices should have same total size as output shape
        assert indices.sizes == {'x': 4, 'y': 5}
        assert indices.data.size == 20


class TestRollingDetectorViewWithLogicalView:
    """Tests for RollingDetectorView integration with LogicalView."""

    def test_as_projection(self) -> None:
        """Test that RollingDetectorView works with LogicalView as projection."""
        # Create a 1D detector with 8 pixels
        detector_number = sc.arange('x_pixel_offset', 1, 9, unit=None)

        # Define downsampling transform: 8 -> 4 pixels (2x binning)
        def transform(da: sc.DataArray) -> sc.DataArray:
            return da.fold(
                dim='x_pixel_offset', sizes={'x_pixel_offset': 4, 'x_bin': 2}
            )

        # Create RollingDetectorView with LogicalView using factory method
        view = raw.RollingDetectorView.with_logical_view(
            detector_number=detector_number,
            window=2,
            transform=transform,
            reduction_dim='x_bin',
        )

        # Add some counts: pixels 1, 2, 3, 4 -> downsampled bins [0, 1]
        view.add_counts([1, 2, 3, 4])
        result = view.get()

        # After downsampling: [1+2, 3+4] = [2, 2] for first two bins
        assert result.sizes == {'x_pixel_offset': 4}
        assert result['x_pixel_offset', 0].value == 2
        assert result['x_pixel_offset', 1].value == 2
        assert result['x_pixel_offset', 2].value == 0
        assert result['x_pixel_offset', 3].value == 0

    def test_make_roi_filter(self) -> None:
        """Test that make_roi_filter() works with LogicalView."""
        detector_number = sc.arange('x_pixel_offset', 1, 9, unit=None)

        def transform(da: sc.DataArray) -> sc.DataArray:
            return da.fold(
                dim='x_pixel_offset', sizes={'x_pixel_offset': 4, 'x_bin': 2}
            )

        view = raw.RollingDetectorView.with_logical_view(
            detector_number=detector_number,
            window=1,
            transform=transform,
            reduction_dim='x_bin',
        )

        # Should not raise - LogicalView has input_indices()
        roi_filter = view.make_roi_filter()

        # The indices should be binned data (check via private attribute for now)
        assert roi_filter._indices.bins is not None
        assert roi_filter._indices.sizes == {'x_pixel_offset': 4}
        # Each bin should contain 2 indices
        assert all(roi_filter._indices.bins.size().values == 2)

    def test_transform_weights(self) -> None:
        """Test that transform_weights() works with LogicalView."""
        detector_number = sc.arange('x_pixel_offset', 1, 9, unit=None)

        def transform(da: sc.DataArray) -> sc.DataArray:
            return da.fold(
                dim='x_pixel_offset', sizes={'x_pixel_offset': 4, 'x_bin': 2}
            )

        view = raw.RollingDetectorView.with_logical_view(
            detector_number=detector_number,
            window=1,
            transform=transform,
            reduction_dim='x_bin',
        )

        # Create weights: all pixels have weight 1.0
        weights = sc.ones(sizes={'x_pixel_offset': 8}, dtype='float32', unit='')

        # Transform weights through the downsampler
        transformed = view.transform_weights(weights)

        # After downsampling: each output bin sums 2 input weights = 2.0
        assert transformed.sizes == {'x_pixel_offset': 4}
        expected = sc.full(
            dims=['x_pixel_offset'], shape=[4], value=2.0, dtype='float32', unit=''
        )
        assert sc.allclose(transformed.data, expected)

    def test_with_non_reducing_view(self) -> None:
        """Test RollingDetectorView with LogicalView without reduction (slicing)."""
        # 12 voxels that will be folded into 2x2x3 and sliced to front layer
        detector_number = sc.arange('voxel', 1, 13, unit=None)

        def transform(da: sc.DataArray) -> sc.DataArray:
            return da.fold(dim='voxel', sizes={'x': 2, 'y': 2, 'z': 3})['z', 0]

        view = raw.RollingDetectorView.with_logical_view(
            detector_number=detector_number,
            window=2,
            transform=transform,
            # No reduction_dim - pure transform
        )

        # Add counts for detector_numbers that map to front layer (z=0)
        # z is innermost, so front layer indices are 0, 3, 6, 9 (every 3rd)
        # detector_numbers are 1-indexed, so front layer det_nums are 1, 4, 7, 10
        view.add_counts([1, 4, 7, 10])
        result = view.get()

        assert result.sizes == {'x': 2, 'y': 2}
        # Each front-layer pixel gets one count
        expected = sc.array(
            dims=['x', 'y'], values=[[1, 1], [1, 1]], dtype='int32', unit='counts'
        )
        assert sc.identical(result.data, expected)

    def test_make_roi_filter_with_non_reducing_view(self) -> None:
        """Test make_roi_filter with non-reducing LogicalView returns dense indices."""
        detector_number = sc.arange('voxel', 1, 13, unit=None)

        def transform(da: sc.DataArray) -> sc.DataArray:
            return da.fold(dim='voxel', sizes={'x': 2, 'y': 2, 'z': 3})['z', 0]

        view = raw.RollingDetectorView.with_logical_view(
            detector_number=detector_number,
            window=1,
            transform=transform,
        )

        roi_filter = view.make_roi_filter()

        # Indices should be dense (not binned) for non-reducing view
        assert roi_filter._indices.bins is None
        assert roi_filter._indices.sizes == {'x': 2, 'y': 2}
