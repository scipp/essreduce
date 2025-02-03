# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import numpy as np
import pytest
import scipp as sc

from ess.reduce.live import raw


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
    assert det.get(0).sum().value == 0
    assert det.get(1).sum().value == 4
    assert det.get(2).sum().value == 4
    assert det.get(3).sum().value == 4
    det.add_counts([1, 3, 2])
    assert det.get(0).sum().value == 0
    assert det.get(1).sum().value == 3
    assert det.get(2).sum().value == 7
    assert det.get(3).sum().value == 7
    det.add_counts([1, 2])
    assert det.get(0).sum().value == 0
    assert det.get(1).sum().value == 2
    assert det.get(2).sum().value == 5
    assert det.get(3).sum().value == 9
    det.add_counts([])
    assert det.get(0).sum().value == 0
    assert det.get(1).sum().value == 0
    assert det.get(2).sum().value == 2
    assert det.get(3).sum().value == 5
    det.add_counts([1, 2])
    assert det.get(0).sum().value == 0
    assert det.get(1).sum().value == 2
    assert det.get(2).sum().value == 2
    assert det.get(3).sum().value == 4


def test_RollingDetectorView_raises_if_subwindow_exceeds_window() -> None:
    detector_number = sc.array(dims=['pixel'], values=[1, 2, 3], unit=None)
    det = raw.RollingDetectorView(detector_number=detector_number, window=3)
    with pytest.raises(ValueError, match="Window size"):
        det.get(4)
    with pytest.raises(ValueError, match="Window size"):
        det.get(-1)


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


def test_ROIFilter_from_RollingDetectorView_with_LogicalView() -> None:
    logical_view = raw.LogicalView(select={'z': 0})
    detector_number = sc.array(
        dims=['x', 'y', 'z'], values=[[[1, 2], [3, 4]], [[5, 6], [7, 8]]], unit=None
    )
    view = raw.RollingDetectorView(
        detector_number=detector_number, window=2, projection=logical_view
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
