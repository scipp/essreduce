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
