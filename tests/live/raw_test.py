# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
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
