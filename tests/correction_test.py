# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc
import scipp.testing

from ess.reduce.correction import (
    normalize_by_monitor_histogram,
    normalize_by_monitor_integrated,
)
from ess.reduce.uncertainty import UncertaintyBroadcastMode


def test_normalize_by_monitor_histogram_aligned_bins_w_event_coord() -> None:
    detector = sc.DataArray(
        sc.array(dims=['w'], values=[0, 10, 30], unit='counts'),
        coords={'w': sc.arange('w', 3.0, unit='Å')},
    ).bin(w=sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å'))
    monitor = sc.DataArray(
        sc.array(dims=['w'], values=[5.0, 6.0], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å')},
    )
    normalized = normalize_by_monitor_histogram(
        detector,
        monitor=monitor,
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )

    expected = sc.DataArray(
        sc.array(dims=['w'], values=[0.0, 44 / 3, 55 / 3], unit='counts'),
        coords={'w': sc.arange('w', 3.0, unit='Å')},
    ).bin(w=sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å'))

    sc.testing.assert_identical(normalized, expected)


def test_normalize_by_monitor_histogram_aligned_bins_wo_event_coord() -> None:
    detector = sc.DataArray(
        sc.array(dims=['w'], values=[0, 10, 30], unit='counts'),
        coords={'w': sc.arange('w', 3.0, unit='Å')},
    ).bin(w=sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å'))
    monitor = sc.DataArray(
        sc.array(dims=['w'], values=[5.0, 6.0], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å')},
    )
    normalized = normalize_by_monitor_histogram(
        detector,
        monitor=monitor,
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )

    expected = sc.DataArray(
        sc.array(dims=['w'], values=[0.0, 44 / 3, 55 / 3], unit='counts'),
        coords={'w': sc.arange('w', 3.0, unit='Å')},
    ).bin(w=sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å'))

    sc.testing.assert_identical(normalized, expected)


def test_normalize_by_monitor_histogram_aligned_bins_hist() -> None:
    detector = sc.DataArray(
        sc.array(dims=['w'], values=[10, 30], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å')},
    )
    monitor = sc.DataArray(
        sc.array(dims=['w'], values=[5.0, 6.0], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å')},
    )
    normalized = normalize_by_monitor_histogram(
        detector,
        monitor=monitor,
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )

    expected = sc.DataArray(
        sc.array(dims=['w'], values=[44 / 3, 55 / 3], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å')},
    )

    sc.testing.assert_identical(normalized, expected)


def test_normalize_by_monitor_histogram_monitor_envelops_detector_bin() -> None:
    detector = sc.DataArray(
        sc.array(dims=['w'], values=[0, 10, 30], unit='counts'),
        coords={'w': sc.arange('w', 3.0, unit='Å')},
    ).bin(w=sc.array(dims=['w'], values=[0.0, 2, 2.5], unit='Å'))
    monitor = sc.DataArray(
        sc.array(dims=['w'], values=[5.0, 6.0], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[-1, 1.5, 3], unit='Å')},
    )
    normalized = normalize_by_monitor_histogram(
        detector,
        monitor=monitor,
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )

    expected = sc.DataArray(
        sc.array(dims=['w'], values=[0.0, 55 / 4, 165 / 8], unit='counts'),
        coords={'w': sc.arange('w', 3.0, unit='Å')},
    ).bin(w=sc.array(dims=['w'], values=[0.0, 2, 2.5], unit='Å'))

    sc.testing.assert_identical(normalized, expected)


def test_normalize_by_monitor_histogram_monitor_envelops_detector_bin_hist() -> None:
    detector = sc.DataArray(
        sc.array(dims=['w'], values=[10, 30], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0.0, 2, 2.5], unit='Å')},
    )
    monitor = sc.DataArray(
        sc.array(dims=['w'], values=[5.0, 6.0], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[-1, 1.5, 3], unit='Å')},
    )
    normalized = normalize_by_monitor_histogram(
        detector,
        monitor=monitor,
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )

    # These values are different from the case with binned data in
    # test_normalize_by_monitor_histogram_monitor_envelops_detector_bin
    # because the monitor gets rebinned to match the detector bins.
    expected = sc.DataArray(
        sc.array(dims=['w'], values=[11.2, 21.0], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0.0, 2, 2.5], unit='Å')},
    )

    sc.testing.assert_identical(normalized, expected)


def test_normalize_by_monitor_histogram_detector_envelops_monitor_bin() -> None:
    detector = sc.DataArray(
        sc.array(dims=['w'], values=[0, 10, 30], unit='counts'),
        coords={'w': sc.arange('w', 3.0, unit='Å')},
    ).bin(w=sc.array(dims=['w'], values=[0.0, 1.5, 3], unit='Å'))
    monitor = sc.DataArray(
        sc.array(dims=['w'], values=[5.0, 6.0], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0, 2, 2.5], unit='Å')},
    )
    with pytest.raises(ValueError, match="smaller than the range of the detector"):
        normalize_by_monitor_histogram(
            detector,
            monitor=monitor,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )


def test_normalize_by_monitor_histogram_detector_envelops_monitor_bin_hist() -> None:
    detector = sc.DataArray(
        sc.array(dims=['w'], values=[10, 30], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0.0, 1.5, 3], unit='Å')},
    )
    monitor = sc.DataArray(
        sc.array(dims=['w'], values=[5.0, 6.0], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0, 2, 2.5], unit='Å')},
    )
    with pytest.raises(ValueError, match="smaller than the range of the detector"):
        normalize_by_monitor_histogram(
            detector,
            monitor=monitor,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )


def test_normalize_by_monitor_histogram_monitor_extra_bins_in_monitor() -> None:
    detector = sc.DataArray(
        sc.array(dims=['w'], values=[0, 10, 30], unit='counts'),
        coords={'w': sc.arange('w', 3.0, unit='Å')},
    ).bin(w=sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å'))
    monitor = sc.DataArray(
        sc.array(dims=['w'], values=[4.0, 5.0, 6.0, 7.0], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[-1.0, 0, 2, 3, 4], unit='Å')},
    )
    normalized = normalize_by_monitor_histogram(
        detector,
        monitor=monitor,
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )

    expected = sc.DataArray(
        sc.array(dims=['w'], values=[0.0, 44 / 3, 55 / 3], unit='counts'),
        coords={'w': sc.arange('w', 3.0, unit='Å')},
    ).bin(w=sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å'))

    sc.testing.assert_identical(normalized, expected)


def test_normalize_by_monitor_histogram_monitor_extra_bins_in_monitor_hist() -> None:
    detector = sc.DataArray(
        sc.array(dims=['w'], values=[10, 30], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å')},
    )
    monitor = sc.DataArray(
        sc.array(dims=['w'], values=[4.0, 5.0, 6.0, 7.0], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[-1.0, 0, 2, 3, 4], unit='Å')},
    )
    normalized = normalize_by_monitor_histogram(
        detector,
        monitor=monitor,
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )

    expected = sc.DataArray(
        sc.array(dims=['w'], values=[44 / 3, 55 / 3], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å')},
    )

    sc.testing.assert_identical(normalized, expected)


def test_normalize_by_monitor_histogram_monitor_extra_bins_in_detector() -> None:
    detector = sc.DataArray(
        sc.array(dims=['w'], values=[-10, 0, 10, 30, 40], unit='counts'),
        coords={'w': sc.arange('w', -1.0, 4.0, unit='Å')},
    ).bin(w=sc.array(dims=['w'], values=[-1.0, 0, 2, 3, 4], unit='Å'))
    monitor = sc.DataArray(
        sc.array(dims=['w'], values=[5.0, 6.0], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0, 2, 3], unit='Å')},
    )
    with pytest.raises(ValueError, match="smaller than the range of the detector"):
        normalize_by_monitor_histogram(
            detector,
            monitor=monitor,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )


def test_normalize_by_monitor_histogram_monitor_finer_bins_in_detector() -> None:
    detector = sc.DataArray(
        sc.array(dims=['w'], values=[0, 10, 30], unit='counts'),
        coords={'w': sc.arange('w', 3.0, unit='Å')},
    ).bin(w=sc.array(dims=['w'], values=[0.0, 1, 2, 3], unit='Å'))
    monitor = sc.DataArray(
        sc.array(dims=['w'], values=[5.0, 6.0], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å')},
    )
    normalized = normalize_by_monitor_histogram(
        detector,
        monitor=monitor,
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )

    expected = sc.DataArray(
        sc.array(dims=['w'], values=[0.0, 44 / 3, 55 / 3], unit='counts'),
        coords={'w': sc.arange('w', 3.0, unit='Å')},
    ).bin(w=sc.array(dims=['w'], values=[0.0, 1, 2, 3], unit='Å'))

    sc.testing.assert_identical(normalized, expected)


def test_normalize_by_monitor_histogram_monitor_finer_bins_in_detector_hist() -> None:
    detector = sc.DataArray(
        sc.array(dims=['w'], values=[0, 10, 30], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0.0, 1, 2, 3], unit='Å')},
    )
    monitor = sc.DataArray(
        sc.array(dims=['w'], values=[5.0, 6.0], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å')},
    )
    normalized = normalize_by_monitor_histogram(
        detector,
        monitor=monitor,
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )

    expected = sc.DataArray(
        sc.array(dims=['w'], values=[0.0, 44 / 3, 55 / 3], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0.0, 1, 2, 3], unit='Å')},
    )

    sc.testing.assert_identical(normalized, expected)


def test_normalize_by_monitor_histogram_monitor_finer_bins_in_monitor() -> None:
    detector = sc.DataArray(
        sc.array(dims=['w'], values=[0, 10, 30], unit='counts'),
        coords={'w': sc.arange('w', 3.0, unit='Å')},
    ).bin(w=sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å'))
    monitor = sc.DataArray(
        sc.array(dims=['w'], values=[5.0, 8.0, 6.0], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0.0, 1, 2, 3], unit='Å')},
    )
    normalized = normalize_by_monitor_histogram(
        detector,
        monitor=monitor,
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )

    expected = sc.DataArray(
        sc.array(dims=['w'], values=[0.0, 95 / 12, 95 / 3], unit='counts'),
        coords={'w': sc.arange('w', 3.0, unit='Å')},
    ).bin(w=sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å'))

    sc.testing.assert_allclose(normalized, expected)


def test_normalize_by_monitor_histogram_monitor_finer_bins_in_monitor_hist() -> None:
    detector = sc.DataArray(
        sc.array(dims=['w'], values=[10, 30], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å')},
    )
    monitor = sc.DataArray(
        sc.array(dims=['w'], values=[5.0, 8.0, 6.0], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0.0, 1, 2, 3], unit='Å')},
    )
    normalized = normalize_by_monitor_histogram(
        detector,
        monitor=monitor,
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )

    expected = sc.DataArray(
        sc.array(dims=['w'], values=[380 / 39, 95 / 3], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å')},
    )

    sc.testing.assert_allclose(normalized, expected)


def test_normalize_by_monitor_histogram_zero_count_bins_are_ignored_hist() -> None:
    detector = sc.DataArray(
        sc.array(dims=['w'], values=[0, 10, 30, 0], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[-1.0, 0, 2, 3, 4], unit='Å')},
    )
    monitor = sc.DataArray(
        sc.array(dims=['w'], values=[5.0, 6.0], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[-0.5, 2, 3], unit='Å')},
    )
    normalized = normalize_by_monitor_histogram(
        detector,
        monitor=monitor,
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )

    # The monitor is rebinned to the detector bins, which introduces
    # a 0/0 in the last bin.
    expected = sc.DataArray(
        sc.array(dims=['w'], values=[0.0, 11, 11, float('NaN')], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[-1.0, 0, 2, 3, 4], unit='Å')},
    )

    sc.testing.assert_identical(normalized, expected)


def test_normalize_by_monitor_histogram_zero_count_bins_are_ignored() -> None:
    detector = sc.DataArray(
        sc.array(dims=['w'], values=[0, 10, 30], unit='counts'),
        coords={'w': sc.arange('w', 3.0, unit='Å')},
    ).bin(w=sc.array(dims=['w'], values=[-1.0, 0, 2, 3, 4], unit='Å'))
    monitor = sc.DataArray(
        sc.array(dims=['w'], values=[5.0, 6.0], unit='counts'),
        coords={'w': sc.array(dims=['w'], values=[-0.5, 2, 3], unit='Å')},
    )
    normalized = normalize_by_monitor_histogram(
        detector,
        monitor=monitor,
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )

    expected = sc.DataArray(
        sc.array(dims=['w'], values=[0.0, 110 / 7, 110 / 7], unit='counts'),
        coords={'w': sc.arange('w', 3.0, unit='Å')},
    ).bin(w=sc.array(dims=['w'], values=[-1.0, 0, 2, 3, 4], unit='Å'))

    sc.testing.assert_allclose(normalized, expected)


def test_normalize_by_monitor_integrated_expected_results() -> None:
    detector = sc.DataArray(
        sc.arange('wavelength', 1, 4, unit='counts'),
        coords={'wavelength': sc.arange('wavelength', 3.0, unit='Å')},
    ).bin(wavelength=sc.array(dims=['wavelength'], values=[0.0, 2, 3], unit='Å'))
    monitor = sc.DataArray(
        sc.array(dims=['wavelength'], values=[4.0, 5.0, 6.0], unit='counts'),
        coords={
            'wavelength': sc.array(
                dims=['wavelength'], values=[0.0, 0.5, 2, 3], unit='Å'
            )
        },
    )
    normalized = normalize_by_monitor_integrated(
        detector,
        monitor=monitor,
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )
    expected = detector / sc.scalar(4 * 0.5 + 5 * 1.5 + 6 * 1, unit='counts * Å')
    sc.testing.assert_identical(normalized, expected)


@pytest.mark.parametrize('event_coord', [True, False])
def test_normalize_by_monitor_integrated_ignores_monitor_values_out_of_range(
    event_coord: bool,
) -> None:
    detector = sc.DataArray(
        sc.arange('wavelength', 4, unit='counts'),
        coords={'wavelength': sc.arange('wavelength', 4.0, unit='Å')},
    )
    if event_coord:
        # Make sure event at 3 is included
        detector = detector.bin(
            wavelength=sc.array(dims=['wavelength'], values=[0.0, 2, 3.1], unit='Å')
        )
        del detector.coords['wavelength']
    else:
        detector = detector.bin(
            wavelength=sc.array(dims=['wavelength'], values=[0.0, 2, 3], unit='Å')
        )
        del detector.bins.coords['wavelength']
    monitor = sc.DataArray(
        sc.array(dims=['wavelength'], values=[4.0, 10.0], unit='counts'),
        coords={
            'wavelength': sc.array(dims=['wavelength'], values=[0.0, 3, 4], unit='Å')
        },
    )
    normalized = normalize_by_monitor_integrated(
        detector,
        monitor=monitor,
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )
    expected = detector / sc.scalar(4.0 * 3, unit='counts')
    sc.testing.assert_identical(normalized, expected)


@pytest.mark.parametrize('event_coord', [True, False])
def test_normalize_by_monitor_integrated_uses_monitor_values_at_boundary(
    event_coord: bool,
) -> None:
    detector = sc.DataArray(
        sc.arange('wavelength', 4, unit='counts'),
        coords={'wavelength': sc.arange('wavelength', 4.0, unit='Å')},
    )
    if event_coord:
        # Make sure event at 3 is included
        detector = detector.bin(
            wavelength=sc.array(dims=['wavelength'], values=[0.0, 2, 3.1], unit='Å')
        )
        del detector.coords['wavelength']
    else:
        detector = detector.bin(
            wavelength=sc.array(dims=['wavelength'], values=[0.0, 2, 3], unit='Å')
        )
        del detector.bins.coords['wavelength']
    monitor = sc.DataArray(
        sc.array(dims=['wavelength'], values=[4.0, 10.0], unit='counts'),
        coords={
            'wavelength': sc.array(dims=['wavelength'], values=[0.0, 2, 4], unit='Å')
        },
    )
    normalized = normalize_by_monitor_integrated(
        detector,
        monitor=monitor,
        uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
    )
    expected = detector / sc.scalar(4.0 * 2 + 10.0 * 2, unit='counts')
    sc.testing.assert_identical(normalized, expected)


def test_normalize_by_monitor_integrated_raises_if_monitor_range_too_narrow() -> None:
    detector = sc.DataArray(
        sc.arange('wavelength', 3, unit='counts'),
        coords={'wavelength': sc.arange('wavelength', 3.0, unit='Å')},
    ).bin(wavelength=sc.array(dims=['wavelength'], values=[0.0, 2, 3], unit='Å'))
    monitor = sc.DataArray(
        sc.array(dims=['wavelength'], values=[4.0, 10.0], unit='counts'),
        coords={
            'wavelength': sc.array(dims=['wavelength'], values=[1.0, 3, 4], unit='Å')
        },
    )
    with pytest.raises(ValueError, match="smaller than the range of the detector"):
        normalize_by_monitor_integrated(
            detector,
            monitor=monitor,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )
