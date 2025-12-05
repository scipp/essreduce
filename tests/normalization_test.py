# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import numpy as np
import pytest
import scipp as sc
import scipp.testing

from ess.reduce.normalization import (
    normalize_by_monitor_histogram,
    normalize_by_monitor_integrated,
)
from ess.reduce.uncertainty import UncertaintyBroadcastMode


class TestNormalizeByMonitorHistogram:
    def test_aligned_bins_w_event_coord(self) -> None:
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
            sc.array(dims=['w'], values=[0.0, 4, 5], unit='counts'),
            coords={'w': sc.arange('w', 3.0, unit='Å')},
        ).bin(w=sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å'))

        sc.testing.assert_identical(normalized, expected)

    def test_aligned_bins_wo_event_coord(self) -> None:
        detector = sc.DataArray(
            sc.array(dims=['w'], values=[0, 10, 30], unit='counts'),
            coords={'w': sc.arange('w', 3.0, unit='Å')},
        ).bin(w=sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å'))
        del detector.bins.coords['w']
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
            sc.array(dims=['w'], values=[0.0, 4, 5], unit='counts'),
            coords={'w': sc.arange('w', 3.0, unit='Å')},
        ).bin(w=sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å'))
        del expected.bins.coords['w']

        sc.testing.assert_identical(normalized, expected)

    def test_aligned_bins_hist(self) -> None:
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
            sc.array(dims=['w'], values=[4.0, 5.0], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å')},
        )

        sc.testing.assert_identical(normalized, expected)

    def test_monitor_envelops_detector_bin(self) -> None:
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
            sc.array(dims=['w'], values=[0.0, 5, 7.5], unit='counts'),
            coords={'w': sc.arange('w', 3.0, unit='Å')},
        ).bin(w=sc.array(dims=['w'], values=[0.0, 2, 2.5], unit='Å'))

        sc.testing.assert_identical(normalized, expected)

    def test_monitor_envelops_detector_bin_hist(
        self,
    ) -> None:
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
            sc.array(dims=['w'], values=[4, 15 / 2], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[0.0, 2, 2.5], unit='Å')},
        )

        sc.testing.assert_identical(normalized, expected)

    def test_detector_envelops_monitor_bin(self) -> None:
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

    def test_detector_envelops_monitor_bin_hist(
        self,
    ) -> None:
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

    def test_extra_bins_in_monitor(self) -> None:
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
            sc.array(dims=['w'], values=[0.0, 4, 5], unit='counts'),
            coords={'w': sc.arange('w', 3.0, unit='Å')},
        ).bin(w=sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å'))

        sc.testing.assert_identical(normalized, expected)

    def test_extra_bins_in_monitor_hist(self) -> None:
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
            sc.array(dims=['w'], values=[4.0, 5], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å')},
        )

        sc.testing.assert_identical(normalized, expected)

    def test_extra_bins_in_detector(self) -> None:
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

    def test_finer_bins_in_detector(self) -> None:
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
            sc.array(dims=['w'], values=[0.0, 4, 5], unit='counts'),
            coords={'w': sc.arange('w', 3.0, unit='Å')},
        ).bin(w=sc.array(dims=['w'], values=[0.0, 1, 2, 3], unit='Å'))

        sc.testing.assert_identical(normalized, expected)

    def test_finer_bins_in_detector_hist(self) -> None:
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
            sc.array(dims=['w'], values=[0.0, 4, 5], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[0.0, 1, 2, 3], unit='Å')},
        )

        sc.testing.assert_identical(normalized, expected)

    def test_finer_bins_in_monitor(self) -> None:
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
            sc.array(dims=['w'], values=[0.0, 5 / 4, 5], unit='counts'),
            coords={'w': sc.arange('w', 3.0, unit='Å')},
        ).bin(w=sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å'))

        sc.testing.assert_allclose(normalized, expected)

    def test_finer_bins_in_monitor_hist(self) -> None:
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
            sc.array(dims=['w'], values=[20 / 13, 5], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å')},
        )

        sc.testing.assert_allclose(normalized, expected)

    def test_monitor_mask_aligned_bins(self) -> None:
        detector = sc.DataArray(
            sc.array(dims=['w'], values=[0, 10, 20, 30], unit='counts'),
            coords={'w': sc.arange('w', 1.0, 5.0, unit='Å')},
        ).bin(w=sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å'))
        monitor = sc.DataArray(
            sc.array(dims=['w'], values=[5.0, 6.0, 7.0], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å')},
            masks={'m': sc.array(dims=['w'], values=[False, True, False])},
        )
        normalized = normalize_by_monitor_histogram(
            detector,
            monitor=monitor,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )

        expected = (
            sc.DataArray(
                sc.array(dims=['w'], values=[0.0, 4, 0, 90 / 7], unit='counts'),
                coords={'w': sc.arange('w', 1.0, 5.0, unit='Å')},
            )
            .bin(w=sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å'))
            .assign_masks(
                _monitor_mask=sc.array(dims=['w'], values=[False, True, False])
            )
        )

        sc.testing.assert_allclose(normalized, expected)

    def test_monitor_mask_multiple(self) -> None:
        detector = sc.DataArray(
            sc.array(dims=['w'], values=[0, 10, 20, 30], unit='counts'),
            coords={'w': sc.arange('w', 1.0, 5.0, unit='Å')},
        ).bin(w=sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å'))
        monitor = sc.DataArray(
            sc.array(dims=['w'], values=[5.0, 6.0, 7.0], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å')},
            masks={
                'm1': sc.array(dims=['w'], values=[False, True, False]),
                'm2': sc.array(dims=['w'], values=[False, True, True]),
            },
        )
        normalized = normalize_by_monitor_histogram(
            detector,
            monitor=monitor,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )

        expected = (
            sc.DataArray(
                sc.array(dims=['w'], values=[0.0, 4, 0, 0], unit='counts'),
                coords={'w': sc.arange('w', 1.0, 5.0, unit='Å')},
            )
            .bin(w=sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å'))
            .assign_masks(
                _monitor_mask=sc.array(dims=['w'], values=[False, True, True])
            )
        )

        sc.testing.assert_identical(normalized, expected)

    def test_monitor_and_detector_mask_aligned_bins(self) -> None:
        detector = (
            sc.DataArray(
                sc.array(dims=['w'], values=[0, 10, 20, 30], unit='counts'),
                coords={'w': sc.arange('w', 1.0, 5.0, unit='Å')},
            )
            .bin(w=sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å'))
            .assign_masks(d=sc.array(dims=['w'], values=[True, False, False]))
        )
        monitor = sc.DataArray(
            sc.array(dims=['w'], values=[5.0, 6.0, 7.0], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å')},
            masks={'m': sc.array(dims=['w'], values=[False, True, False])},
        )
        normalized = normalize_by_monitor_histogram(
            detector,
            monitor=monitor,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )

        expected = (
            sc.DataArray(
                sc.array(dims=['w'], values=[0.0, 4, 0, 90 / 7], unit='counts'),
                coords={'w': sc.arange('w', 1.0, 5.0, unit='Å')},
            )
            .bin(w=sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å'))
            .assign_masks(
                d=sc.array(dims=['w'], values=[True, False, False]),
                _monitor_mask=sc.array(dims=['w'], values=[False, True, False]),
            )
        )

        sc.testing.assert_allclose(normalized, expected)

    def test_monitor_mask_unaligned_bins(self) -> None:
        detector = sc.DataArray(
            sc.array(dims=['w'], values=[-10, 10, 20, 30], unit='counts'),
            coords={'w': sc.arange('w', 1.0, 5.0, unit='Å')},
        ).bin(w=sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å'))
        monitor = sc.DataArray(
            sc.array(dims=['w'], values=[5.0, 6.0, 7.0, 8.0], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[0.0, 2, 3.5, 4, 7], unit='Å')},
            masks={'m': sc.array(dims=['w'], values=[False, True, False, False])},
        )

        normalized = normalize_by_monitor_histogram(
            detector,
            monitor=monitor,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )

        expected = (
            sc.DataArray(
                sc.array(dims=['w'], values=[-4, 0, 0, 45 / 4], unit='counts'),
                coords={'w': sc.arange('w', 1.0, 5.0, unit='Å')},
            )
            .bin(w=sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å'))
            .assign_masks(
                _monitor_mask=sc.array(dims=['w'], values=[True, True, False])
            )
        )

        sc.testing.assert_identical(normalized, expected)

    def test_monitor_mask_at_edge(self) -> None:
        detector = sc.DataArray(
            sc.array(dims=['w'], values=[0, 10, 30], unit='counts'),
            coords={'w': sc.arange('w', 3.0, unit='Å')},
        ).bin(w=sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å'))
        monitor = sc.DataArray(
            sc.array(dims=['w'], values=[5.0, 6.0], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å')},
            masks={'m': sc.array(dims=['w'], values=[False, True])},
        )
        normalized = normalize_by_monitor_histogram(
            detector,
            monitor=monitor,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )

        expected = (
            sc.DataArray(
                sc.array(dims=['w'], values=[0, 4, 0], unit='counts'),
                coords={'w': sc.arange('w', 3.0, unit='Å')},
            )
            .bin(w=sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å'))
            .assign_masks(_monitor_mask=sc.array(dims=['w'], values=[False, True]))
        )

        sc.testing.assert_identical(normalized, expected)

    def test_monitor_mask_different_bin_dim(self) -> None:
        detector = sc.DataArray(
            sc.array(dims=['e'], values=[0, 10, 20, 30], unit='counts'),
            coords={
                'w': sc.arange('e', 1.0, 5.0, unit='Å'),
                'd': sc.array(dims=['e'], values=[0, 1, 0, 2]),
            },
        ).group('d')
        monitor = sc.DataArray(
            sc.array(dims=['w'], values=[5.0, 6.0, 7.0], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å')},
            masks={'m': sc.array(dims=['w'], values=[False, True, False])},
        )
        normalized = normalize_by_monitor_histogram(
            detector,
            monitor=monitor,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )

        expected = sc.DataArray(
            sc.array(dims=['e'], values=[0.0, 4, 0, 90 / 7], unit='counts'),
            coords={
                'w': sc.arange('e', 1.0, 5.0, unit='Å'),
                'd': sc.array(dims=['e'], values=[0, 1, 0, 2]),
            },
            masks={
                '_monitor_mask': sc.array(
                    dims=['e'], values=[False, False, True, False]
                )
            },
        ).group('d')

        sc.testing.assert_allclose(normalized, expected)

    @pytest.mark.parametrize("nonfinite_value", [np.nan, np.inf])
    def test_nonfinite_in_monitor_gets_masked(
        self,
        nonfinite_value: float,
    ) -> None:
        detector = sc.DataArray(
            sc.array(dims=['w'], values=[0, 10, 20, 30], unit='counts'),
            coords={'w': sc.arange('w', 1.0, 5.0, unit='Å')},
        ).bin(w=sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å'))
        monitor = sc.DataArray(
            sc.array(dims=['w'], values=[nonfinite_value, 6.0, 7.0], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å')},
        )
        normalized = normalize_by_monitor_histogram(
            detector,
            monitor=monitor,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )

        expected = (
            sc.DataArray(
                sc.array(
                    dims=['w'],
                    values=[1 / nonfinite_value, 1 / nonfinite_value, 10 / 3, 90 / 7],
                    unit='counts',
                ),
                coords={'w': sc.arange('w', 1.0, 5.0, unit='Å')},
            )
            .bin(w=sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å'))
            .assign_masks(
                _monitor_mask=sc.array(dims=['w'], values=[True, False, False])
            )
        )

        sc.testing.assert_allclose(normalized, expected)

    def test_zeros_in_monitor_get_masked(self) -> None:
        detector = sc.DataArray(
            sc.array(dims=['w'], values=[0, 10, 20, 30], unit='counts'),
            coords={'w': sc.arange('w', 1.0, 5.0, unit='Å')},
        ).bin(w=sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å'))
        monitor = sc.DataArray(
            sc.array(dims=['w'], values=[0.0, 0.0, 7.0], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å')},
        )
        normalized = normalize_by_monitor_histogram(
            detector,
            monitor=monitor,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )

        expected = (
            sc.DataArray(
                sc.array(
                    dims=['w'],
                    values=[np.nan, np.inf, np.inf, 90 / 7],
                    unit='counts',
                ),
                coords={'w': sc.arange('w', 1.0, 5.0, unit='Å')},
            )
            .bin(w=sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å'))
            .assign_masks(
                _monitor_mask=sc.array(dims=['w'], values=[True, True, False])
            )
        )

        sc.testing.assert_allclose(normalized, expected)

    def test_different_dims_dense(self) -> None:
        detector = sc.DataArray(
            sc.array(  # sizes={'x': 3, 'y': 2}
                dims=['x', 'y'], values=[[11.0, 10], [9, 8], [7, 6]], unit='counts'
            ),
            coords={
                'x': sc.array(dims=['x'], values=[0.0, 1, 2, 3], unit='m'),
                'y': sc.array(dims=['y'], values=[-5, -1, 4], unit='kg'),
                'w': sc.array(  # bin edges in x
                    dims=['x', 'y'], values=[[1.0, 2], [2, 3], [4, 5], [5, 3]], unit='Å'
                ),
            },
        )
        monitor = sc.DataArray(
            sc.array(dims=['w'], values=[3.0, 5, 7], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[0.0, 2, 4, 6], unit='Å')},
            masks={'M': sc.array(dims=['w'], values=[False, True, False])},
        )
        normalized = normalize_by_monitor_histogram(
            detector,
            monitor=monitor,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )

        expected = sc.DataArray(
            sc.array(
                dims=['x', 'y'],
                values=[[22 / 3, np.nan], [np.nan, 16 / 7], [2.0, 12 / 7]],
                unit='counts',
            ),
            coords=detector.coords,
            masks={
                '_monitor_mask': sc.array(
                    dims=['x', 'y'],
                    values=[[False, True], [True, False], [False, False]],
                ),
            },
        )

        sc.testing.assert_allclose(normalized, expected)

    def test_independent_of_monitor_binning_bin(self) -> None:
        detector = sc.DataArray(
            sc.array(dims=['w'], values=[3, 10, 20, 30], unit='counts'),
            coords={'w': sc.arange('w', 1.0, 5.0, unit='Å')},
        ).bin(w=sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å'))

        monitor1 = sc.DataArray(
            sc.array(dims=['w'], values=[5.0, 6.0, 7.0], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[1.0, 2, 4, 8], unit='Å')},
        )
        monitor2 = monitor1.rebin(
            w=sc.array(dims=['w'], values=[1.0, 2, 3, 4, 7], unit='Å')
        )

        normalized1 = normalize_by_monitor_histogram(
            detector,
            monitor=monitor1,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )
        normalized2 = normalize_by_monitor_histogram(
            detector,
            monitor=monitor2,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )

        sc.testing.assert_identical(normalized1, normalized2)

    def test_independent_of_monitor_binning_hist(self) -> None:
        detector = sc.DataArray(
            sc.array(dims=['w'], values=[10, 20, 30], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å')},
        )

        monitor1 = sc.DataArray(
            sc.array(dims=['w'], values=[5.0, 6.0, 7.0], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[1.0, 2, 4, 8], unit='Å')},
        )
        monitor2 = monitor1.rebin(
            w=sc.array(dims=['w'], values=[1.0, 2, 3, 4, 7], unit='Å')
        )

        normalized1 = normalize_by_monitor_histogram(
            detector,
            monitor=monitor1,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )
        normalized2 = normalize_by_monitor_histogram(
            detector,
            monitor=monitor2,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )

        sc.testing.assert_identical(normalized1, normalized2)


class TestNormalizeByMonitorIntegrated:
    def test_expected_results_bin(self) -> None:
        detector = sc.DataArray(
            sc.arange('w', 1, 4, unit='counts'),
            coords={'w': sc.arange('w', 3.0, unit='Å')},
        ).bin(w=sc.array(dims=['w'], values=[0.0, 2, 3], unit='Å'))
        monitor = sc.DataArray(
            sc.array(dims=['w'], values=[4.0, 5.0, 6.0, 10], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[0.0, 0.5, 2, 3, 4], unit='Å')},
        )
        normalized = normalize_by_monitor_integrated(
            detector,
            monitor=monitor,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )
        expected = detector / monitor.sum()
        sc.testing.assert_identical(normalized, expected)

    def test_expected_results_hist(self) -> None:
        detector = sc.DataArray(
            sc.arange('w', 1, 4, unit='counts'),
            coords={'w': sc.arange('w', 4.0, unit='Å')},
        )
        monitor = sc.DataArray(
            sc.array(dims=['w'], values=[4.0, 5.0, 6.0, 10], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[0.0, 0.5, 2, 3, 4], unit='Å')},
        )
        normalized = normalize_by_monitor_integrated(
            detector,
            monitor=monitor,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )
        expected = detector / monitor.sum()
        sc.testing.assert_identical(normalized, expected)

    def test_monitor_mask_different_bin_dim(self) -> None:
        detector = sc.DataArray(
            sc.array(dims=['e'], values=[0, 10, 20, 30], unit='counts'),
            coords={
                'w': sc.arange('e', 1.0, 5.0, unit='Å'),
                'd': sc.array(dims=['e'], values=[0, 1, 0, 2]),
            },
        ).group('d')
        monitor = sc.DataArray(
            sc.array(dims=['w'], values=[5.0, 6.0, 7.0], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å')},
            masks={'m': sc.array(dims=['w'], values=[False, True, False])},
        )
        normalized = normalize_by_monitor_integrated(
            detector,
            monitor=monitor,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )

        expected = sc.DataArray(
            sc.array(
                dims=['e'], values=[0 / 12, 10 / 12, 20 / 12, 30 / 12], unit='counts'
            ),
            coords={
                'w': sc.arange('e', 1.0, 5.0, unit='Å'),
                'd': sc.array(dims=['e'], values=[0, 1, 0, 2]),
            },
            masks={
                '_monitor_mask': sc.array(
                    dims=['e'], values=[False, False, True, False]
                )
            },
        ).group('d')

        sc.testing.assert_allclose(normalized, expected)

    def test_raises_if_monitor_range_too_narrow(
        self,
    ) -> None:
        detector = sc.DataArray(
            sc.arange('wavelength', 3, unit='counts'),
            coords={'wavelength': sc.arange('wavelength', 3.0, unit='Å')},
        ).bin(wavelength=sc.array(dims=['wavelength'], values=[0.0, 2, 3], unit='Å'))
        monitor = sc.DataArray(
            sc.array(dims=['wavelength'], values=[4.0, 10.0], unit='counts'),
            coords={
                'wavelength': sc.array(
                    dims=['wavelength'], values=[1.0, 3, 4], unit='Å'
                )
            },
        )
        with pytest.raises(ValueError, match="smaller than the range of the detector"):
            normalize_by_monitor_integrated(
                detector,
                monitor=monitor,
                uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
            )

    def test_independent_of_monitor_binning(self) -> None:
        detector = sc.DataArray(
            sc.array(dims=['w'], values=[3, 10, 20, 30], unit='counts'),
            coords={'w': sc.arange('w', 1.0, 5.0, unit='Å')},
        ).bin(w=sc.array(dims=['w'], values=[1.0, 3, 4, 7], unit='Å'))

        monitor1 = sc.DataArray(
            sc.array(dims=['w'], values=[5.0, 6.0, 7.0], unit='counts'),
            coords={'w': sc.array(dims=['w'], values=[1.0, 2, 4, 7], unit='Å')},
        )
        monitor2 = monitor1.rebin(
            w=sc.array(dims=['w'], values=[1.0, 2, 3, 5, 7], unit='Å')
        )

        normalized1 = normalize_by_monitor_integrated(
            detector,
            monitor=monitor1,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )
        normalized2 = normalize_by_monitor_integrated(
            detector,
            monitor=monitor2,
            uncertainty_broadcast_mode=UncertaintyBroadcastMode.fail,
        )

        sc.testing.assert_identical(normalized1, normalized2)
