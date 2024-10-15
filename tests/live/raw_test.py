# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import scipp as sc

from ess.reduce.live import raw


def test_clear_counts_resets_counts_to_zero() -> None:
    params = raw.DetectorParams(
        detector_number=sc.array(dims=['pixel'], values=[1, 2, 3], unit=None)
    )
    det = raw.Detector(params)
    assert det.data.sum().value == 0
    det.add_counts([1, 2, 3, 2])
    assert det.data.sum().value == 4
    det.clear_counts()
    assert det.data.sum().value == 0
