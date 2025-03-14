# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import numpy as np


class Interpolator:
    def __init__(
        self,
        time_edges: np.ndarray,
        distance_edges: np.ndarray,
        pulse_edges: np.ndarray,
        values: np.ndarray,
        **kwargs,
    ):
        from scipy.interpolate import RegularGridInterpolator

        default_args = {
            "method": "linear",
            "bounds_error": False,
            "fill_value": np.nan,
        }

        self._interp = RegularGridInterpolator(
            (
                pulse_edges,
                distance_edges,
                time_edges,
            ),
            values,
            **{**default_args, **kwargs},
        )

    def __call__(
        self, times: np.ndarray, distances: np.ndarray, pulse_indices: np.ndarray
    ) -> np.ndarray:
        return self._interp((pulse_indices, distances, times))
