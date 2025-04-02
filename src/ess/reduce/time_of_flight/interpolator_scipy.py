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
        method: str = "linear",
        bounds_error: bool = False,
        fill_value: float = np.nan,
        **kwargs,
    ):
        """
        Interpolator for 3D regular grid data (SciPy implementation).

        Parameters
        ----------
        time_edges:
            1D array of time edges.
        distance_edges:
            1D array of distance edges.
        pulse_edges:
            1D array of pulse edges.
        values:
            3D array of values on the grid. The shape must be (nz, ny, nx).
        method:
            Method of interpolation. Default is "linear".
        bounds_error:
            If True, when interpolated values are requested outside of the domain,
            a ValueError is raised. If False, fill_value is used.
        fill_value:
            Value to use for points outside of the grid.
        kwargs:
            Additional arguments to pass to scipy.interpolate.RegularGridInterpolator.
        """
        from scipy.interpolate import RegularGridInterpolator

        self._interp = RegularGridInterpolator(
            (
                pulse_edges,
                distance_edges,
                time_edges,
            ),
            values,
            method=method,
            bounds_error=bounds_error,
            fill_value=fill_value,
            **kwargs,
        )

    def __call__(
        self, times: np.ndarray, distances: np.ndarray, pulse_indices: np.ndarray
    ) -> np.ndarray:
        return self._interp((pulse_indices, distances, times))
