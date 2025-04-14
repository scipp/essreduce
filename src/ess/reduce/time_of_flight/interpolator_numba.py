# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
from numba import njit, prange


@njit(boundscheck=False, cache=True, fastmath=False, parallel=True)
def interpolate(
    x: np.ndarray,
    y: np.ndarray,
    values: np.ndarray,
    xp: np.ndarray,
    yp: np.ndarray,
    xoffset: np.ndarray | None,
    deltax: float,
    fill_value: float,
    out: np.ndarray,
):
    """
    Linear interpolation of data on a 2D regular grid.

    Parameters
    ----------
    x:
        1D array of grid edges along the x-axis (size nx). They must be linspaced.
    y:
        1D array of grid edges along the y-axis (size ny). They must be linspaced.
    values:
        2D array of values on the grid. The shape must be (ny, nx).
    xp:
        1D array of x-coordinates where to interpolate (size N).
    yp:
        1D array of y-coordinates where to interpolate (size N).
    xoffset:
        1D array of integer offsets to apply to the x-coordinates (size N).
    deltax:
        Multiplier to apply to the integer offsets (i.e. the step size).
    fill_value:
        Value to use for points outside of the grid.
    out:
        1D array where the interpolated values will be stored (size N).
    """
    if not (len(xp) == len(yp) == len(out)):
        raise ValueError("Interpolator: all input arrays must have the same size.")

    nx = len(x)
    ny = len(y)
    npoints = len(xp)
    xmin = x[0]
    xmax = x[nx - 1]
    ymin = y[0]
    ymax = y[ny - 1]
    dx = x[1] - xmin
    dy = y[1] - ymin

    one_over_dx = 1.0 / dx
    one_over_dy = 1.0 / dy
    norm = one_over_dx * one_over_dy

    for i in prange(npoints):
        xx = xp[i] + (xoffset[i] * deltax if xoffset is not None else 0.0)
        yy = yp[i]

        if (xx < xmin) or (xx > xmax) or (yy < ymin) or (yy > ymax):
            out[i] = fill_value

        else:
            ix = nx - 2 if xx == xmax else int((xx - xmin) * one_over_dx)
            iy = ny - 2 if yy == ymax else int((yy - ymin) * one_over_dy)

            x1 = x[ix]
            x2 = x[ix + 1]
            y1 = y[iy]
            y2 = y[iy + 1]

            a11 = values[iy, ix]
            a21 = values[iy, ix + 1]
            a12 = values[iy + 1, ix]
            a22 = values[iy + 1, ix + 1]

            x2mxx = x2 - xx
            xxmx1 = xx - x1

            out[i] = (
                (y2 - yy) * (x2mxx * a11 + xxmx1 * a21)
                + (yy - y1) * (x2mxx * a12 + xxmx1 * a22)
            ) * norm


class Interpolator:
    def __init__(
        self,
        time_edges: np.ndarray,
        distance_edges: np.ndarray,
        values: np.ndarray,
        fill_value: float = np.nan,
    ):
        """
        Interpolator for 2D regular grid data (Numba implementation).

        Parameters
        ----------
        time_edges:
            1D array of time edges.
        distance_edges:
            1D array of distance edges.
        values:
            2D array of values on the grid. The shape must be (ny, nx).
        fill_value:
            Value to use for points outside of the grid.
        """
        self.time_edges = time_edges
        self.distance_edges = distance_edges
        self.values = values
        self.fill_value = fill_value

    def __call__(
        self,
        times: np.ndarray,
        distances: np.ndarray,
        pulse_period: float = 0.0,
        pulse_index: np.ndarray | None = None,
    ) -> np.ndarray:
        out = np.empty_like(times)
        interpolate(
            x=self.time_edges,
            y=self.distance_edges,
            values=self.values,
            xp=times,
            yp=distances,
            xoffset=pulse_index,
            deltax=pulse_period,
            fill_value=self.fill_value,
            out=out,
        )
        return out
