# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
from numba import njit, prange


@njit(boundscheck=False, cache=True, fastmath=False, parallel=True)
def interpolate(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    values: np.ndarray,
    xp: np.ndarray,
    yp: np.ndarray,
    zp: np.ndarray,
    fill_value: float,
    out: np.ndarray,
):
    """
    Linear interpolation of data on a 3D regular grid.

    Parameters
    ----------
    x:
        1D array of grid edges along the x-axis. They must be linspaced.
    y:
        1D array of grid edges along the y-axis. They must be linspaced.
    z:
        1D array of grid edges along the z-axis. They must be linspaced.
    values:
        3D array of values on the grid. The shape must be (nz, ny, nx).
    xp:
        1D array of x-coordinates where to interpolate (size N).
    yp:
        1D array of y-coordinates where to interpolate (size N).
    zp:
        1D array of z-coordinates where to interpolate (size N).
    fill_value:
        Value to use for points outside of the grid.
    out:
        1D array where the interpolated values will be stored (size N).
    """
    if not (len(xp) == len(yp) == len(zp) == len(out)):
        raise ValueError("Interpolator: all input arrays must have the same size.")

    nx = len(x)
    ny = len(y)
    nz = len(z)
    npoints = len(xp)
    xmin = x[0]
    xmax = x[nx - 1]
    ymin = y[0]
    ymax = y[ny - 1]
    zmin = z[0]
    zmax = z[nz - 1]
    dx = x[1] - xmin
    dy = y[1] - ymin
    dz = z[1] - zmin

    one_over_dx = 1.0 / dx
    one_over_dy = 1.0 / dy
    one_over_dz = 1.0 / dz
    norm = one_over_dx * one_over_dy * one_over_dz

    for i in prange(npoints):
        xx = xp[i]
        yy = yp[i]
        zz = zp[i]

        if (
            (xx < xmin)
            or (xx > xmax)
            or (yy < ymin)
            or (yy > ymax)
            or (zz < zmin)
            or (zz > zmax)
        ):
            out[i] = fill_value

        else:
            ix = nx - 2 if xx == xmax else int((xx - xmin) * one_over_dx)
            iy = ny - 2 if yy == ymax else int((yy - ymin) * one_over_dy)
            iz = nz - 2 if zz == zmax else int((zz - zmin) * one_over_dz)

            x1 = x[ix]
            x2 = x[ix + 1]
            y1 = y[iy]
            y2 = y[iy + 1]
            z1 = z[iz]
            z2 = z[iz + 1]

            a111 = values[iz, iy, ix]
            a211 = values[iz, iy, ix + 1]
            a121 = values[iz, iy + 1, ix]
            a221 = values[iz, iy + 1, ix + 1]
            a112 = values[iz + 1, iy, ix]
            a212 = values[iz + 1, iy, ix + 1]
            a122 = values[iz + 1, iy + 1, ix]
            a222 = values[iz + 1, iy + 1, ix + 1]

            x2mxx = x2 - xx
            xxmx1 = xx - x1
            y2myy = y2 - yy
            yymy1 = yy - y1
            out[i] = (
                (z2 - zz)
                * (
                    y2myy * (x2mxx * a111 + xxmx1 * a211)
                    + yymy1 * (x2mxx * a121 + xxmx1 * a221)
                )
                + (zz - z1)
                * (
                    y2myy * (x2mxx * a112 + xxmx1 * a212)
                    + yymy1 * (x2mxx * a122 + xxmx1 * a222)
                )
            ) * norm


class Interpolator:
    def __init__(
        self,
        time_edges: np.ndarray,
        distance_edges: np.ndarray,
        pulse_edges: np.ndarray,
        values: np.ndarray,
        fill_value: float = np.nan,
    ):
        """
        Interpolator for 3D regular grid data (Numba implementation).

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
        fill_value:
            Value to use for points outside of the grid.
        """
        self.time_edges = time_edges
        self.distance_edges = distance_edges
        self.pulse_edges = pulse_edges
        self.values = values
        self.fill_value = fill_value

    def __call__(
        self, times: np.ndarray, distances: np.ndarray, pulse_indices: np.ndarray
    ) -> np.ndarray:
        out = np.empty_like(times)
        interpolate(
            x=self.time_edges,
            y=self.distance_edges,
            z=self.pulse_edges,
            values=self.values,
            xp=times,
            yp=distances,
            zp=pulse_indices,
            fill_value=self.fill_value,
            out=out,
        )
        return out
