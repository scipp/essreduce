# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
import numpy as np
from numba import njit, prange


@njit(boundscheck=False, cache=True, fastmath=True, parallel=True)
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
    npoints = len(xp)
    xmin = x[0]
    xmax = x[-1]
    ymin = y[0]
    ymax = y[-1]
    zmin = z[0]
    zmax = z[-1]
    dx = x[1] - xmin
    dy = y[1] - ymin
    dz = z[1] - zmin

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
            ix = int((xx - xmin) / dx)
            iy = int((yy - ymin) / dy)
            iz = int((zz - zmin) / dz)

            y2 = y[iy + 1]
            y1 = y[iy]
            x2 = x[ix + 1]
            x1 = x[ix]
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

            x2mx1 = x2 - x1
            y2my1 = y2 - y1
            z2mz1 = z2 - z1

            x2mxox2mx1 = (x2 - xx) / x2mx1
            xmx1ox2mx1 = (xx - x1) / x2mx1
            y2myoy2my1 = (y2 - yy) / y2my1
            ymy1oy2my1 = (yy - y1) / y2my1
            z2mzoz2mz1 = (z2 - zz) / z2mz1
            zmz1oz2mz1 = (zz - z1) / z2mz1

            out[i] = z2mzoz2mz1 * (
                y2myoy2my1 * (x2mxox2mx1 * a111 + xmx1ox2mx1 * a211)
                + ymy1oy2my1 * (x2mxox2mx1 * a121 + xmx1ox2mx1 * a221)
            ) + zmz1oz2mz1 * (
                y2myoy2my1 * (x2mxox2mx1 * a112 + xmx1ox2mx1 * a212)
                + ymy1oy2my1 * (x2mxox2mx1 * a122 + xmx1ox2mx1 * a222)
            )


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
