# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import numpy as np

from ess.reduce.time_of_flight.interpolator_numba import (
    Interpolator as InterpolatorNumba,
)
from ess.reduce.time_of_flight.interpolator_scipy import (
    Interpolator as InterpolatorScipy,
)


def _f(x, y):
    """
    Function to interpolate, copied from Scipy docs. See
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RegularGridInterpolator.html
    """

    return 2 * x**3 + 3 * y**2


def _make_interpolators():
    time_edges = np.linspace(0, 71, 101)
    distance_edges = np.linspace(40, 70, 201)
    time_g, distance_g = np.meshgrid(
        time_edges, distance_edges, indexing='ij', sparse=True
    )
    values = _f(time_g, distance_g).T

    numba_interp = InterpolatorNumba(
        time_edges=time_edges, distance_edges=distance_edges, values=values
    )

    scipy_interp = InterpolatorScipy(
        time_edges=time_edges, distance_edges=distance_edges, values=values
    )
    return numba_interp, scipy_interp


def test_numba_and_scipy_interpolators_yield_same_results():
    numba_interp, scipy_interp = _make_interpolators()

    rng = np.random.default_rng(seed=42)
    npoints = 1000
    times = rng.uniform(0, 71, npoints)
    distances = rng.uniform(40, 70, npoints)

    numba_result = numba_interp(times, distances)
    scipy_result = scipy_interp(times, distances)

    # Do not use equal_nan because there should be no NaNs here
    assert np.allclose(numba_result, scipy_result)


def test_numba_and_scipy_interpolators_yield_same_results_with_pulse_offset():
    numba_interp, scipy_interp = _make_interpolators()

    rng = np.random.default_rng(seed=42)
    npoints = 1000
    times = rng.uniform(0, 71, npoints)
    distances = rng.uniform(40, 70, npoints)
    offsets = rng.uniform(0, 2, npoints)
    period = 1.0

    numba_result = numba_interp(times, distances, period, offsets)
    scipy_result = scipy_interp(times, distances, period, offsets)

    assert np.allclose(numba_result, scipy_result, equal_nan=True)


def test_numba_and_scipy_interpolators_yield_same_results_with_out_of_bounds():
    numba_interp, scipy_interp = _make_interpolators()

    rng = np.random.default_rng(seed=42)
    npoints = 1000
    times = rng.uniform(-1, 72, npoints)
    distances = rng.uniform(39, 71, npoints)

    numba_result = numba_interp(times, distances)
    scipy_result = scipy_interp(times, distances)

    assert np.allclose(numba_result, scipy_result, equal_nan=True)


def test_numba_and_scipy_interpolators_yield_same_results_with_values_on_edges():
    numba_interp, scipy_interp = _make_interpolators()

    rng = np.random.default_rng(seed=42)
    npoints = 2

    times = np.array([0.0, 71.0])
    distances = rng.uniform(39, 71, npoints)
    numba_result = numba_interp(times, distances)
    scipy_result = scipy_interp(times, distances)
    assert np.allclose(numba_result, scipy_result, equal_nan=True)

    times = rng.uniform(0, 71, npoints)
    distances = np.array([40.0, 70.0])
    numba_result = numba_interp(times, distances)
    scipy_result = scipy_interp(times, distances)
    assert np.allclose(numba_result, scipy_result, equal_nan=True)
