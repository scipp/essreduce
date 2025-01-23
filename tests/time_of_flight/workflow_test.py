# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)

import pytest
import scipp as sc

from ess.reduce import time_of_flight
from ess.reduce.time_of_flight import fakes


@pytest.fixture(scope="module")
def simulation_results():
    return time_of_flight.simulate_beamline(
        choppers=fakes.psc_choppers(), neutrons=100_000, seed=432
    )


def test_create_workflow(simulation_results):
    wf = time_of_flight.TofWorkflow(
        simulated_neutrons=simulation_results,
        ltotal_range=(sc.scalar(77.0, unit='m'), sc.scalar(82.0, unit='m')),
        error_threshold=1.0e-3,
    )

    assert wf.compute(time_of_flight.LookupTableRelativeErrorThreshold) == 1.0e-3
    assert (
        wf.compute(time_of_flight.PulsePeriod)
        == time_of_flight.default_parameters()[time_of_flight.PulsePeriod]
    )


def test_workflow_setitem(simulation_results):
    wf = time_of_flight.TofWorkflow(
        simulated_neutrons=simulation_results,
        ltotal_range=(sc.scalar(77.0, unit='m'), sc.scalar(82.0, unit='m')),
        error_threshold=1.0e-3,
    )

    assert wf.compute(time_of_flight.LookupTableRelativeErrorThreshold) == 1.0e-3

    wf[time_of_flight.LookupTableRelativeErrorThreshold] = 1.0e-4
    assert wf.compute(time_of_flight.LookupTableRelativeErrorThreshold) == 1.0e-4

    wf[time_of_flight.PulsePeriod] = sc.scalar(1.0, unit='s')
    assert wf.compute(time_of_flight.PulsePeriod) == sc.scalar(1.0, unit='s')


def test_workflow_copy(simulation_results):
    wf = time_of_flight.TofWorkflow(
        simulated_neutrons=simulation_results,
        ltotal_range=(sc.scalar(77.0, unit='m'), sc.scalar(82.0, unit='m')),
        error_threshold=1.0e-3,
    )

    wf_copy = wf.copy()
    assert wf.compute(
        time_of_flight.LookupTableRelativeErrorThreshold
    ) == wf_copy.compute(time_of_flight.LookupTableRelativeErrorThreshold)
    assert wf.compute(time_of_flight.PulsePeriod) == wf_copy.compute(
        time_of_flight.PulsePeriod
    )
