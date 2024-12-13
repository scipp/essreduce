# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import NewType

import pytest
import scippnexus as snx
from sciline import Pipeline, Scope

from ess.reduce.nexus.types import (
    BackgroundRun,
    DetectorData,
    EmptyBeamRun,
    IncidentMonitor,
    Monitor1,
    Monitor2,
    Monitor3,
    Monitor4,
    Monitor5,
    Monitor6,
    MonitorData,
    NeXusComponentLocationSpec,
    RunType,
    SampleRun,
    TransmissionMonitor,
)
from ess.reduce.nexus.workflow import (
    GenericNeXusWorkflow,
)
from ess.reduce.workflow import prune_nexus_domain_types


class A(Scope[RunType, int], int): ...


class B(Scope[RunType, int], int): ...


class C(Scope[RunType, int], int): ...


class D(Scope[RunType, int], int): ...


E = NewType('E', int)


def foo(a: A[RunType]) -> B[RunType]:
    return B[RunType](a + 1)


def bar(a: A[RunType]) -> C[RunType]:
    return C[RunType](a + 2)


def baz(a: A[RunType]) -> D[RunType]:
    return D[RunType](a + 3)


def combine(b_s: B[SampleRun], b_b: B[BackgroundRun], c: C[SampleRun]) -> E:
    return E(b_s + b_b + c)


def test_pruning_nexus_workflow_includes_only_given_run_types() -> None:
    wf = GenericNeXusWorkflow()
    wf = prune_nexus_domain_types(
        wf,
        targets_per_run=[DetectorData],
        targets_per_run_and_monitor=[MonitorData],
        run_types=[SampleRun],
    )
    graph = wf.underlying_graph
    assert DetectorData[SampleRun] in graph
    assert DetectorData[BackgroundRun] not in graph
    assert MonitorData[SampleRun, Monitor1] in graph
    assert MonitorData[SampleRun, Monitor2] in graph
    assert MonitorData[SampleRun, Monitor3] in graph
    assert MonitorData[BackgroundRun, Monitor1] not in graph
    assert MonitorData[BackgroundRun, Monitor2] not in graph
    assert MonitorData[BackgroundRun, Monitor3] not in graph
    assert NeXusComponentLocationSpec[Monitor1, SampleRun] in graph
    assert NeXusComponentLocationSpec[Monitor2, SampleRun] in graph
    assert NeXusComponentLocationSpec[Monitor3, SampleRun] in graph
    assert NeXusComponentLocationSpec[snx.NXdetector, SampleRun] in graph
    assert NeXusComponentLocationSpec[snx.NXsample, SampleRun] in graph
    assert NeXusComponentLocationSpec[snx.NXsource, SampleRun] in graph
    assert NeXusComponentLocationSpec[Monitor1, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[Monitor2, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[Monitor3, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[snx.NXdetector, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[snx.NXsample, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[snx.NXsource, BackgroundRun] not in graph


def test_pruning_nexus_workflow_includes_only_given_run_and_monitor_types() -> None:
    wf = GenericNeXusWorkflow()
    wf = prune_nexus_domain_types(
        wf,
        targets_per_run=[DetectorData],
        targets_per_run_and_monitor=[MonitorData],
        run_types=[SampleRun],
        monitor_types=[Monitor1, Monitor3],
    )
    graph = wf.underlying_graph
    assert DetectorData[SampleRun] in graph
    assert DetectorData[BackgroundRun] not in graph
    assert MonitorData[SampleRun, Monitor1] in graph
    assert MonitorData[SampleRun, Monitor2] not in graph
    assert MonitorData[SampleRun, Monitor3] in graph
    assert MonitorData[BackgroundRun, Monitor1] not in graph
    assert MonitorData[BackgroundRun, Monitor2] not in graph
    assert MonitorData[BackgroundRun, Monitor3] not in graph
    assert NeXusComponentLocationSpec[Monitor1, SampleRun] in graph
    assert NeXusComponentLocationSpec[Monitor2, SampleRun] not in graph
    assert NeXusComponentLocationSpec[Monitor3, SampleRun] in graph
    assert NeXusComponentLocationSpec[snx.NXdetector, SampleRun] in graph
    assert NeXusComponentLocationSpec[snx.NXsample, SampleRun] in graph
    assert NeXusComponentLocationSpec[snx.NXsource, SampleRun] in graph
    assert NeXusComponentLocationSpec[Monitor1, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[Monitor2, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[Monitor3, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[snx.NXdetector, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[snx.NXsample, BackgroundRun] not in graph
    assert NeXusComponentLocationSpec[snx.NXsource, BackgroundRun] not in graph


def test_pruning_nexus_workflow_includes_all_monitor_types_by_default() -> None:
    wf = GenericNeXusWorkflow()
    wf = prune_nexus_domain_types(
        wf,
        targets_per_run=[],
        targets_per_run_and_monitor=[MonitorData],
        run_types=[SampleRun],
    )
    graph = wf.underlying_graph
    assert MonitorData[SampleRun, Monitor1] in graph
    assert MonitorData[SampleRun, Monitor2] in graph
    assert MonitorData[SampleRun, Monitor3] in graph
    assert MonitorData[SampleRun, Monitor4] in graph
    assert MonitorData[SampleRun, Monitor5] in graph
    assert MonitorData[SampleRun, Monitor6] in graph
    assert MonitorData[SampleRun, IncidentMonitor] in graph
    assert MonitorData[SampleRun, TransmissionMonitor] in graph
    assert MonitorData[BackgroundRun, Monitor1] not in graph
    assert MonitorData[BackgroundRun, Monitor2] not in graph
    assert MonitorData[BackgroundRun, Monitor3] not in graph
    assert MonitorData[BackgroundRun, Monitor4] not in graph
    assert MonitorData[BackgroundRun, Monitor5] not in graph
    assert MonitorData[BackgroundRun, Monitor6] not in graph
    assert MonitorData[BackgroundRun, IncidentMonitor] not in graph
    assert MonitorData[BackgroundRun, TransmissionMonitor] not in graph


def test_pruning_workflow_includes_multiple_targets() -> None:
    wf = Pipeline((foo, bar, baz))
    wf = prune_nexus_domain_types(
        wf,
        targets_per_run=[B, C],
        run_types=[SampleRun],
    )
    graph = wf.underlying_graph
    assert A[SampleRun] in graph
    assert B[SampleRun] in graph
    assert C[SampleRun] in graph
    assert D[SampleRun] not in graph
    assert A[BackgroundRun] not in graph
    assert B[BackgroundRun] not in graph
    assert C[BackgroundRun] not in graph
    assert D[BackgroundRun] not in graph
    assert A[EmptyBeamRun] not in graph
    assert B[EmptyBeamRun] not in graph
    assert C[EmptyBeamRun] not in graph
    assert D[EmptyBeamRun] not in graph


def test_pruning_workflow_includes_multiple_run_types() -> None:
    wf = Pipeline((foo, bar, baz))
    wf = prune_nexus_domain_types(
        wf,
        targets_per_run=[B, C],
        run_types=[SampleRun, BackgroundRun],
    )
    graph = wf.underlying_graph
    assert A[SampleRun] in graph
    assert B[SampleRun] in graph
    assert C[SampleRun] in graph
    assert D[SampleRun] not in graph
    assert A[BackgroundRun] in graph
    assert B[BackgroundRun] in graph
    assert C[BackgroundRun] in graph
    assert D[BackgroundRun] not in graph
    assert A[EmptyBeamRun] not in graph
    assert B[EmptyBeamRun] not in graph
    assert C[EmptyBeamRun] not in graph
    assert D[EmptyBeamRun] not in graph


def test_pruning_workflow_removes_children() -> None:
    wf = Pipeline((foo, bar, baz, combine))
    wf = prune_nexus_domain_types(
        wf,
        targets_per_run=[A],
        run_types=[SampleRun],
    )
    graph = wf.underlying_graph
    assert A[SampleRun] in graph
    assert B[SampleRun] not in graph
    assert C[SampleRun] not in graph
    assert D[SampleRun] not in graph
    assert E not in graph
    assert A[BackgroundRun] not in graph
    assert B[BackgroundRun] not in graph
    assert C[BackgroundRun] not in graph
    assert D[BackgroundRun] not in graph


def test_prune_workflow_by_non_parametrized_type() -> None:
    wf = Pipeline((foo, bar, baz, combine))
    wf = prune_nexus_domain_types(
        wf,
        targets=(E,),
    )
    graph = wf.underlying_graph
    assert A[SampleRun] in graph
    assert A[BackgroundRun] in graph
    assert A[EmptyBeamRun] not in graph
    assert B[SampleRun] in graph
    assert B[BackgroundRun] in graph
    assert B[EmptyBeamRun] not in graph
    assert C[SampleRun] in graph
    assert C[BackgroundRun] not in graph
    assert C[EmptyBeamRun] not in graph
    assert D[SampleRun] not in graph
    assert D[BackgroundRun] not in graph
    assert D[EmptyBeamRun] not in graph
    assert E in graph


def test_pruning_nexus_workflow_requires_run_targets_if_run_types_given() -> None:
    wf = GenericNeXusWorkflow()
    with pytest.raises(ValueError, match='targets_per_run'):
        prune_nexus_domain_types(
            wf,
            run_types=[SampleRun],
        )


def test_pruning_nexus_workflow_requires_monitor_targets_if_monitor_types_given() -> (
    None
):
    wf = GenericNeXusWorkflow()
    with pytest.raises(ValueError, match='targets_per_run_and_monitor'):
        prune_nexus_domain_types(
            wf,
            targets_per_run=[DetectorData],
            run_types=[SampleRun],
            monitor_types=[Monitor1, Monitor3],
        )
