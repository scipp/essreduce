# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from ess.reduce import data
from ess.reduce.nexus import generic_types as gt
from ess.reduce.nexus import types as ct  # common types
from ess.reduce.nexus.generic_workflow import (
    GenericNeXusWorkflow,
    LoadDetectorWorkflow,
    LoadMonitorWorkflow,
)


def test_load_monitor_workflow() -> None:
    wf = LoadMonitorWorkflow()
    wf[gt.Filename[gt.SampleRun]] = data.loki_tutorial_sample_run_60250()
    wf[gt.NeXusMonitorName[gt.Monitor1]] = 'monitor_1'
    da = wf.compute(gt.MonitorData[gt.SampleRun, gt.Monitor1])
    assert 'position' in da.coords
    assert 'source_position' in da.coords
    assert da.bins is not None
    assert da.dims == ('event_time_zero',)


def test_load_detector_workflow() -> None:
    wf = LoadDetectorWorkflow()
    wf[gt.Filename[gt.SampleRun]] = data.loki_tutorial_sample_run_60250()
    wf[ct.NeXusDetectorName] = 'larmor_detector'
    da = wf.compute(gt.DetectorData[gt.SampleRun])
    assert 'position' in da.coords
    assert 'sample_position' in da.coords
    assert 'source_position' in da.coords
    assert da.bins is not None
    assert da.dims == ('detector_number',)


def test_generic_nexus_workflow() -> None:
    wf = GenericNeXusWorkflow()
    wf[gt.Filename[gt.SampleRun]] = data.loki_tutorial_sample_run_60250()
    wf[gt.NeXusMonitorName[gt.Monitor1]] = 'monitor_1'
    wf[ct.NeXusDetectorName] = 'larmor_detector'
    da = wf.compute(gt.DetectorData[gt.SampleRun])
    assert 'position' in da.coords
    assert 'sample_position' in da.coords
    assert 'source_position' in da.coords
    assert da.bins is not None
    assert da.dims == ('detector_number',)
    da = wf.compute(gt.MonitorData[gt.SampleRun, gt.Monitor1])
    assert 'position' in da.coords
    assert 'source_position' in da.coords
    assert da.bins is not None
    assert da.dims == ('event_time_zero',)
