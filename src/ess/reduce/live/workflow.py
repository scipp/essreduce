# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Tools for creating live data reduction workflows for Beamlime."""

from pathlib import Path
from typing import NewType, TypeVar

import sciline
import scipp as sc
import scippnexus as snx

from ess.reduce import streaming
from ess.reduce.nexus import types as nt
from ess.reduce.nexus.json_nexus import JSONGroup

JSONEventData = NewType('JSONEventData', dict[str, JSONGroup])


def _load_json_event_data(name: str, nxevent_data: JSONEventData) -> sc.DataArray:
    return snx.Group(nxevent_data[name], definitions=snx.base_definitions())[()]


def load_json_event_data_for_sample_run(
    name: nt.NeXusName[nt.Component], nxevent_data: JSONEventData
) -> nt.NeXusData[nt.Component, nt.SampleRun]:
    return nt.NeXusData[nt.Component, nt.SampleRun](
        _load_json_event_data(name, nxevent_data)
    )


def load_json_event_data_for_sample_transmission_run(
    name: nt.NeXusName[nt.Component], nxevent_data: JSONEventData
) -> nt.NeXusData[nt.Component, nt.TransmissionRun[nt.SampleRun]]:
    return nt.NeXusData[nt.Component, nt.TransmissionRun[nt.SampleRun]](
        _load_json_event_data(name, nxevent_data)
    )


T = TypeVar('T', bound='LiveWorkflow')


class LiveWorkflow:
    """A workflow class that fulfills Beamlime's LiveWorkflow protocol."""

    def __init__(
        self,
        *,
        streamed: streaming.StreamProcessor,
        outputs: dict[str, sciline.typing.Key],
    ) -> None:
        self._streamed = streamed
        self._outputs = outputs

    @classmethod
    def from_workflow(
        cls: type[T],
        *,
        workflow: sciline.Pipeline,
        accumulators: dict[sciline.typing.Key, streaming.Accumulator],
        outputs: dict[str, sciline.typing.Key],
        run_type: type[nt.RunType],
        nexus_filename: Path,
    ) -> T:
        """
        Create a live workflow from a base workflow and other parameters.

        Parameters
        ----------
        workflow:
            Base workflow to use for live data reduction.
        accumulators:
            Accumulators forwarded to the stream processor.
        outputs:
            Mapping from output names to keys in the workflow. The keys correspond to
            workflow results that will be computed.
        run_type:
            Type of the run to process. This defines which run is the dynamic run being
            processed. The NeXus template file will be set as the filename for this run.
        nexus_filename:
            Path to the NeXus file to process.

        Returns
        -------
        :
            Live workflow object.
        """

        workflow = workflow.copy()
        if run_type is nt.SampleRun:
            workflow.insert(load_json_event_data_for_sample_run)
        elif run_type is nt.TransmissionRun[nt.SampleRun]:
            workflow.insert(load_json_event_data_for_sample_transmission_run)
        else:
            raise NotImplementedError(f"Run type {run_type} not supported yet.")
        workflow[nt.Filename[run_type]] = nexus_filename
        streamed = streaming.StreamProcessor(
            base_workflow=workflow,
            dynamic_keys=(JSONEventData,),
            target_keys=outputs.values(),
            accumulators=accumulators,
        )
        return cls(streamed=streamed, outputs=outputs)

    def __call__(
        self, nxevent_data: dict[str, JSONGroup], nxlog: dict[str, JSONGroup]
    ) -> dict[str, sc.DataArray]:
        """
        Implements the __call__ method required by the LiveWorkflow protocol.

        Parameters
        ----------
        nxevent_data:
            NeXus event data.
        nxlog:
            NeXus log data. WARNING: This is currently not used.

        Returns
        -------
        :
            Dictionary of computed and plottable results.
        """
        # Beamlime passes full path, but the workflow only needs the name of the monitor
        # or detector group.
        nxevent_data = {
            key.lstrip('/').split('/')[2]: value for key, value in nxevent_data.items()
        }
        results = self._streamed.add_chunk({JSONEventData: nxevent_data})
        return {name: results[key] for name, key in self._outputs.items()}
