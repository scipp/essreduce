# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

from contextlib import nullcontext
from pathlib import Path
from typing import BinaryIO, ContextManager, NewType, Optional, Union

import scipp as sc
import scippnexus as snx

FilePath = NewType('FilePath', Path)
"""Full path to a NeXus file on disk."""
NeXusFile = NewType('NeXusFile', BinaryIO)
"""An open NeXus file.

Can be any file handle for reading binary data.

Note that this cannot be used as a parameter in Sciline as there are no
concrete implementations of ``BinaryIO``.
The type alias is provided for callers of load functions outside of pipelines.
"""
NeXusGroup = NewType('NeXusGroup', snx.Group)
"""A ScippNexus group in an open file."""

InstrumentName = NewType('InstrumentName', str)
"""Name of an instrument in a NeXus file."""
DetectorName = NewType('DetectorName', str)
"""Name of a detector (bank) in a NeXus file."""

RawDetector = NewType('RawDetector', sc.DataArray)
"""A Scipp DataArray containing raw data from a detector."""
RawMonitor = NewType('RawMonitor', sc.DataArray)
"""A Scipp DataArray containing raw data from a monitor."""


def load_detector(
    file_path: Union[FilePath, NeXusFile, NeXusGroup],
    instrument_name: Optional[InstrumentName] = None,
    detector_name: Optional[DetectorName] = None,
) -> RawDetector:
    with _open_nexus_file(file_path) as f:
        return RawDetector(
            f[f'entry/{instrument_name}/{detector_name}/{detector_name}_events'][...]
        )


def _open_nexus_file(
    file_path: Union[FilePath, NeXusFile, NeXusGroup]
) -> ContextManager:
    if isinstance(file_path, NeXusGroup.__supertype__):
        return nullcontext(file_path)
    return snx.File(file_path)
