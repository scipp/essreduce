"""NeXus domain types for use with Sciline."""

from pathlib import Path
from typing import (
    BinaryIO,
    NewType,
)

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

NeXusDetectorName = NewType('NeXusDetectorName', str)
"""Name of a detector (bank) in a NeXus file."""
NeXusEntryName = NewType('NeXusEntryName', str)
"""Name of an entry in a NeXus file."""
NeXusMonitorName = NewType('NeXusMonitorName', str)
"""Name of a monitor in a NeXus file."""
NeXusSourceName = NewType('NeXusSourceName', str)
"""Name of a source in a NeXus file."""

RawDetectorData = NewType('RawDetectorData', sc.DataArray)
"""Data extracted from a RawDetector."""
RawMonitorData = NewType('RawMonitorData', sc.DataArray)
"""Data extracted from a RawMonitor."""

NeXusDetector = NewType('NeXusDetector', sc.DataGroup)
"""Full raw data from a NeXus detector."""
NeXusMonitor = NewType('NeXusMonitor', sc.DataGroup)
"""Full raw data from a NeXus monitor."""
NeXusSample = NewType('NeXusSample', sc.DataGroup)
"""Raw data from a NeXus sample."""
NeXusSource = NewType('NeXusSource', sc.DataGroup)
"""Raw data from a NeXus source."""
NeXusEventData = NewType('NeXusEventData', sc.DataArray)
"""Data array loaded from a NeXus NXevent_data group."""
