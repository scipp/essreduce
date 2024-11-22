# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""NeXus utilities.

This module defines functions and domain types that can be used
to build Sciline pipelines for simple workflows.
If multiple different kinds of files (e.g., sample and background runs)
are needed, custom types and providers need to be defined to wrap
the basic ones here.

Providers / functions are available from here directly.
The submodule :mod:`types` defines all domain types.
"""

from . import types
from .workflow import GenericNeXusWorkflow
from ._nexus_loader import (
    load_data,
    group_event_data,
    load_component,
    load_all_components,
    compute_component_position,
    extract_signal_data_array,
)

__all__ = [
    'types',
    'group_event_data',
    'load_all_components',
    'load_data',
    'load_component',
    'compute_component_position',
    'extract_signal_data_array',
    'GenericNeXusWorkflow',
]
