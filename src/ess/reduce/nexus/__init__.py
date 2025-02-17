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
from ._nexus_loader import (
    compute_component_position,
    extract_signal_data_array,
    group_event_data,
    load_all_components,
    load_component,
    load_data,
)
from .workflow import GenericNeXusWorkflow

__all__ = [
    'GenericNeXusWorkflow',
    'compute_component_position',
    'extract_signal_data_array',
    'group_event_data',
    'load_all_components',
    'load_component',
    'load_data',
    'types',
]
