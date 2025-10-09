# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Data files bundled with ESSreduce.

This module requires the Pooch package which is not a hard dependency of ESSreduce.
It has to be installed separately with either pip or conda.
"""

from ._registry import Entry, LocalRegistry, PoochRegistry, Registry, make_registry

__all__ = [
    'Entry',
    'LocalRegistry',
    'PoochRegistry',
    'Registry',
    'make_registry',
]
