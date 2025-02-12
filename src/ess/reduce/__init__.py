# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, F401, I

import importlib.metadata

from . import nexus, time_of_flight, uncertainty

try:
    __version__ = importlib.metadata.version("essreduce")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.0.0"

del importlib

__all__ = ["nexus", "time_of_flight", "uncertainty"]
