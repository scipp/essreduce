# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import Any, Protocol

from .parameter import Parameter

Node = Any


class Workflow(Protocol):
    @property
    def typical_outputs(self) -> tuple[Node]:
        """Return a tuple of outputs that are used regularly."""

    @property
    def possible_outputs(self) -> tuple[Node]:
        """All possible outputs."""

    def parameters(self, outputs: tuple[Node]) -> dict[Node, Parameter]:
        """Return a dictionary of parameters for the workflow."""

    def __setitem__(self, node: Node, value: Any):
        """Set a value for a node."""

    def compute(self, outputs: tuple[Node]) -> dict[Node, Any]:
        """Run the workflow to compute outputs."""
