# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
"""Metadata utilities.

.. rubric:: Models

Pydantic models that comprise the intermediate representation.

.. autosummary::
    :toctree:

    Beamline
    Experiment
    Person
    Software

.. rubric:: Miscellany

Various helpers.

.. autosummary::
    :toctree:

    DOI
    ORCIDiD
"""

from ._model import Beamline, Experiment, Person, Software
from ._orcid import ORCIDiD
from ._doi import DOI

__all__ = ['Beamline', 'DOI', 'Experiment', 'ORCIDiD', 'Person', 'Software']
