# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from datetime import datetime
from typing import Annotated

import pydantic
import pydantic_core

from ._doi import DOI
from ._orcid import ORCIDiD


class BaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        # If a user added extra fields, they would be ignored in all representations.
        # So forbid extra fields to help catch typos.
        extra='forbid',
        # Metadata models may be edited throughout a workflow.
        # Make sure that models are always valid.
        validate_assignment=True,
    )


class Beamline(BaseModel):
    """A beamline (instrument) at a facility.

    Examples
    --------
    For `Amor <https://www.psi.ch/en/sinq/amor>`_, we can use

        >>> from ess.reduce.meta import Beamline
        >>> amor = Beamline(
        ...     name='Amor',
        ...     facility='SINQ',
        ...     site='PSI',
        ...     revision='1',
        ... )
    """

    name: str
    """The name of the beamline / instrument."""
    facility: str | None = None
    """The facility of the beamline."""
    site: str | None = None
    """The site where the facility is located.

    Omit if ``site`` is the same as ``facility`` (e.g., at ESS).
    """
    revision: str | None = None
    """A version identifier of the beamline."""


class Experiment(BaseModel):
    """An experiment at a beamline."""

    title: str
    """The title of the experiment."""
    beamline: Beamline
    """The beamline that the experiment was conducted at."""
    start_date: datetime | None = None
    """The data and time when the experiment was started.

    Make sure to specify a timezone.
    """
    proposal_id: str | None = None
    """ID of the associated proposal for this experiment."""
    doi: DOI | None = None
    """A DOI for this experiment or proposal.

    This DOI should refer to the whole experiment / proposal,
    _not_ individual data files or datasets.
    """


class Person(BaseModel):
    """A person."""

    name: str
    """Name of the person in free-form.

    Use whichever format is preferred.
    """
    orcid: ORCIDiD | None = None
    """The ORCID iD of the person."""

    corresponding: bool = False
    """Determines whether this person is a contact for experiment / analysis.

    If ``True``, this person is a "corresponding author" or "contact" for
    the associated experiment or data.
    This field may determine how and whether this person is written to the output.
    """
    owner: bool = True
    """Determines whether this person owns the associated data.

    This field may determine how and whether this person is written to the output.
    """
    role: str | None = None
    """The role of this person in the experiment / analysis.

    This field is free-form but some output representations use
    a concrete list of allowed roles.
    See, e.g.,

    - `NXuser <https://manual.nexusformat.org/classes/base_classes/NXuser.html>`_
    - CIF roles: search for ``save_audit_author_role.role`` in
    `cif_core.dic <https://github.com/COMCIFS/cif_core/blob/fc3d75a298fd7c0c3cde43633f2a8616e826bfd5/cif_core.dic#L15195>`_
    """

    address: str | None = None
    """The physical address of this person."""
    email: pydantic.EmailStr | None = None
    """An e-mail address of this person."""
    affiliation: str | None = None
    """An affiliation of this person."""


class Software(BaseModel):
    """A piece of software used to produce some data."""

    name: str
    """The name of the software."""
    version: str
    """The version number of the software."""
    url: (
        Annotated[
            pydantic_core.Url,
            pydantic.UrlConstraints(allowed_schemes=['http', 'https', 'ftp']),
        ]
        | None
    ) = None
    """A URL that identifies the software."""
