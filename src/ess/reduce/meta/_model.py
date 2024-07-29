# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from datetime import datetime

import pydantic

from ._doi import DOI
from ._orcid import ORCIDiD


class BaseModel(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        extra='forbid',
    )


class Beamline(BaseModel):
    name: str
    facility: str | None = None
    site: str | None = None
    revision: str | None = None


class Experiment(BaseModel):
    title: str
    beamline: Beamline
    start_date: datetime | None = None
    proposal_id: str | None = None
    doi: DOI | None = None


class Person(BaseModel):
    name: str
    orcid: ORCIDiD | None = None

    corresponding: bool = False
    owner: bool = True
    role: str | None = None

    address: str | None = None
    email: pydantic.EmailStr | None = None
    affiliation: str | None = None


class Software(BaseModel):
    name: str
    version: str
    url: str | None = None
