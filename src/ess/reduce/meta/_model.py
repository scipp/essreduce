from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, EmailStr

from ._orcid import ORCIDiD


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
    doi: str | None = None


class Person(BaseModel):
    name: str
    orcid: ORCIDiD | None = None

    corresponding: bool = False
    owner: bool = True
    role: str | None = None

    address: str | None = None
    email: EmailStr | None = None
    affiliation: str | None = None


class Software(BaseModel):
    name: str
    version: str
    url: str | None = None
