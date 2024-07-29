# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import pytest
from ess.reduce.meta import ORCIDiD
from pydantic import BaseModel


class Model(BaseModel):
    orcid: ORCIDiD


def test_orcid_init_no_domain() -> None:
    m = Model(orcid='0000-0000-0000-0001')
    assert isinstance(m.orcid, str)
    assert m.orcid == 'https://orcid.org/0000-0000-0000-0001'


def test_orcid_init_valid_domain() -> None:
    m = Model(orcid='https://orcid.org/0000-0000-0000-0001')
    assert isinstance(m.orcid, str)
    assert m.orcid == 'https://orcid.org/0000-0000-0000-0001'


def test_orcid_init_invalid_domain() -> None:
    with pytest.raises(ValueError, match='Invalid ORCID URL'):
        Model(orcid='https://my-orcid.org/0000-0000-0000-0001')


def test_orcid_init_invalid_structure() -> None:
    with pytest.raises(ValueError, match='Incorrect structure'):
        Model(orcid='0000-0000-0001')
    with pytest.raises(ValueError, match='Incorrect structure'):
        Model(orcid='0000-0000-0001-123')


def test_orcid_init_invalid_checksum() -> None:
    with pytest.raises(ValueError, match='Checksum does not match'):
        Model(orcid='0000-0000-0000-0000')


def test_orcid_serialize() -> None:
    m = Model(orcid='0000-0000-0000-0001')
    res = m.model_dump()
    assert res == {'orcid': 'https://orcid.org/0000-0000-0000-0001'}
