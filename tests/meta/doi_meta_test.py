# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import pytest
from ess.reduce.meta import DOI
from pydantic import BaseModel


class Model(BaseModel):
    doi: DOI


def test_doi_init_no_domain() -> None:
    m = Model(doi='10.1000/182')
    assert isinstance(m.doi, str)
    assert m.doi == 'https://doi.org/10.1000/182'


@pytest.mark.parametrize('domain', ['https://doi.org/', 'https://dx.doi.org/', 'doi:'])
def test_doi_init_valid_prefix(domain: str) -> None:
    m = Model(doi=domain + '10.1000/demo_DOI')
    assert isinstance(m.doi, str)
    assert m.doi == 'https://doi.org/10.1000/demo_DOI'


def test_doi_init_invalid_domain() -> None:
    with pytest.raises(ValueError, match='Invalid DOI domain'):
        Model(doi='https://my-doi.org/10.1000/demo_DOI')


def test_doi_init_invalid_prefix() -> None:
    with pytest.raises(ValueError, match="Must start with '10.'"):
        Model(doi='1023/wrong-DOI')
    with pytest.raises(ValueError, match="Must start with '10.'"):
        Model(doi='https://doi.org/1023/wrong-DOI')


def test_doi_pydantic_model_serialize() -> None:
    m = Model(doi='https://doi.org/10.1000/demo_DOI')
    res = m.model_dump()
    assert res == {'doi': 'https://doi.org/10.1000/demo_DOI'}
