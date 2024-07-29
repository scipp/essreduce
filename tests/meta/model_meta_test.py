# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import pytest
from ess.reduce.meta import Person


def test_person_construct_valid() -> None:
    person = Person(
        name='Jane Doe',
        email='john.doe@example.com',
        orcid='https://orcid.org/0000-0000-0000-0001',
    )
    assert person.name == 'Jane Doe'
    assert person.email == 'john.doe@example.com'
    assert person.orcid == 'https://orcid.org/0000-0000-0000-0001'


def test_person_edit_valid() -> None:
    person = Person(
        name='Jane Doe',
        email='john.doe@example.com',
        orcid='https://orcid.org/0000-0000-0000-0001',
    )
    person.orcid = '0123-9876-4554-0003'
    person.email = None
    assert person.orcid == 'https://orcid.org/0123-9876-4554-0003'
    assert person.email is None


def test_person_construct_extra_not_allowed() -> None:
    with pytest.raises(ValueError, match='Extra inputs are not permitted'):
        Person(name='John Doe', extra='not-allowed')  # type: ignore[call-arg]


def test_person_construct_invalid() -> None:
    with pytest.raises(ValueError, match='Invalid ORCID'):
        Person(name='John Doe', orcid='0123-9876-4554-0000')


def test_person_edit_invalid() -> None:
    person = Person(
        name='Jane Doe',
        email='john.doe@example.com',
        orcid='https://orcid.org/0000-0000-0000-0001',
    )
    with pytest.raises(ValueError, match='Invalid ORCID'):
        person.orcid = '0123-9876-4554-0000'
