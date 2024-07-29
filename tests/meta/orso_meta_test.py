# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import platform
from datetime import datetime, timezone

import pytest
from ess.reduce import meta

orso_base = pytest.importorskip(
    'orsopy.fileio.base', reason='ORSO support requires orsopy'
)
orso_data_source = pytest.importorskip('orsopy.fileio.data_source')
orso_reduction = pytest.importorskip('orsopy.fileio.reduction')
orso = pytest.importorskip('ess.reduce.meta.orso')


def test_from_orso_unsupported_type() -> None:
    with pytest.raises(TypeError, match='converted from ORSO'):
        orso.from_orso(123)


def test_to_orso_unsupported_type() -> None:
    with pytest.raises(TypeError, match='converted to ORSO'):
        orso.to_orso(83)


def test_person_from_orso() -> None:
    orso_person = orso_base.Person(
        name='Jean Dupont',
        affiliation='École Polytechnique de Paris',
        contact='jean.dupont@example.com',
        comment='A test person',
    )
    person = orso.from_orso(orso_person)
    expected = meta.Person(
        name='Jean Dupont',
        email='jean.dupont@example.com',
        affiliation='École Polytechnique de Paris',
    )
    assert person == expected


@pytest.mark.parametrize('corresponding', [True, False])
@pytest.mark.parametrize('owner', [True, False])
def test_person_to_orso(corresponding: bool, owner: bool) -> None:
    person = meta.Person(
        name='Jean Dupont',
        orcid='0123-4567-8901-2346',
        corresponding=corresponding,
        owner=owner,
        role='PI',
        address='Rte de Saclay, 91120 Palaiseau, France',
        email='jean.dupont@example.com',
        affiliation='École Polytechnique de Paris',
    )
    orso_person = orso.to_orso(person)
    expected = orso_base.Person(
        name='Jean Dupont',
        affiliation='École Polytechnique de Paris',
        contact='jean.dupont@example.com',
        comment=None,
    )
    assert orso_person == expected


def test_experiment_from_orso() -> None:
    orso_experiment = orso_data_source.Experiment(
        title='Test Experiment 5',
        instrument='Amor',
        facility='SINQ',
        start_date=datetime(2024, 4, 18, 15, 9, 53, tzinfo=timezone.utc),
        proposalID='1234567890',
        doi='https://doi.org/10.1000/demo_DOI',
        probe='neutron',
    )
    experiment = orso.from_orso(orso_experiment)
    expected = meta.Experiment(
        title='Test Experiment 5',
        beamline=meta.Beamline(
            name='Amor',
            facility='SINQ',
        ),
        start_date=datetime(2024, 4, 18, 15, 9, 53, tzinfo=timezone.utc),
        proposal_id='1234567890',
        doi='10.1000/demo_DOI',
    )
    assert experiment == expected


def test_experiment_to_orso() -> None:
    experiment = meta.Experiment(
        title='Test Experiment 5',
        beamline=meta.Beamline(
            name='Amor',
            facility='SINQ',
            site='Paul Scherrer Institut',
            revision='1.0',
        ),
        start_date=datetime(2024, 4, 18, 15, 9, 53, tzinfo=timezone.utc),
        proposal_id='1234567890',
        doi='10.1000/demo_DOI',
    )
    orso_experiment = orso.to_orso(experiment)
    expected = orso_data_source.Experiment(
        title='Test Experiment 5',
        instrument='Amor',
        facility='SINQ',
        start_date=datetime(2024, 4, 18, 15, 9, 53, tzinfo=timezone.utc),
        proposalID='1234567890',
        doi='https://doi.org/10.1000/demo_DOI',
        probe='neutron',
    )
    assert orso_experiment == expected


def test_experiment_missing_field() -> None:
    experiment = meta.Experiment(
        title='Test Experiment 5',
        beamline=meta.Beamline(
            name='Amor',
        ),
        start_date=datetime(2024, 4, 18, 15, 9, 53, tzinfo=timezone.utc),
    )
    experiment.start_date = None
    with pytest.raises(
        ValueError,
        match="Field 'Experiment.start_date' must not be None when converting to ORSO.",
    ):
        orso.to_orso(experiment)


def test_software_from_orso() -> None:
    orso_software = orso_reduction.Software(
        name='ESSreduce',
        version='24.07.1',
        platform=platform.system(),
        comment='Generic reduction ESS tools',
    )
    software = orso.from_orso(orso_software)
    expected = meta.Software(
        name='ESSreduce',
        version='24.07.1',
        url=None,
    )
    assert software == expected


def test_software_to_orso() -> None:
    software = meta.Software(
        name='ESSreduce',
        version='24.07.1',
        url='https://scipp.github.io/essreduce/index.html',
    )
    orso_software = orso.to_orso(software)
    expected = orso_reduction.Software(
        name='ESSreduce',
        version='24.07.1',
        platform=platform.system(),
        comment=None,
    )
    assert orso_software == expected
