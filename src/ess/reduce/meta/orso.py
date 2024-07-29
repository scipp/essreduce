# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

import platform

from ._meta_conversions import Converter
from ._model import Beamline, Experiment, Person, Software

try:
    from orsopy.fileio import base as orso_base
    from orsopy.fileio import data_source as orso_data_source
    from orsopy.fileio import reduction as orso_reduction
except ModuleNotFoundError:
    raise ImportError(
        'ORSO support requires orsopy. Install it using `pip install orsopy`.'
    ) from None


_converter = Converter('ORSO')
# orsopy is untyped, so there is no use in listing overloads here.
from_orso = _converter.make_parse()
to_orso = _converter.make_represent()


@_converter.parser(orso_base.Person)
def _person_from_orso(_conv: Converter, person: orso_base.Person) -> Person:
    return Person(
        name=person.name,
        email=person.contact,
        affiliation=person.affiliation,
    )


@_converter.representer(Person)
def _person_to_orso(conv: Converter, person: Person) -> orso_base.Person:
    return orso_base.Person(
        name=conv.expect(person, 'name'),
        affiliation=conv.expect(person, 'affiliation'),
        contact=person.email,
    )


@_converter.parser(orso_data_source.Experiment)
def _experiment_from_orso(
    _conv: Converter, experiment: orso_data_source.Experiment
) -> Experiment:
    return Experiment(
        title=experiment.title,
        beamline=Beamline(
            name=experiment.instrument,
            facility=experiment.facility,
        ),
        start_date=experiment.start_date,
        proposal_id=experiment.proposalID,
        doi=experiment.doi,
    )


@_converter.representer(Experiment)
def _experiment_to_orso(
    conv: Converter, experiment: Experiment
) -> orso_data_source.Experiment:
    return orso_data_source.Experiment(
        title=conv.expect(experiment, 'title'),
        instrument=conv.expect(experiment, 'beamline.name'),
        facility=experiment.beamline.facility,
        start_date=conv.expect(experiment, 'start_date'),
        proposalID=experiment.proposal_id,
        doi=str(experiment.doi),
        probe='neutron',
    )


@_converter.parser(orso_reduction.Software)
def _software_from_orso(
    _conv: Converter, software: orso_reduction.Software
) -> Software:
    return Software(
        name=software.name,
        version=software.version,
    )


@_converter.representer(Software)
def _software_to_orso(conv: Converter, software: Software) -> orso_reduction.Software:
    return orso_reduction.Software(
        name=conv.expect(software, 'name'),
        version=software.version,
        platform=platform.system(),
    )
