# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

_DOI_DOMAIN: str = 'https://doi.org/'
_INPUT_DOI_DOMAINS = (
    _DOI_DOMAIN,
    'doi:',
    'https://dx.doi.org/',
    'http://dx.doi.org/',
    'http://doi.org/',
)


class _DOI:
    """A DOI

    Ensures that the DOI is structurally valid during initialization.
    Does not check whether the DOI can be resolved.

    This class can be used with Pydantic models.

    Examples
    --------
    DOI objects can be constructed from strings.
    When formatting, they always show a full URL that can be copied
    into a browser to look up the referenced object.

        >>> from ess.reduce.meta import DOI
        >>> doi = DOI('10.1000/demo_DOI')
        >>> doi
        https://doi.org/10.1000/demo_DOI

    The domain or a 'doi:' prefix can be specified manually.
    All the following are equivalent and are formatted with a
    'https://doi.org/' domain:

        >>> a = DOI('10.1000/demo_DOI')
        >>> b = DOI('https://doi.org/10.1000/demo_DOI')
        >>> c = DOI('https://doi.org/10.1000/demo_DOI')
        >>> d = DOI('doi:10.1000/demo_DOI')
    """

    __slots__ = ('_doi',)

    def __init__(self, doi: str | _DOI) -> None:
        if isinstance(doi, _DOI):
            self._doi: str = doi._doi
        else:
            self._doi = _parse_doi(doi)

    def __str__(self) -> str:
        return f'{_DOI_DOMAIN}{self._doi}'

    def __repr__(self) -> str:
        return f'DOI({self!s})'

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _DOI):
            return self._doi == other._doi
        if isinstance(other, str):
            return self._doi == _parse_doi(other)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __hash__(self) -> int:
        return hash(str(self._doi))

    @classmethod
    def __get_pydantic_core_schema__(
        cls, _source_type: Any, _handler: GetCoreSchemaHandler
    ) -> core_schema.CoreSchema:
        return core_schema.no_info_after_validator_function(
            _parse_pydantic,
            core_schema.union_schema(
                [core_schema.is_instance_schema(_DOI), core_schema.str_schema()]
            ),
            serialization=core_schema.plain_serializer_function_ser_schema(
                cls.__str__, info_arg=False, return_schema=core_schema.str_schema()
            ),
        )


if TYPE_CHECKING:
    # Make type checkers happy with assigning strings to doi fields.
    DOI = Annotated[str, ...]
else:
    DOI = _DOI


def _parse_pydantic(value: str | _DOI) -> _DOI:
    return _DOI(value)


def _parse_doi(value: str) -> str:
    for domain in _INPUT_DOI_DOMAINS:
        if value.startswith(domain):
            value = value.removeprefix(domain)
            break  # break because only one domain is allowed

    if 'http' in value:
        raise ValueError(
            f"Invalid DOI domain: '{value}'. "
            f"DOIs must start with one of {_INPUT_DOI_DOMAINS}."
        )

    prefix, suffix = value.split('/', 1)

    if not prefix.startswith('10.'):
        raise ValueError(f"Invalid DOI: {value}. Must start with '10.'")

    return value
