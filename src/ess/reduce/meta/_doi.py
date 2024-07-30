# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from pydantic import GetCoreSchemaHandler, GetJsonSchemaHandler, json_schema
from pydantic_core import core_schema

_DOI_DOMAIN: str = 'https://doi.org/'
_INPUT_DOI_DOMAINS = (
    _DOI_DOMAIN,
    'doi:',
    'https://dx.doi.org/',
    'http://dx.doi.org/',
    'http://doi.org/',
)


if TYPE_CHECKING:
    # Make type checkers happy with assigning strings to doi fields.
    DOI = Annotated[str, ...]
else:

    class DOI:
        """A DOI field type for Pydantic models.

        Validates structural correctness of a DOI.
        Does not check whether the DOI can be resolved.

        Examples
        --------

            >>> from ess.reduce.meta import DOI
            >>> from pydantic import BaseModel
            >>> class Model(BaseModel):
            ...     doi: DOI
            >>> m = Model(doi='10.1000/demo_DOI')
            >>> m.doi
            'https://doi.org/10.1000/demo_DOI'

        The domain or a 'doi:' prefix can be specified manually.
        All the following are equivalent and are formatted with a
        'https://doi.org/' domain:

            >>> a = Model(doi='10.1000/demo_DOI')
            >>> b = Model(doi='https://doi.org/10.1000/demo_DOI')
            >>> c = Model(doi='https://dx.doi.org/10.1000/demo_DOI')
            >>> d = Model(doi='doi:10.1000/demo_DOI')
        """

        @classmethod
        def __get_pydantic_core_schema__(
            cls, _source_type: type[Any], _handler: GetCoreSchemaHandler
        ) -> core_schema.CoreSchema:
            return core_schema.no_info_after_validator_function(
                cls._validate,
                core_schema.str_schema(),
            )

        @classmethod
        def __get_pydantic_json_schema__(
            cls,
            core_schema_: core_schema.CoreSchema,
            handler: GetJsonSchemaHandler,
        ) -> json_schema.JsonSchemaValue:
            field_schema = handler(core_schema_)
            field_schema.update(type='string', format='uri')
            return field_schema

        @classmethod
        def _validate(cls, value: str, /) -> str:
            return _parse_doi(value)


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

    return _DOI_DOMAIN + value
