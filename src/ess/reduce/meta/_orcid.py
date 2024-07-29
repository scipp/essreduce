from __future__ import annotations

from typing import TYPE_CHECKING, Annotated, Any

from pydantic import GetCoreSchemaHandler
from pydantic_core import core_schema

_ORCID_DOMAIN: str = 'https://orcid.org'

if TYPE_CHECKING:
    # Make type checkers happy with assigning strings to orcid fields.
    ORCIDiD = Annotated[str, ...]
else:

    class ORCIDiD:
        """An ORCID iD field type for Pydantic models.

        Validates structural correctness of an ORCID iD.
        See https://support.orcid.org/hc/en-us/articles/360006897674-Structure-of-the-ORCID-Identifier
        Does not check whether the id exists.

        Examples
        --------

            >>> from ess.reduce.meta import ORCIDiD
            >>> from pydantic import BaseModel
            >>> class Model(BaseModel):
            ...     orcid_id: ORCIDiD
            >>> m = Model(orcid_id='0000-0000-0000-0001')
            >>> m.orcid_id
            'https://orcid.org/0000-0000-0000-0001'

        Or equivalently with an explicit domain:

            >>> m = Model(orcid_id='https://orcid.org/0000-0000-0000-0001')
            >>> m.orcid_id
            'https://orcid.org/0000-0000-0000-0001'
        """

        @classmethod
        def __get_pydantic_core_schema__(
            cls, _source_type: type[Any], _handler: GetCoreSchemaHandler
        ) -> core_schema.CoreSchema:
            return core_schema.no_info_after_validator_function(
                cls._validate, core_schema.str_schema()
            )

        @classmethod
        def _validate(cls, value: str, /) -> str:
            return _parse_orcid_id(value)


def _parse_orcid_id(value: str) -> str:
    parts = value.rsplit('/', 1)
    if len(parts) == 2:
        domain, orcid_id = parts
        if domain != _ORCID_DOMAIN:
            # must be the correct ORCID URL
            raise ValueError(
                f"Invalid ORCID URL: '{domain}'. Must be '{_ORCID_DOMAIN}'"
            )
    else:
        (orcid_id,) = parts

    segments = orcid_id.split('-')
    if len(segments) != 4 or not all(len(s) == 4 for s in segments):
        # must have four blocks of numbers
        # and each block must have 4 digits
        raise ValueError(f"Invalid ORCID iD: '{orcid_id}'. Incorrect structure.")
    if _orcid_id_checksum(orcid_id) != orcid_id[-1]:
        # checksum must match the last digit
        raise ValueError(f"Invalid ORCID iD: '{orcid_id}'. Checksum does not match.")

    return f'{_ORCID_DOMAIN}/{orcid_id}'


def _orcid_id_checksum(orcid_id: str) -> str:
    total = 0
    for c in orcid_id[:-1].replace('-', ''):
        total = (total + int(c)) * 2
    result = (12 - total % 11) % 11
    return 'X' if result == 10 else str(result)
