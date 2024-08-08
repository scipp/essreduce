# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass
from enum import Enum
from typing import Generic, TypeVar

import scipp as sc
from sciline.typing import Key

T = TypeVar('T')
C = TypeVar('C', bound='Parameter')


class KeepDefaultType:
    pass


keep_default = KeepDefaultType()


@dataclass
class Parameter(Generic[T]):
    """Interface between workflow parameters and widgets.

    .. versionadded:: RELEASE_PLACEHOLDER
    """

    name: str
    description: str
    default: T
    optional: bool = False
    """If True, widget has radio buttons switch between "None" and param widget."""
    switchable: bool = False
    """If True, widget has checkbox to enable/disable parameter."""

    def with_default(self: C, default: T | KeepDefaultType = keep_default) -> C:
        if default == keep_default:
            return self
        # TODO I think some subclasses currently cannot be instantiated with this method
        return type(self)(
            self.name, self.description, default, self.optional, self.switchable
        )

    @classmethod
    def from_type(
        cls: type[C],
        t: type[T],
        default: T | None = None,
        optional: bool = False,
        switchable: bool = False,
    ) -> C:
        # TODO __doc__ not correct when using NewType
        # TODO __doc__ not correct when using Generic
        # use sciline type->string helper
        return cls(
            name=str(t),
            description=t.__doc__,
            default=default,
            optional=optional,
            switchable=switchable,
        )


@dataclass(kw_only=True)
class ParamWithOptions(Parameter[T]):
    options: Enum

    @classmethod
    def from_enum(cls: type[C], t: type[T], default: T) -> C:
        return cls(name=str(t), description=t.__doc__, options=t, default=default)


@dataclass
class FilenameParameter(Parameter[str]):
    """Widget for entering a filename or selecting one in a file dialog."""

    # TODO need specifics for different file types, nexus, ...


@dataclass
class MultiFilenameParameter(Parameter[tuple[str, ...]]):
    """Widget for entering multiple filenames or selecting multiple in a file dialog."""


@dataclass(kw_only=True)
class BinEdgesParameter(Parameter[sc.Variable]):
    """Widget for entering bin edges."""

    # dim and unit displayed in widget to provide context of numbers
    dim: str
    unit: str

    def __init__(self, t: type[T], dim: str, unit: str):
        self.dim = dim
        self.unit = unit
        super().__init__(name=str(t), description=t.__doc__, default=None)


@dataclass
class BooleanParameter(Parameter[bool]):
    pass


@dataclass
class StringParameter(Parameter[str]):
    pass


@dataclass(kw_only=True)
class ParamWithBounds(Parameter[T]):
    bounds: tuple[T, T]


@dataclass(kw_only=True)
class ScalarParameter(Parameter[T]):
    """Fixed unit displayed in widget"""

    unit: str


@dataclass(kw_only=True)
class ScalarParamWithUnitOptions(Parameter[T]):
    """User can select between compatible units"""

    unit_options: list[str]


@dataclass(kw_only=True)
class VectorParameter(Parameter[sc.Variable]):
    """Widget for entering a vector."""


class ParameterRegistry(MutableMapping):
    def __init__(self):
        self._parameters = {}

    def __getitem__(self, key: Key) -> Parameter:
        return self._parameters[key]

    def __setitem__(self, key: Key, value: Parameter):
        self._parameters[key] = value

    def __delitem__(self, key: Key):
        del self._parameters[key]

    def __iter__(self):
        return iter(self._parameters)

    def __len__(self) -> int:
        return len(self._parameters)


parameter_registry = ParameterRegistry()


parameter_mappers = {}
