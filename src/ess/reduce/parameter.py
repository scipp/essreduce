# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2023 Scipp contributors (https://github.com/scipp)
from dataclasses import dataclass
from typing import Callable, Generic, TypeVar

T = TypeVar('T')


@dataclass
class Parameter(Generic[T]):
    name: str
    description: str
    default: T
    validators: list[Callable[[T], bool]]


@dataclass
class ParamWithOptions(Parameter[T]):
    options: list[T]


@dataclass
class ParamWithBounds(Parameter[T]):
    bounds: tuple[T, T]


@dataclass
class ScalarParameter(Parameter[T]):
    """Fixed unit displayed in widget"""

    unit: str


@dataclass
class ScalarParamWithUnitOptions(Parameter[T]):
    """User can select between compatible units"""

    unit_options: list[str]
