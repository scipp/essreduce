# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from __future__ import annotations

from collections.abc import Callable
from typing import Any, TypeAlias, TypeVar

from ._model import BaseModel


class Converter:
    """Registry and dispatcher for conversions to/from IR and a concrete representation.

    To use:

    1. Instantiate a ``Converter`` for each supported concrete representation.
    2. Register parsers for each model type using the
       ``Converter.parser`` decorator.
    3. Register representers for each model type using the
       ``Converter.representer`` decorator.
    4. Generate conversion functions using ``Converter.make_parse`` and
       ``Converter.make_represent``.

    The generated conversion function can be used to convert to/from models in the
    intermediate representation to the concrete representation.
    """

    def __init__(self, repr_name: str) -> None:
        """Create a new converter for a concrete representation.

        Parameters
        ----------
        repr_name:
            Name of the concrete representation.
        """
        self._parsers: dict[type, _Parser] = {}
        self._representers: dict[type, _Representer] = {}
        self._repr_name = repr_name

    def parser(self, typ: type) -> Callable[[_P], _P]:
        """Register a new parser."""

        def deco(func: _P) -> _P:
            self._parsers[typ] = func
            return func

        return deco

    def representer(self, typ: type[BaseModel]) -> Callable[[_R], _R]:
        """Register a new representer."""

        def deco(func: _R) -> _R:
            self._representers[typ] = func
            return func

        return deco

    def make_parse(self):
        """Make a function that can parse all known models."""

        def parse(model: Any, /) -> BaseModel:
            try:
                parser = self._parsers[type(model)]
            except KeyError:
                raise TypeError(
                    f"Objects of type '{type(model)}' cannot be "
                    f"converted from {self._repr_name}"
                ) from None
            return parser(self, model)

        parse.__doc__ = f"""Convert a model to the intermediate representation from {self._repr_name}.

    Parameters
    ----------
    model:
        The {self._repr_name} model to convert.

    Returns
    -------
    :
        The converted model in the intermediate representation.
"""  # noqa: E501

        return parse

    def make_represent(self):
        """Make a function that can represent all known models."""

        def represent(model: BaseModel, /) -> Any:
            try:
                representer = self._representers[type(model)]
            except KeyError:
                raise TypeError(
                    f"Objects of type '{type(model)}' cannot be "
                    f"converted to {self._repr_name}"
                ) from None
            return representer(self, model)

        represent.__doc__ = f"""Convert a model from the intermediate representation to {self._repr_name}.

    Parameters
    ----------
    model:
        The model in the intermediate representation to convert.

    Returns
    -------
    :
        The converted {self._repr_name} model.
"""  # noqa: E501

        return represent

    def expect(self, model: Any, field_name: str) -> Any:
        """Get an attribute of an object or raise if it is None."""
        value = model
        for segment in field_name.split('.'):
            if (value := getattr(value, segment, None)) is None:
                raise ValueError(
                    f"Field '{model.__class__.__name__}.{field_name}' must not be None "
                    f'when converting to {self._repr_name}.'
                )
        return value


_Parser: TypeAlias = Callable[[Converter, Any], BaseModel]
_Representer: TypeAlias = Callable[[Converter, BaseModel], Any]
_P = TypeVar('_P', bound=_Parser)
_R = TypeVar('_R', bound=_Representer)
