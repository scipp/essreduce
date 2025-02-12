# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
# ruff: noqa: E402, F401
from functools import singledispatch
from typing import Any, Protocol

import ipywidgets as widgets

from ..parameter import (
    BinEdgesParameter,
    BooleanParameter,
    FilenameParameter,
    MultiFilenameParameter,
    MultiStringParameter,
    Parameter,
    ParamWithBounds,
    ParamWithOptions,
    StringParameter,
    Vector2dParameter,
    Vector3dParameter,
)
from ._binedges_widget import BinEdgesWidget
from ._bounds_widget import BoundsWidget
from ._config import default_layout, default_style
from ._filename_widget import FilenameWidget, MultiFilenameWidget
from ._linspace_widget import LinspaceWidget
from ._optional_widget import OptionalWidget
from ._spinner import Spinner
from ._string_widget import MultiStringWidget, StringWidget
from ._switchable_widget import SwitchWidget
from ._vector_widget import VectorWidget


class EssWidget(Protocol):
    """Protocol for ESS widgets.

    All widgets should have a `value` property that returns the value of the widget.
    It can be composed from multiple widgets.
    """

    @property
    def value(self) -> Any: ...


from collections.abc import Callable
from functools import wraps


def switchable_widget(
    func: Callable[[Parameter], widgets.Widget],
) -> Callable[[Parameter], widgets.Widget]:
    """Wrap a widget in a switchable widget."""

    @wraps(func)
    def wrapper(param: Parameter) -> widgets.Widget:
        widget = func(param)
        if param.switchable:
            return SwitchWidget(widget, name=param.name)
        return widget

    return wrapper


def optional_widget(
    func: Callable[[Parameter], widgets.Widget],
) -> Callable[[Parameter], widgets.Widget]:
    """Wrap a widget in a optional widget."""

    @wraps(func)
    def wrapper(param: Parameter) -> widgets.Widget:
        widget = func(param)
        if param.optional:
            return OptionalWidget(widget, name=param.name)
        return widget

    return wrapper


@switchable_widget
@optional_widget  # optional_widget should be applied first
@singledispatch
def create_parameter_widget(param: Parameter) -> widgets.Widget:
    """Create a widget for a parameter depending on the ``param`` type.

    If the type of the parameter is not supported, a text widget is returned.
    """
    return widgets.Text(
        '', description=param.name, layout=default_layout, style=default_style
    )


@create_parameter_widget.register(BooleanParameter)
def boolean_parameter_widget(param: BooleanParameter):
    name = param.name.split('.')[-1]
    description = param.description
    return widgets.Checkbox(
        value=param.default,
        description=name,
        tooltip=description,
        layout=default_layout,
        style=default_style,
    )


@create_parameter_widget.register(StringParameter)
def string_parameter_widget(param: StringParameter):
    return StringWidget(
        description=param.name,
        value=param.default,
        layout=default_layout,
        style=default_style,
    )


@create_parameter_widget.register(MultiStringParameter)
def multi_string_parameter_widget(param: MultiStringParameter):
    return MultiStringWidget(
        description=param.name,
        value=param.default,
        layout=default_layout,
        style=default_style,
    )


@create_parameter_widget.register(FilenameParameter)
def filename_parameter_widget(param: FilenameParameter):
    return FilenameWidget(
        description=param.name,
        value=param.default,
        layout=default_layout,
        style=default_style,
    )


@create_parameter_widget.register(MultiFilenameParameter)
def multi_filename_parameter_widget(param: MultiFilenameParameter):
    return MultiFilenameWidget(
        description=param.name,
        value=param.default,
        layout=default_layout,
        style=default_style,
    )


@create_parameter_widget.register(ParamWithOptions)
def param_with_option_widget(param: ParamWithOptions):
    return widgets.Dropdown(
        description=param.name,
        options=param.options,
        layout=default_layout,
        style=default_style,
    )


@create_parameter_widget.register(ParamWithBounds)
def param_with_bounds_widget(param: ParamWithBounds):
    return BoundsWidget()


@create_parameter_widget.register(BinEdgesParameter)
def bin_edges_parameter_widget(param: BinEdgesParameter):
    return BinEdgesWidget(
        name=param.name,
        dim=param.dim,
        start=param.start,
        stop=param.stop,
        nbins=param.nbins,
        unit=param.unit,
        log=param.log,
    )


@create_parameter_widget.register(Vector2dParameter)
def vector_2d_parameter_widget(param: Vector2dParameter):
    return VectorWidget(name=param.name, variable=param.default, components="xy")


@create_parameter_widget.register(Vector3dParameter)
def vector_3d_parameter_widget(param: Vector3dParameter):
    return VectorWidget(name=param.name, variable=param.default, components="xyz")


__all__ = [
    'BinEdgesWidget',
    'BoundsWidget',
    'EssWidget',
    'FilenameWidget',
    'LinspaceWidget',
    'MultiFilenameWidget',
    'OptionalWidget',
    'Spinner',
    'SwitchWidget',
    'VectorWidget',
    'create_parameter_widget',
]
