# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from functools import singledispatch
from typing import Any, Protocol

import ipywidgets as widgets
import scipp as sc

from .parameter import (
    BinEdgesParameter,
    BooleanParameter,
    FilenameParameter,
    MultiFilenameParameter,
    Parameter,
    ParamWithOptions,
    StringParameter,
    VectorParameter,
)

_layout = widgets.Layout(width='80%')
_style = {
    'description_width': 'auto',
    'value_width': 'auto',
    'button_width': 'auto',
}


class EssWidget(Protocol):
    """Protocol for ESS widgets.

    All widgets should have a `value` property that returns the value of the widget.
    It can be composed from multiple widgets.
    ```
    """

    @property
    def value(self) -> Any: ...


class LinspaceWidget(widgets.GridBox, widgets.ValueWidget):
    def __init__(self, dim: str, unit: str):
        super().__init__()

        self.fields = {
            'dim': widgets.Label(value=dim, description='dim'),
            'start': widgets.FloatText(description='start', value=0.0),
            'end': widgets.FloatText(description='end', value=1.0),
            'num': widgets.IntText(description='num', value=100),
            'unit': widgets.Label(description='unit', value=unit),
        }
        self.children = [
            widgets.Label(value="Select range:"),
            self.fields['dim'],
            self.fields['unit'],
            self.fields['start'],
            self.fields['end'],
            self.fields['num'],
        ]

    @property
    def value(self) -> sc.Variable:
        return sc.linspace(
            self.fields['dim'].value,
            self.fields['start'].value,
            self.fields['end'].value,
            self.fields['num'].value,
            unit=self.fields['unit'].value,
        )


class BoundsWidget(widgets.GridBox, widgets.ValueWidget):
    def __init__(self):
        super().__init__()

        self.fields = {
            'start': widgets.FloatText(description='start'),
            'end': widgets.FloatText(description='end'),
            'unit': widgets.Text(description='unit'),
        }
        self.children = [
            widgets.Label(value="Select bound:"),
            self.fields['unit'],
            self.fields['start'],
            self.fields['end'],
        ]

    @property
    def value(self):
        return (
            sc.scalar(self.fields['start'].value, unit=self.fields['unit']),
            sc.scalar(self.fields['end'].value, unit=self.fields['unit']),
        )


class VectorWidget(widgets.GridBox, widgets.ValueWidget):
    def __init__(self, variable: sc.Variable):
        super().__init__()

        self.fields = {
            "x": widgets.FloatText(description="x", value=variable.fields.x.value),
            "y": widgets.FloatText(description="y", value=variable.fields.y.value),
            "z": widgets.FloatText(description="z", value=variable.fields.z.value),
            "unit": widgets.Text(description="unit", value=str(variable.unit)),
        }
        self.children = [
            widgets.Label(value="(x, y, z) ="),
            self.fields['x'],
            self.fields['y'],
            self.fields['z'],
            self.fields['unit'],
        ]

    @property
    def value(self):
        return sc.vector(
            value=[
                self.fields['x'].value,
                self.fields['y'].value,
                self.fields['z'].value,
            ],
            unit=self.fields['unit'].value,
        )


@singledispatch
def create_parameter_widget(param: Parameter) -> widgets.Widget:
    """Create a widget for a parameter depending on the ``param`` type.

    If the type of the parameter is not supported, a text widget is returned.
    """
    return widgets.Text('', description=param.name, layout=_layout, style=_style)


@create_parameter_widget.register(VectorParameter)
def vector_parameter_widget(param: VectorParameter):
    return VectorWidget(param.default)


@create_parameter_widget.register(BooleanParameter)
def boolean_parameter_widget(param: BooleanParameter):
    name = param.name.split('.')[-1]
    description = param.description
    return widgets.Checkbox(
        value=param.default,
        description=name,
        tooltip=description,
        layout=_layout,
        style=_style,
    )


@create_parameter_widget.register(StringParameter)
def string_parameter_widget(param: StringParameter):
    name = param.name
    description = param.description
    if param.switchable:
        # TODO: Make switch widgets
        return widgets.Text(
            description=name, tooltip=description, layout=_layout, style=_style
        )
    else:
        return widgets.Text(
            value=param.default,
            description=name,
            tooltip=description,
            layout=_layout,
            style=_style,
        )


@create_parameter_widget.register(BinEdgesParameter)
def bin_edges_parameter_widget(param: BinEdgesParameter):
    dim = param.dim
    unit = param.unit
    return LinspaceWidget(dim, unit)


@create_parameter_widget.register(FilenameParameter)
def filename_parameter_widget(param: FilenameParameter):
    # TODO: Need to add the file upload widget
    return widgets.Text(
        description=param.name, layout=_layout, style=_style, value=param.default
    )


@create_parameter_widget.register(MultiFilenameParameter)
def multi_filename_parameter_widget(param: MultiFilenameParameter):
    # TODO: Need to add the file upload widget
    return widgets.Text(
        description=param.name, layout=_layout, style=_style, value=param.default
    )


@create_parameter_widget.register(ParamWithOptions)
def param_with_option_widget(param: ParamWithOptions):
    return widgets.Dropdown(
        description=param.name, options=param.options, layout=_layout, style=_style
    )
