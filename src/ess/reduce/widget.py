from functools import singledispatch

import ipywidgets as widgets
import scipp as sc
from ess.reduce import parameter

_layout = widgets.Layout(width='80%')
_style = {
    'description_width': 'auto',
    'value_width': 'auto',
    'button_width': 'auto',
}


class LinspaceWidget(widgets.GridBox, widgets.ValueWidget):
    def __init__(self, dim, unit):
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
    def value(self):
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
    def __init__(self, variable):
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
def create_parameter_widget(param):
    return widgets.Text('', layout=_layout, style=_style)


@create_parameter_widget.register(parameter.VectorParameter)
def _(param):
    return VectorWidget(param.default)


@create_parameter_widget.register(parameter.BooleanParameter)
def _(param):
    name = param.name.split('.')[-1]
    description = param.description
    if param.switchable:
        # TODO: Make switch widgets
        return widgets.Checkbox(
            description=name, tooltip=description, layout=_layout, style=_style
        )
    else:
        return widgets.Checkbox(
            value=param.default,
            description=name,
            tooltip=description,
            layout=_layout,
            style=_style,
        )


@create_parameter_widget.register(parameter.StringParameter)
def _(param):
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


@create_parameter_widget.register(parameter.BinEdgesParameter)
def _(param):
    dim = param.dim
    unit = param.unit
    return LinspaceWidget(dim, unit)


@create_parameter_widget.register(parameter.FilenameParameter)
def _(param):
    # TODO: Need to add the file upload widget
    return widgets.Text(
        description=param.name, layout=_layout, style=_style, value=param.default
    )


@create_parameter_widget.register(parameter.MultiFilenameParameter)
def _(param):
    # TODO: Need to add the file upload widget
    return widgets.Text(
        description=param.name, layout=_layout, style=_style, value=param.default
    )


@create_parameter_widget.register(parameter.ParamWithOptions)
def _(param):
    return widgets.Dropdown(
        description=param.name, options=param.options, layout=_layout, style=_style
    )
