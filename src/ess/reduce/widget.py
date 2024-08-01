from functools import partial, singledispatch

import ipywidgets as widgets
import scipp as sc
from ess.reduce import parameter
from ess.reduce.autowidget import AutoWidget

_layout = widgets.Layout(width='80%')
_style = {
    'description_width': 'auto',
    'value_width': 'auto',
    'button_width': 'auto',
}


@singledispatch
def create_parameter_widget(param):
    return widgets.Text('', layout=_layout, style=_style)


@create_parameter_widget.register(parameter.VectorParameter)
def _(param):
    entry = partial(
        widgets.FloatText,
        layout=widgets.Layout(width='5em'),
    )
    tri_tuple = AutoWidget(
        lambda _, x: x,
        (
            widgets.Label(value="(x, y, z) = "),
            AutoWidget(
                lambda x, y, z: (x, y, z),
                (
                    entry(value=param.default.fields.x.value),
                    entry(value=param.default.fields.y.value),
                    entry(value=param.default.fields.z.value),
                ),
            ),
        ),
    )
    return AutoWidget(
        sc.vector,
        (tri_tuple,),
        dict(  # noqa: C408
            unit=widgets.Text(
                description='unit of vector',
                value=str('m'),
                layout=widgets.Layout(width='10em'),
            )
        ),
        description=param.name,
        layout=widgets.Layout(border='0.5px solid'),
        child_layout=widgets.VBox,
    )


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
    return AutoWidget(
        lambda space, **kwargs: space(**kwargs),
        (),
        dict(  # noqa: C408
            space=widgets.Dropdown(
                default=sc.linspace,
                options=[sc.linspace, sc.geomspace],
                description='bin space',
            ),
            dim=widgets.Text(value=param.dim, description='dim'),
            start=widgets.FloatText(description='left edge'),
            stop=widgets.FloatText(description='right edge'),
            num=widgets.IntText(value=1, description='num. edges'),
            unit=widgets.Text(value=str(param.unit), description='unit'),
        ),
        description=param.name,
        child_layout=widgets.VBox,
        layout=widgets.Layout(border='0.5px solid'),
        style=_style,
    )


@create_parameter_widget.register(parameter.FilenameParameter)
def _(param):
    # TODO: Need to add the file upload widget
    return widgets.Text(description=param.name, layout=_layout, style=_style)


@create_parameter_widget.register(parameter.MultiFilenameParameter)
def _(param):
    # TODO: Need to add the file upload widget
    return widgets.Text(description=param.name, layout=_layout, style=_style)


@create_parameter_widget.register(parameter.ParamWithOptions)
def _(param):
    return widgets.Dropdown(
        description=param.name, options=param.options, layout=_layout, style=_style
    )
