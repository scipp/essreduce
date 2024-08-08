# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable, Iterable
from typing import Any, cast

import ipywidgets as widgets
import sciline
from IPython import display
from ipywidgets import Layout, TwoByTwoLayout

from .widgets import SwitchWidget, create_parameter_widget
from .workflow import (
    Key,
    assign_parameter_values,
    get_parameters,
    get_possible_outputs,
    get_typical_outputs,
    workflow_registry,
)

_style = {
    'description_width': 'auto',
    'value_width': 'auto',
    'button_width': 'auto',
}
workflow_select = widgets.Dropdown(
    options=[(workflow.__name__, workflow) for workflow in workflow_registry],
    description='Workflow:',
    value=None,
)

typical_outputs_widget = widgets.SelectMultiple(
    description='Outputs:', layout=Layout(width='80%', height='150px')
)

possible_outputs_widget = widgets.SelectMultiple(
    description='Extended Outputs:',
    style=_style,
    layout=Layout(width='80%', height='150px'),
)


def handle_workflow_select(change) -> None:
    selected_workflow: sciline.Pipeline = change.new()
    typical_outputs_widget.options = get_typical_outputs(selected_workflow)
    possible_outputs_widget.options = get_possible_outputs(selected_workflow)


workflow_select.observe(handle_workflow_select, names='value')

generate_parameter_button = widgets.Button(
    description='Generate Parameters',
    disabled=False,
    button_style='info',
    tooltip='Generate Parameters',
)

reset_button = widgets.Button(
    description='Reset',
    disabled=True,
    button_style='info',
    tooltip='Reset',
)

parameter_box = widgets.VBox([])


def generate_parameter_widgets():
    workflow_constructor = cast(Callable[[], sciline.Pipeline], workflow_select.value)
    selected_workflow = workflow_constructor()
    outputs = possible_outputs_widget.value + typical_outputs_widget.value
    registry = get_parameters(selected_workflow, outputs)

    for parameter in registry.values():
        temp_widget = create_parameter_widget(parameter)
        temp = widgets.HBox([temp_widget])
        parameter_box.children = (*parameter_box.children, temp)


def on_button_clicked(b):
    generate_parameter_widgets()
    generate_parameter_button.disabled = True
    reset_button.disabled = False
    run_button.disabled = False
    output.clear_output()


generate_parameter_button.on_click(on_button_clicked)


def reset_button_clicked(b):
    generate_parameter_button.disabled = False
    reset_button.disabled = True
    parameter_box.children = []
    output.clear_output()


reset_button.on_click(reset_button_clicked)

button_box = widgets.HBox([generate_parameter_button, reset_button])
workflow_box = widgets.VBox(
    [workflow_select, typical_outputs_widget, possible_outputs_widget, button_box]
)

run_button = widgets.Button(
    description='Run',
    disabled=True,
    button_style='success',
    tooltip='Run',
)

output = widgets.Output()


def collect_values(
    parameter_box: widgets.VBox, param_keys: Iterable[Key]
) -> dict[Key, Any]:
    return {
        node: parameter_box.children[i].children[0].value
        for i, node in enumerate(param_keys)
        if (
            not isinstance(
                widget := parameter_box.children[i].children[0], SwitchWidget
            )
        )
        or widget.enabled
    }


def run_workflow(_: widgets.Button) -> None:
    workflow_constructor = cast(Callable[[], sciline.Pipeline], workflow_select.value)
    selected_workflow = workflow_constructor()
    outputs = possible_outputs_widget.value + typical_outputs_widget.value
    registry = get_parameters(selected_workflow, outputs)

    values = collect_values(parameter_box, registry.keys())

    workflow = assign_parameter_values(selected_workflow, values)

    with output:
        compute_result = workflow.compute(
            outputs, scheduler=sciline.scheduler.NaiveScheduler()
        )
        results.clear()
        results.update(compute_result)
        for i in compute_result.values():
            display.display(display.HTML(i._repr_html_()))


run_button.on_click(run_workflow)

layout = TwoByTwoLayout(
    top_left=workflow_box,
    bottom_left=widgets.VBox([run_button, output]),
    bottom_right=parameter_box,
)
"""Widget for selecting a workflow and its parameters.

To render this in a voila server, create a notebook which imports this
layout and then render it.
```python
    from ess.reduce import ui
    ui.layout
```
Results will be in ui.results
"""
results = {}
