# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable
from typing import Any

import ipywidgets as widgets
import sciline as sl
from IPython import display
from ipywidgets import Layout

from .parameter import Parameter
from .widgets import Spinner, SwitchWidget, create_parameter_widget, default_layout
from .widgets._base import get_fields, set_fields
from .workflow import (
    Key,
    assign_parameter_values,
    get_parameters,
    get_possible_outputs,
    get_typical_outputs,
    workflow_registry,
)


def _wrap_foldable(
    wrapped: widgets.Widget, title: str | None = None
) -> widgets.Accordion:
    return widgets.Accordion(
        [wrapped],
        layout=Layout(width='99%', height='auto'),
        titles=(title,),
    )


class OutputSelectionWidget(widgets.VBox):
    def __init__(self, workflow: sl.Pipeline, **kwargs):
        self.typical_outputs_widget = widgets.SelectMultiple(
            options=get_typical_outputs(workflow),
            layout=Layout(width='90%', height='250px'),
        )
        self.possible_outputs_widget = widgets.SelectMultiple(
            options=get_possible_outputs(workflow),
            layout=Layout(width='90%', height='auto'),
        )
        # Wrapping the selection widget in an individual accordion.
        # It is not possible to have multiple sections open in one ``Accordion`` widget.
        _typical_selection = _wrap_foldable(
            self.typical_outputs_widget, title='Typical Outputs'
        )
        _typical_selection.selected_index = 0  # Open typical outputs by default
        _possible_selection = _wrap_foldable(
            self.possible_outputs_widget, title='Extended Outputs'
        )
        super().__init__([_typical_selection, _possible_selection], **kwargs)

    @property
    def value(self) -> set[Key]:
        return set(
            self.typical_outputs_widget.value + self.possible_outputs_widget.value
        )


class ParameterBox(widgets.VBox):
    def __init__(
        self,
        registry_getter: Callable[[], dict[Key, Parameter]],
        **kwargs,
    ):
        self.parameter_refresh_button = widgets.Button(
            description='Refresh Parameters',
            disabled=False,
            button_style='success',
            tooltip='Generate Parameter Input Widgets',
        )
        self._input_registry = {}
        self._input_widgets = {}
        self._input_box = widgets.VBox()

        def _refresh_input_box(_: widgets.Button):
            new_input_parameters = registry_getter()
            self._input_registry.clear()
            self._input_registry.update(new_input_parameters)
            self._input_widgets.clear()
            self._input_widgets.update(
                {
                    node: create_parameter_widget(parameter)
                    for node, parameter in new_input_parameters.items()
                }
            )
            self._input_box.children = [
                widgets.HBox([widget]) for widget in self._input_widgets.values()
            ]

        self.parameter_refresh_button.on_click(_refresh_input_box)

        super().__init__([self.parameter_refresh_button, self._input_box], **kwargs)

    @property
    def value(self) -> dict[Key, Any]:
        """Return the current parameter values with matching types as a dictionary."""
        return {
            node: widget.value
            for node, widget_box in self._input_widgets.items()
            if (not isinstance((widget := widget_box), SwitchWidget)) or widget.enabled
        }


class ResultBox(widgets.VBox):
    def __init__(
        self,
        workflow_runner: Callable[[], dict[type, Any]],
        result_registry: dict | None = None,
        **kwargs,
    ):
        self.output = widgets.Output()
        self.run_button = widgets.Button(
            description='Run',
            disabled=False,
            button_style='success',
            tooltip='Run',
        )
        output_clear_button = widgets.Button(
            description='Clear Output',
            button_style='warning',
            tooltip='Clear Output',
        )

        def run_workflow(_: widgets.Button) -> None:
            self.output.clear_output()
            with self.output:
                display.display(Spinner())
                compute_result = workflow_runner()
                display.clear_output()
                if result_registry is not None:
                    result_registry.clear()
                    result_registry.update(compute_result)
                for i in compute_result.values():
                    display.display(i)

        def clear_output(_: widgets.Button) -> None:
            self.output.clear_output()

        self.run_button.on_click(run_workflow)
        output_clear_button.on_click(clear_output)
        button_box = widgets.HBox([self.run_button, output_clear_button])
        super().__init__([button_box, self.output], **kwargs)


def connect_refresh_button(
    refresh_button: widgets.Button, output_widget: widgets.Output
) -> None:
    def refresh_output(_: widgets.Button):
        output_widget.clear_output()

    refresh_button.on_click(refresh_output)


def connect_output_selection_and_parameter_run_button(
    *output_selection_widgets: widgets.Widget,
    parameter_refresh_button: widgets.Button,
    run_button: widgets.Button,
) -> None:
    # Disable run button when output selection changes
    def observe_selection_change(_) -> None:
        run_button.disabled = True
        run_button.tooltip = 'To run the workflow, refresh parameters.'

    for output_selection_widget in output_selection_widgets:
        output_selection_widget.observe(observe_selection_change)

    # Enable run button when parameters are generated
    original_run_button_tooltip = run_button.tooltip

    def observe_parameter_refrheshed(_) -> None:
        run_button.disabled = False
        run_button.tooltip = original_run_button_tooltip

    parameter_refresh_button.on_click(observe_parameter_refrheshed)


class WorkflowWidget(widgets.TwoByTwoLayout):
    def __init__(
        self, workflow: sl.Pipeline, result_registry: dict | None = None, **kwargs
    ):
        self.output_selection_box = OutputSelectionWidget(workflow)

        def registry_getter() -> dict[Key, Parameter]:
            """Return the parameter registry for the workflow."""
            return get_parameters(workflow, tuple(self.output_selection_box.value))

        self.parameter_box = ParameterBox(registry_getter)

        def workflow_runner() -> dict[type, Any]:
            """Run the workflow with the current parameter values."""
            return assign_parameter_values(
                workflow.copy(), self.parameter_box.value
            ).compute(self.output_selection_box.value)

        self.result_box = ResultBox(workflow_runner, result_registry)
        connect_refresh_button(
            self.parameter_box.parameter_refresh_button, self.result_box.output
        )
        connect_output_selection_and_parameter_run_button(
            self.output_selection_box.typical_outputs_widget,
            self.output_selection_box.possible_outputs_widget,
            parameter_refresh_button=self.parameter_box.parameter_refresh_button,
            run_button=self.result_box.run_button,
        )
        for box in (self.output_selection_box, self.parameter_box, self.result_box):
            box.layout.border = '1px solid black'

        super().__init__(
            top_left=self.output_selection_box,
            top_right=self.parameter_box,
            bottom_left=self.result_box,
            grid_gap="10px",
            layout=default_layout,
            **kwargs,
        )


def workflow_widget(result_registry: dict | None = None) -> widgets.Widget:
    """Create a widget for a workflow selected from a dropdown."""
    workflow_select = widgets.Dropdown(
        options=[(workflow.__name__, workflow) for workflow in workflow_registry],
        description='Workflow:',
        value=None,
        layout=default_layout,
        tooltip='Select a workflow.',
    )

    def refresh_workflow_box(change) -> None:
        workflow_box.children = [WorkflowWidget(change.new(), result_registry)]

    workflow_select.observe(refresh_workflow_box, names='value')

    workflow_selection_box = widgets.HBox([workflow_select], layout=default_layout)
    workflow_box = widgets.Box(layout=default_layout)
    return widgets.VBox([workflow_selection_box, workflow_box])


def _get_parameter_box(widget: WorkflowWidget | ParameterBox) -> ParameterBox:
    if isinstance(widget, WorkflowWidget):
        return widget.parameter_box
    elif isinstance(widget, ParameterBox):
        return widget
    else:
        raise TypeError(
            f"Expected target_widget to be a WorkflowWidget or ParameterBox, "
            f"got {type(widget)}."
        )


def set_parameter_widget_values(
    widget: WorkflowWidget | ParameterBox, new_parameter_values: dict[type, Any]
) -> None:
    """Set the values of the input widgets in the target widget.

    Nodes that don't exist in the input widgets will be ignored.

    Example
    -------
    .. code-block::

        set_parameter_widget_values(widget, {
            'WavelengthBins': {'start': 1.0, 'stop': 14.0, 'nbins': 500}
        })

    Parameters
    ----------
    widget:
        The widget containing the input widgets.
    new_parameter_values:
        A dictionary of values/state to set each fields/state or value of input widgets.

    Raises
    ------
    TypeError:
        If the widget is not a WorkflowWidget or a ParameterBox.

    """
    parameter_box = _get_parameter_box(widget)
    # Walk through the existing input widgets and set the values
    # ``node`s that don't exist in the input widgets will be ignored.
    for node, widget in parameter_box._input_widgets.items():
        if node in new_parameter_values:
            # We shouldn't use `get` here because ``None`` is a valid value.
            set_fields(widget, new_parameter_values[node])


def get_parameter_widget_values(
    widget: WorkflowWidget | ParameterBox,
) -> dict[type, Any]:
    """Return the current values of the input widgets in the target widget.

    The result of this function can be used to set the values of the input widgets
    using the :py:func:`~set_parameter_widget_values` function.

    Parameters
    ----------
    widget:
        The widget containing the input widgets.

    Returns
    -------
    :
        A dictionary of the current values/state of each input widget.

    Raises
    ------
    TypeError:
        If the widget is not a WorkflowWidget or a ParameterBox.

    """
    return {
        node: get_fields(widget)
        for node, widget in _get_parameter_box(widget)._input_widgets.items()
    }
