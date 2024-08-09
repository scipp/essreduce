# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable
from typing import Any, NewType

import sciline as sl
from ess.reduce.parameter import Parameter, parameter_registry
from ess.reduce.ui import WorkflowWidget
from ess.reduce.widgets import SwitchWidget, create_parameter_widget
from ipywidgets import FloatText, IntText

SwitchableInt = NewType('SwitchableInt', int)
SwitchableFloat = NewType('SwitchableFloat', float)


class IntParam(Parameter): ...


class FloatParam(Parameter): ...


SwitchableIntParam = IntParam('a', 'a', 1, switchable=True)
SwitchableFloatParam = FloatParam('b', 'b', 2.0, switchable=True)

parameter_registry[SwitchableInt] = SwitchableIntParam
parameter_registry[SwitchableFloat] = SwitchableFloatParam


@create_parameter_widget.register(IntParam)
def _(param: IntParam) -> IntText:
    return IntText(value=param.default, description=param.name)


@create_parameter_widget.register(FloatParam)
def _(param: FloatParam) -> FloatText:
    return FloatText(value=param.default, description=param.name)


def ready_widget(
    *,
    providers: list[Callable] | None = None,
    params: dict[type, Any] | None = None,
    output_selections: list[type],
) -> WorkflowWidget:
    widget = WorkflowWidget(sl.Pipeline(providers or [], params=params or {}))
    widget.output_selection_box.typical_outputs_widget.value = output_selections
    widget.parameter_box.parameter_refresh_button.click()
    return widget


def get_param_widget(widget: WorkflowWidget, param_type: type) -> Any:
    return widget.parameter_box._input_widgets[param_type].children[0]


def test_switchable_widget_dispatch() -> None:
    switchable_param = Parameter('a', 'a', 1, switchable=True)
    assert isinstance(create_parameter_widget(switchable_param), SwitchWidget)
    non_switchable_param = Parameter('b', 'b', 2, switchable=False)
    assert not isinstance(create_parameter_widget(non_switchable_param), SwitchWidget)


def provider_with_switch(a: SwitchableInt, b: SwitchableFloat) -> str:
    return f"{a} + {b}"


def test_switchable_parameter_switch_widget() -> None:
    widget = ready_widget(providers=[provider_with_switch], output_selections=[str])

    int_widget = get_param_widget(widget, SwitchableInt)
    float_widget = get_param_widget(widget, SwitchableFloat)

    assert isinstance(int_widget, SwitchWidget)
    assert isinstance(float_widget, SwitchWidget)

    assert not float_widget.enabled
    assert not int_widget.enabled


def test_collect_values_from_disabled_switchable_widget() -> None:
    widget = ready_widget(providers=[provider_with_switch], output_selections=[str])

    assert not get_param_widget(widget, SwitchableFloat).enabled
    assert not get_param_widget(widget, SwitchableInt).enabled
    assert widget.parameter_box.collect_values() == {}


def test_collect_values_from_enabled_switchable_widget() -> None:
    widget = ready_widget(providers=[provider_with_switch], output_selections=[str])

    float_widget = get_param_widget(widget, SwitchableFloat)
    float_widget.enabled = True
    float_widget.value = 0.2

    assert widget.parameter_box.collect_values() == {SwitchableFloat: 0.2}
