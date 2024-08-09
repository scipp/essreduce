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


parameter_registry[SwitchableInt] = IntParam('_', '_', 1, switchable=True)
parameter_registry[SwitchableFloat] = FloatParam('_', '_', 2.0, switchable=True)
parameter_registry[int] = IntParam('_', '_', 1)
parameter_registry[float] = FloatParam('_', '_', 2.0)


@create_parameter_widget.register(IntParam)
def _(param: IntParam) -> IntText:
    return IntText(value=param.default, description=param.name)


@create_parameter_widget.register(FloatParam)
def _(param: FloatParam) -> FloatText:
    return FloatText(value=param.default, description=param.name)


def _ready_widget(
    *,
    providers: list[Callable] | None = None,
    params: dict[type, Any] | None = None,
    output_selections: list[type],
    result_registry: dict[type, Any] | None = None,
) -> WorkflowWidget:
    widget = WorkflowWidget(
        sl.Pipeline(providers or [], params=params or {}),
        result_registry=result_registry,
    )
    widget.output_selection_box.typical_outputs_widget.value = output_selections
    widget.parameter_box.parameter_refresh_button.click()
    return widget


def strict_provider(a: int, b: float) -> str:
    return f"{a} + {b}"


def provider_with_switch(a: SwitchableInt, b: SwitchableFloat) -> str:
    return f"{a} + {b}"


def _get_param_widget(widget: WorkflowWidget, param_type: type) -> Any:
    return widget.parameter_box._input_widgets[param_type].children[0]


def test_parameter_default_value_test() -> None:
    widget = _ready_widget(providers=[strict_provider], output_selections=[str])
    assert _get_param_widget(widget, int).value == 1
    assert _get_param_widget(widget, float).value == 2.0


def test_result_registry() -> None:
    registry = {}
    widget = _ready_widget(
        providers=[strict_provider], output_selections=[str], result_registry=registry
    )
    _get_param_widget(widget, int).value = 2
    _get_param_widget(widget, float).value = 0.1
    assert registry == {}
    widget.result_box.run_button.click()
    assert registry == {str: '2 + 0.1'}


def test_switchable_widget_dispatch() -> None:
    switchable_param = Parameter('a', 'a', 1, switchable=True)
    assert isinstance(create_parameter_widget(switchable_param), SwitchWidget)
    non_switchable_param = Parameter('b', 'b', 2, switchable=False)
    assert not isinstance(create_parameter_widget(non_switchable_param), SwitchWidget)


def test_switchable_parameter_switch_widget() -> None:
    widget = _ready_widget(providers=[provider_with_switch], output_selections=[str])

    int_widget = _get_param_widget(widget, SwitchableInt)
    float_widget = _get_param_widget(widget, SwitchableFloat)

    assert isinstance(int_widget, SwitchWidget)
    assert isinstance(float_widget, SwitchWidget)

    assert not float_widget.enabled
    assert not int_widget.enabled


def test_collect_values_from_disabled_switchable_widget() -> None:
    widget = _ready_widget(providers=[provider_with_switch], output_selections=[str])

    assert not _get_param_widget(widget, SwitchableFloat).enabled
    assert not _get_param_widget(widget, SwitchableInt).enabled
    assert widget.parameter_box.collect_values() == {}


def test_collect_values_from_enabled_switchable_widget() -> None:
    widget = _ready_widget(providers=[provider_with_switch], output_selections=[str])

    float_widget = _get_param_widget(widget, SwitchableFloat)
    float_widget.enabled = True
    float_widget.value = 0.2

    assert widget.parameter_box.collect_values() == {SwitchableFloat: 0.2}
