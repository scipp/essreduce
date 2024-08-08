# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import pytest
from ess.reduce.parameter import Parameter
from ess.reduce.widgets import SwitchWidget, create_parameter_widget


@pytest.mark.filterwarnings(
    'ignore::DeprecationWarning'
)  # Ignore deprecation warning from widget library
def test_switchable_widget_dispatch() -> None:
    switchable_param = Parameter('a', 'a', 1, switchable=True)
    assert isinstance(create_parameter_widget(switchable_param), SwitchWidget)
    non_switchable_param = Parameter('b', 'b', 2, switchable=False)
    assert not isinstance(create_parameter_widget(non_switchable_param), SwitchWidget)


@pytest.mark.filterwarnings(
    'ignore::DeprecationWarning'
)  # Ignore deprecation warning from widget library
def test_collect_values_from_disabled_switchable_widget() -> None:
    from ess.reduce.ui import collect_values
    from ipywidgets import Box, Text, VBox

    enabled_switch_widget = SwitchWidget(Text('int'), name='int')
    enabled_switch_widget.enabled = True
    disabled_switch_widget = SwitchWidget(Text('float'), name='float')
    disabled_switch_widget.enabled = False
    non_switch_widget = Text('str')
    test_box = VBox(
        [
            Box([enabled_switch_widget]),
            Box([disabled_switch_widget]),
            Box([non_switch_widget]),
        ]
    )

    values = collect_values(test_box, (int, float, str))
    assert values == {int: 'int', str: 'str'}
