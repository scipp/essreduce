# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import Any

from ipywidgets import Checkbox, HBox, Label, Stack, Widget

from ._config import default_style


class SwitchWidget(HBox):
    """Wrapper widget to handle switchable widgets.

    When you retrieve the value of this widget,
    it will return the value of the wrapped widget.
    It is expected not to be set in the workflow if ``enabled`` is False.
    """

    def __init__(self, wrapped: Widget, name: str = '') -> None:
        super().__init__()
        self.enable_box = Checkbox(description='Use Parameter', style=default_style)
        self.wrapped = wrapped
        wrapped_stack = Stack([Label(name, style=default_style), self.wrapped])
        wrapped_stack.selected_index = 0

        def handle_checkbox(change) -> None:
            wrapped_stack.selected_index = 1 if change.new else 0

        self.enable_box.observe(handle_checkbox, names='value')

        self.children = [self.enable_box, wrapped_stack]

    @property
    def enabled(self) -> bool:
        return self.enable_box.value

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self.enable_box.value = value

    @property
    def value(self) -> Any:
        return self.wrapped.value
