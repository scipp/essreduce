# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import Any

from ipywidgets import Checkbox, HBox, Label, Stack, Widget

from ._base import WidgetWithFieldsProtocol, get_fields, set_fields
from ._config import default_style


class SwitchWidget(HBox):
    """Wrapper widget to handle switchable widgets.

    When you retrieve the value of this widget,
    it will return the value of the wrapped widget.
    It is expected not to be set in the workflow if ``enabled`` is False.
    """

    def __init__(self, wrapped: Widget, name: str = '') -> None:
        super().__init__()
        self._enable_box = Checkbox(description='', style=default_style)
        # The layout is not applied if they are set in the constructor
        self._enable_box.layout.description_width = '0px'
        self._enable_box.layout.width = 'auto'

        self.wrapped = wrapped
        wrapped_stack = Stack([Label(name, style=default_style), self.wrapped])
        # We wanted to implement this by greying out the widget when disabled
        # but ``disabled`` is not a common property of all widgets
        wrapped_stack.selected_index = 0

        def handle_checkbox(change) -> None:
            wrapped_stack.selected_index = 1 if change.new else 0

        self._enable_box.observe(handle_checkbox, names='value')
        self.children = [self._enable_box, wrapped_stack]

    @property
    def enabled(self) -> bool:
        return self._enable_box.value

    @enabled.setter
    def enabled(self, value: bool) -> None:
        self._enable_box.value = value

    @property
    def value(self) -> Any:
        return self.wrapped.value

    @value.setter
    def value(self, value: Any) -> None:
        self.wrapped.value = value

    def set_fields(self, new_values: dict[str, Any]) -> None:
        # Check if the new values are a dictionary
        if not isinstance(new_values, dict):
            raise ValueError(f"Expected a dictionary, got {new_values}.")
        # Retrieve and set the enabled flag first
        new_values = dict(new_values)
        enabled_flag = new_values.pop('enabled', self.enabled)
        if not isinstance(enabled_flag, bool):
            raise ValueError(f"`enabled` must be a boolean, got {enabled_flag}")
        self.enabled = enabled_flag
        # Set fields or value of the wrapped widget
        if isinstance(self.wrapped, WidgetWithFieldsProtocol):
            # Expecting {'enabled': True/False, **wrapped_fields}
            self.wrapped.set_fields(new_values)
        elif 'value' in new_values:  # Skip if 'value' is not in new_values
            # i.e. User might only want to change the enabled flag
            # We use ``set_fields`` to set the value of the wrapped widget
            # so that it handles availability of a value setter.
            # Expecting {'enabled': True/False, 'value': value}
            set_fields(self.wrapped, new_values.pop('value'))
            if new_values:
                # Check if there are any fields left except for 'enabled' and 'value'
                raise ValueError(
                    f"Unexpected field(s) {new_values.keys()} for widget {self.wrapped}"
                )

    def get_fields(self) -> dict[str, Any]:
        wrapped_fields = get_fields(self.wrapped)
        if isinstance(self.wrapped, WidgetWithFieldsProtocol) and isinstance(
            wrapped_fields, dict
        ):
            return {'enabled': self.enabled, **wrapped_fields}
        else:
            return {'enabled': self.enabled, 'value': wrapped_fields}
