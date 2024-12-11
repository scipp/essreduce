# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import warnings
from typing import Any, Protocol, runtime_checkable

from ipywidgets import Widget


@runtime_checkable
class WidgetWithFieldsProtocol(Protocol):
    def set_fields(self, new_values: dict[str, Any]) -> None: ...

    def get_fields(self) -> dict[str, Any]: ...


class WidgetWithFieldsMixin:
    def set_fields(self, new_values: dict[str, Any]) -> None:
        # Extract valid fields
        new_field_names = set(new_values.keys())
        valid_field_names = new_field_names & set(self.fields.keys())
        # Warn for invalid fields
        invalid_field_names = new_field_names - valid_field_names
        for field_name in invalid_field_names:
            warning_msg = f"Cannot set field '{field_name}'."
            " The field does not exist in the widget."
            "The field value will be ignored."
            warnings.warn(warning_msg, UserWarning, stacklevel=1)
        # Set valid fields
        for field_name in valid_field_names:
            self.fields[field_name].value = new_values[field_name]

    def get_fields(self) -> dict[str, Any]:
        return {
            field_name: field_sub_widget.value
            for field_name, field_sub_widget in self.fields.items()
        }


def _has_widget_value_setter(widget: Widget) -> bool:
    widget_type = type(widget)
    return (
        widget_property := getattr(widget_type, 'value', None)
    ) is not None and getattr(widget_property, 'fset', None) is not None


def set_fields(widget: Widget, new_values: Any) -> None:
    if isinstance(widget, WidgetWithFieldsProtocol) and isinstance(new_values, dict):
        widget.set_fields(new_values)
    elif _has_widget_value_setter(widget):
        widget.value = new_values
    else:
        warnings.warn(
            f"Cannot set value or fields for widget of type {type(widget)}."
            " The new_value(s) will be ignored.",
            UserWarning,
            stacklevel=1,
        )


def get_fields(widget: Widget) -> Any:
    if isinstance(widget, WidgetWithFieldsProtocol):
        return widget.get_fields()
    return widget.value
