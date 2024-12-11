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


def set_fields(widget: Widget, new_values: Any) -> None:
    if isinstance(widget, WidgetWithFieldsProtocol) and isinstance(new_values, dict):
        widget.set_fields(new_values)
    else:
        try:
            widget.value = new_values
        except AttributeError as error:
            # Checking if the widget value property has a setter in advance, i.e.
            # ```python
            # (widget_property := getattr(type(widget), 'value', None)) is not None
            # and getattr(widget_property, 'fset', None) is not None
            # ```
            # does not work with a class that inherits Traitlets class.
            # In those classes, even if a property has a setter,
            # it may not have `fset` attribute.
            # It is not really feasible to check all possible cases of value setters.
            # Instead, we try setting the value and catch the AttributeError.
            # to determine if the widget has a value setter.
            warnings.warn(
                f"Cannot set value for widget of type {type(widget)}."
                " The new_value(s) will be ignored."
                f" Setting value caused the following error: {error}",
                UserWarning,
                stacklevel=1,
            )


def get_fields(widget: Widget) -> Any:
    if isinstance(widget, WidgetWithFieldsProtocol):
        return widget.get_fields()
    return widget.value
