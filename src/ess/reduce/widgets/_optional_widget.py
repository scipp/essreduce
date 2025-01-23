# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import Any

from ipywidgets import HTML, HBox, Layout, RadioButtons, Widget

from ._base import get_fields, set_fields
from ._config import default_style


class OptionalWidget(HBox):
    """Wrapper widget to handle optional widgets.

    When you retrieve the value of this widget,
    it will return the value of the wrapped widget.
    The parameter should be set as ``None`` or the actual value.
    """

    def __init__(self, wrapped: Widget, name: str = '', **kwargs) -> None:
        self.name = name
        self.wrapped = wrapped
        self._option_box = RadioButtons(
            description="",
            style=default_style,
            layout=Layout(width="auto", min_width="80px"),
            options={str(None): None, "": self.name},
        )
        self._option_box.value = None
        if hasattr(wrapped, "disabled"):
            # Disable the wrapped widget by default if possible
            # since the option box is set to None by default
            wrapped.disabled = True

        # Make the wrapped radio box horizontal
        self.add_class("widget-optional")
        _style_html = HTML(
            "<style>.widget-optional .widget-radio-box "
            "{flex-direction: row !important;} </style>",
            layout=Layout(display="none"),
        )

        def disable_wrapped(change) -> None:
            if change["new"] is None:
                if hasattr(wrapped, "disabled"):
                    wrapped.disabled = True
            else:
                if hasattr(wrapped, "disabled"):
                    wrapped.disabled = False

        self._option_box.observe(disable_wrapped, names="value")

        super().__init__([self._option_box, wrapped, _style_html], **kwargs)

    @property
    def value(self) -> Any:
        if self._option_box.value is None:
            self._option_box.value = None
            return None
        return self.wrapped.value

    @value.setter
    def value(self, value: Any) -> None:
        if value is None:
            self._option_box.value = None
        else:
            self._option_box.value = self.name
            self.wrapped.value = value

    def set_fields(self, new_values: dict[str, Any]) -> None:
        new_values = dict(new_values)
        # Set the value of the option box
        opted_out_flag = new_values.pop(
            # We assume ``essreduce-opted-out`` is not used in any wrapped widget
            'essreduce-opted-out',
            self._option_box.value is None,
        )
        if not isinstance(opted_out_flag, bool):
            raise ValueError(
                f"Invalid value for 'essreduce-opted-out' field: {opted_out_flag}."
                " The value should be a boolean."
            )
        self._option_box.value = None if opted_out_flag else self.name
        # Set the value of the wrapped widget
        set_fields(self.wrapped, new_values)

    def get_fields(self) -> dict[str, Any] | None:
        return {
            **(get_fields(self.wrapped) or {}),
            'essreduce-opted-out': self._option_box.value is None,
        }
