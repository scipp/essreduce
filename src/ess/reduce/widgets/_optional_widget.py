# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from typing import Any

from ipywidgets import HTML, HBox, Layout, RadioButtons, Widget

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
