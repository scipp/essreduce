# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from ipywidgets import HBox, Text, ValueWidget

from ._config import default_layout


class StringWidget(HBox, ValueWidget):
    def __init__(self, description: str, value: str | None = None, **kwargs):
        super().__init__(layout=default_layout)
        self.text_widget = Text(description=description, value=value, **kwargs)
        self.children = [self.text_widget]

    @property
    def value(self) -> str | None:
        v = self.text_widget.value.strip()
        if not v:
            return None
        return v

    @value.setter
    def value(self, value: str | None):
        if value is None:
            self.text_widget.value = ''
        else:
            self.text_widget.value = value


class MultiStringWidget(StringWidget):
    def __init__(
        self, description: str, value: tuple[str, ...] | None = None, **kwargs
    ):
        # Special case handling to allow initialising with a single string
        if not isinstance(value, str) and value is not None:
            value = ', '.join(value)

        super().__init__(description, value, **kwargs)

    @property
    def value(self) -> tuple[str, ...]:
        v = super().value
        if v is None:
            return ()
        return tuple(s.strip() for s in v.split(','))

    @value.setter
    def value(self, value: tuple[str, ...]):
        self.text_widget.value = ', '.join(value)
