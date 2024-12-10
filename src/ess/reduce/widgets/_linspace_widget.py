# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ipywidgets import FloatText, GridBox, IntText, Label, ValueWidget

from ._base import WidgetWithFieldsMixin


class LinspaceWidget(GridBox, ValueWidget, WidgetWithFieldsMixin):
    def __init__(self, dim: str, unit: str):
        super().__init__()

        self.fields = {
            'dim': Label(value=dim, description='dim'),
            'start': FloatText(description='start', value=0.0),
            'end': FloatText(description='end', value=1.0),
            'num': IntText(description='num', value=100),
            'unit': Label(description='unit', value=unit),
        }
        self.children = [
            Label(value="Select range:"),
            self.fields['dim'],
            self.fields['unit'],
            self.fields['start'],
            self.fields['end'],
            self.fields['num'],
        ]

    @property
    def value(self) -> sc.Variable:
        return sc.linspace(
            self.fields['dim'].value,
            self.fields['start'].value,
            self.fields['end'].value,
            self.fields['num'].value,
            unit=self.fields['unit'].value,
        )
