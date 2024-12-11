# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ipywidgets import FloatText, GridBox, Label, Text, ValueWidget

from ..parameter import ParamWithBounds
from ._base import WidgetWithFieldsMixin


class BoundsWidget(GridBox, ValueWidget, WidgetWithFieldsMixin):
    def __init__(self):
        super().__init__()

        self.fields = {
            'start': FloatText(description='start'),
            'end': FloatText(description='end'),
            'unit': Text(description='unit'),
        }
        self.children = [
            Label(value="Select bound:"),
            self.fields['unit'],
            self.fields['start'],
            self.fields['end'],
        ]

    @property
    def value(self):
        return (
            sc.scalar(self.fields['start'].value, unit=self.fields['unit']),
            sc.scalar(self.fields['end'].value, unit=self.fields['unit']),
        )

    @staticmethod
    def from_ess_parameter(_: ParamWithBounds) -> 'BoundsWidget':
        return BoundsWidget()
