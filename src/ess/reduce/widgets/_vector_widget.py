# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import scipp as sc
from ipywidgets import FloatText, HBox, Label, Text, ValueWidget

from ._base import WidgetWithFieldsMixin


class VectorWidget(HBox, ValueWidget, WidgetWithFieldsMixin):
    def __init__(self, name: str, variable: sc.Variable, components: str):
        super().__init__()

        style = {
            "layout": {"width": "130px"},
            "style": {"description_width": "initial"},
        }
        self.fields = {
            c: FloatText(
                description=f"{c} =", value=getattr(variable.fields, c).value, **style
            )
            for c in components
        }
        self.fields["unit"] = Text(
            description="unit:", value=str(variable.unit), **style
        )
        self.children = [Label(value=f"{name}: "), *list(self.fields.values())]

    @property
    def value(self):
        return sc.vector(
            value=[
                self.fields['x'].value,
                self.fields['y'].value,
                self.fields.get('z', sc.scalar(0.0)).value,
            ],
            unit=self.fields['unit'].value,
        )
