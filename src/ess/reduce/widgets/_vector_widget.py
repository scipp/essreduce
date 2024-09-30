# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import scipp as sc
from ipywidgets import FloatText, HBox, Label, Text, ValueWidget


class VectorWidget(HBox, ValueWidget):
    def __init__(self, name: str, variable: sc.Variable):
        super().__init__()

        style = {
            "layout": {"width": "130px"},
            "style": {"description_width": "initial"},
        }
        self.fields = {
            "x": FloatText(description="x =", value=variable.fields.x.value, **style),
            "y": FloatText(description="y =", value=variable.fields.y.value, **style),
            "z": FloatText(description="z =", value=variable.fields.z.value, **style),
            "unit": Text(description="unit:", value=str(variable.unit), **style),
        }
        self.children = [
            Label(value=f"{name}: "),
            self.fields['x'],
            self.fields['y'],
            self.fields['z'],
            self.fields['unit'],
        ]

    @property
    def value(self):
        return sc.vector(
            value=[
                self.fields['x'].value,
                self.fields['y'].value,
                self.fields['z'].value,
            ],
            unit=self.fields['unit'].value,
        )
