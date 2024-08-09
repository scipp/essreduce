# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import scipp as sc
from ipywidgets import FloatText, GridBox, Label, Text, ValueWidget


class VectorWidget(GridBox, ValueWidget):
    def __init__(self, variable: sc.Variable):
        super().__init__()

        self.fields = {
            "x": FloatText(description="x", value=variable.fields.x.value),
            "y": FloatText(description="y", value=variable.fields.y.value),
            "z": FloatText(description="z", value=variable.fields.z.value),
            "unit": Text(description="unit", value=str(variable.unit)),
        }
        self.children = [
            Label(value="(x, y, z) ="),
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
