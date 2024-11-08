# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import scipp as sc
from ipywidgets import FloatText, HBox, Label, Text, ValueWidget

from ..parameter import ParamWithBounds


class BoundsWidget(HBox, ValueWidget):
    def __init__(
        self,
        name: str,
        dim: str,
        start: float | None = None,
        stop: float | None = None,
        unit: str | None = None,
    ):
        super().__init__()
        style = {
            "layout": {"width": "100px"},
            "style": {"description_width": "initial"},
        }

        self.fields = {
            "dim": Label(str(dim)),
            'start': FloatText(description='start', **style),
            'end': FloatText(description='end', **style),
            'unit': Text(description='unit', **style),
        }
        self.children = [
            Label(f"{name}:"),
            self.fields['dim'],
            self.fields['unit'],
            self.fields['start'],
            self.fields['end'],
        ]

    @property
    def value(self):
        return (
            self.fields['dim'].value,
            slice(
                sc.scalar(self.fields['start'].value, unit=self.fields['unit'].value),
                sc.scalar(self.fields['end'].value, unit=self.fields['unit'].value),
                None,
            ),
        )

    @staticmethod
    def from_ess_parameter(_: ParamWithBounds) -> 'BoundsWidget':
        return BoundsWidget()
