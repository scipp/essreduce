# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
import ipywidgets as ipw
import scipp as sc

from ._base import WidgetWithFieldsMixin

UNITS_LIBRARY = {
    "wavelength": {"options": ("angstrom", "nm")},
    "Q": {"options": ("1/angstrom", "1/nm")},
    "Qx": {"options": ("1/angstrom", "1/nm")},
    "Qy": {"options": ("1/angstrom", "1/nm")},
    "tof": {"options": ("s", "ms", "us", "ns"), "selected": "us"},
    "dspacing": {"options": ("angstrom", "nm")},
    "energy_transfer": {"options": ("meV",)},
    "theta": {"options": ("rad", "deg")},
    "two_theta": {"options": ("rad", "deg")},
    "phi": {"options": ("rad", "deg")},
    "time": {"options": ("s", "ms", "us", "ns")},
    "temperature": {"options": ("K", "C", "F")},
}


class BinEdgesWidget(ipw.HBox, ipw.ValueWidget, WidgetWithFieldsMixin):
    def __init__(
        self,
        name: str,
        dim: str,
        start: float | None = None,
        stop: float | None = None,
        nbins: int = 1,
        unit: str | None = "undefined",
        log: bool = False,
    ):
        super().__init__()
        style = {
            "layout": {"width": "100px"},
            "style": {"description_width": "initial"},
        }
        units = UNITS_LIBRARY[dim] if unit == "undefined" else unit
        if isinstance(units, str):
            units = {"options": (units,)}
        if not isinstance(units, dict):
            units = {"options": units}
        self.fields = {
            "dim": ipw.Label(str(dim)),
            "unit": ipw.Dropdown(
                options=units["options"],
                value=units.get("selected", units["options"][0]),
                layout={"width": "initial"},
            ),
            "start": ipw.FloatText(description='start:', value=start, **style),
            "stop": ipw.FloatText(description='stop:', value=stop, **style),
            "nbins": ipw.BoundedIntText(
                description='nbins:', value=nbins, min=1, max=int(1e9), **style
            ),  # Note that a max value is required as it defaults to 100 without it.
            "spacing": ipw.Dropdown(
                options=['linear', 'log'],
                value='log' if log else 'linear',
                layout={"width": "initial"},
            ),
        }
        self.children = [ipw.Label(f"{name}:")] + list(self.fields.values())[1:]

    @property
    def value(self) -> sc.Variable:
        func = sc.geomspace if self.fields["spacing"].value == "log" else sc.linspace
        return func(
            dim=self.fields["dim"].value,
            start=self.fields["start"].value,
            stop=self.fields["stop"].value,
            num=self.fields["nbins"].value + 1,
            unit=self.fields["unit"].value,
        )
