# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import os
import plopp as pp
import scipp as sc

from mpltoolbox import Hspans, Vspans


def _define_rect_mask(da, rect_info):
    """
    Function that creates a mask inside the area
    covered by the rectangle.
    """
    x = rect_info["x"]
    y = rect_info["y"]
    xcoord = da.coords[x["dim"]]
    ycoord = da.coords[y["dim"]]
    return (
        (xcoord >= x["lower"])
        & (xcoord <= x["upper"])
        & (ycoord >= y["lower"])
        & (ycoord <= y["upper"])
    )


def _get_rect_info(artist, figure):
    """
    Convert the raw rectangle info to a dict containing the dimensions of
    each axis, and values with units.
    """
    x1 = artist.xy[0]
    x2 = artist.xy[0] + artist.width
    y1 = artist.xy[1]
    y2 = artist.xy[1] + artist.height
    return lambda: {
        "x": {
            "dim": figure.canvas.dims["x"],
            "lower": sc.scalar(min(x1, x2), unit=figure.canvas.units["x"]),
            "upper": sc.scalar(max(x1, x2), unit=figure.canvas.units["x"]),
        },
        "y": {
            "dim": figure.canvas.dims["y"],
            "lower": sc.scalar(min(y1, y2), unit=figure.canvas.units["y"]),
            "upper": sc.scalar(max(y1, y2), unit=figure.canvas.units["y"]),
        },
    }


def _define_span_mask(da, span_info):
    info = next(iter(span_info.values()))
    coord = da.coords[info["dim"]]
    return (coord >= info["lower"]) & (coord <= info["upper"])


def _get_vspan_info(artist, figure):
    x1 = artist.left
    x2 = artist.right
    return lambda: {
        "x": {
            "dim": figure.canvas.dims["x"],
            "lower": sc.scalar(min(x1, x2), unit=figure.canvas.units["x"]),
            "upper": sc.scalar(max(x1, x2), unit=figure.canvas.units["x"]),
        }
    }


def _get_hspan_info(artist, figure):
    y1 = artist.bottom
    y2 = artist.top
    return lambda: {
        "y": {
            "dim": figure.canvas.dims["y"],
            "lower": sc.scalar(min(y1, y2), unit=figure.canvas.units["y"]),
            "upper": sc.scalar(max(y1, y2), unit=figure.canvas.units["y"]),
        }
    }


def _apply_masks(da, *masks):
    out = da.copy(deep=False)
    for i, mask in enumerate(masks):
        out.masks[str(i)] = mask
    return out


class MaskingTool:
    def __init__(self, data):
        import ipywidgets as ipw
        from plopp.widgets import DrawingTool, style
        from mpltoolbox import Rectangles

        # Convert potential bin edge coords to midpoints
        da = data.copy(deep=False)
        for dim, coord in data.coords.items():
            if data.coords.is_edges(dim):
                da.coords[dim] = sc.midpoints(coord)

        self.data_node = pp.Node(da)
        self.masking_node = pp.Node(_apply_masks, self.data_node)
        self.fig = pp.imagefigure(self.masking_node, norm="log")

        common = {
            "figure": self.fig,
            "input_node": self.data_node,
            "destination": self.masking_node,
        }
        rects = DrawingTool(
            tool=Rectangles,
            get_artist_info=_get_rect_info,
            icon="vector-square",
            func=_define_rect_mask,
            **common,
        )
        vspans = DrawingTool(
            tool=Vspans,
            get_artist_info=_get_vspan_info,
            icon="grip-lines-vertical",
            func=_define_span_mask,
            **common,
        )
        hspans = DrawingTool(
            tool=Hspans,
            get_artist_info=_get_hspan_info,
            icon="grip-lines",
            func=_define_span_mask,
            **common,
        )
        self.controls = [rects, vspans, hspans]

        self.fig.top_bar.add(ipw.HBox([], layout={"width": "120px"}))
        for c in self.controls:
            self.fig.top_bar.add(c)
            c._tool._on_change.clear()
            c._tool.on_vertex_release(c.update_node)
            c._tool.on_drag_release(c.update_node)
            c._tool.on_remove(c.update_node)
            c._tool.on_remove(self.masking_node.notify_children)
            c.observe(self.toggle_button_states, names="value")

        self.filename = ipw.Text(placeholder="Save to file")
        self.filename.observe(self.validate_filename, names="value")
        self.save_button = ipw.Button(
            icon="save", tooltip="Save masks", disabled=True, **style.BUTTON_LAYOUT
        )
        self.save_button.on_click(self.save_masks)
        self.fig.top_bar.add(self.filename)
        self.fig.top_bar.add(self.save_button)

    def toggle_button_states(self, change):
        if change["new"]:
            for c in self.controls:
                if c.value and c is not change["owner"]:
                    c.value = False

    def validate_filename(self, change):
        path = change["new"]
        if os.path.exists(path):
            self.filename.style.background = "#ff9999"
            self.save_button.disabled = True
        else:
            self.filename.style.background = "#99ff99"
            self.save_button.disabled = False

    def save_masks(self, _=None):
        masks = {}
        mask_counter = 0
        for c in self.controls:
            for node in c._draw_nodes.values():
                info = node()
                mask_dims = "".join([axis["dim"] for axis in info.values()])
                mask_name = f"{mask_dims}_{mask_counter}"
                masks[mask_name] = sc.DataGroup(
                    {
                        axis["dim"]: sc.concat(
                            [axis["lower"], axis["upper"]], dim=axis["dim"]
                        )
                        for axis in info.values()
                    }
                )
                mask_counter += 1
        out = sc.DataGroup(masks)
        out.save_hdf5(self.filename.value)
        return out


def masking_tool(da):
    return MaskingTool(da)
