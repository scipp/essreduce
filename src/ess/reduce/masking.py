# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

import plopp as pp
import scipp as sc

from plopp.widgets.drawing import DrawingTool
from functools import partial
from mpltoolbox import Rectangles, Vspans


def define_rect_mask(da, rect_info):
    """
    Function that creates a mask inside the area
    covered by the rectangle.
    """
    x = rect_info["x"]
    y = rect_info["y"]
    b = min(y["bottom"], y["top"])
    t = max(y["bottom"], y["top"])
    l = min(x["left"], x["right"])
    r = max(x["left"], x["right"])

    xcoord = sc.midpoints(da.coords[x["dim"]])
    ycoord = sc.midpoints(da.coords[y["dim"]])
    return (xcoord >= l) & (xcoord <= r) & (ycoord >= b) & (ycoord <= t)


def _get_rect_info(artist, figure):
    """
    Convert the raw rectangle info to a dict containing the dimensions of
    each axis, and values with units.
    """
    return lambda: {
        "x": {
            "dim": figure.canvas.dims["x"],
            "left": sc.scalar(artist.xy[0], unit=figure.canvas.units["x"]),
            "right": sc.scalar(
                artist.xy[0] + artist.width, unit=figure.canvas.units["x"]
            ),
        },
        "y": {
            "dim": figure.canvas.dims["y"],
            "bottom": sc.scalar(artist.xy[1], unit=figure.canvas.units["y"]),
            "top": sc.scalar(
                artist.xy[1] + artist.height, unit=figure.canvas.units["y"]
            ),
        },
    }


RectangleTool = partial(
    DrawingTool, tool=Rectangles, get_artist_info=_get_rect_info, icon="vector-square"
)


def define_vspan_mask(da, span_info):
    """ """
    xcoord = sc.midpoints(da.coords[span_info["dim"]])
    return (xcoord >= span_info["left"]) & (xcoord <= span_info["right"])


def _get_vspan_info(artist, figure):
    """ """
    return lambda: {
        "dim": figure.canvas.dims["x"],
        "left": sc.scalar(artist.left, unit=figure.canvas.units["x"]),
        "right": sc.scalar(artist.right, unit=figure.canvas.units["x"]),
    }


VspansTool = partial(
    DrawingTool,
    tool=Vspans,
    get_artist_info=_get_vspan_info,
    icon="grip-lines-vertical",
)


def _apply_masks(da, *masks):
    out = da.copy(deep=False)
    for i, mask in enumerate(masks):
        out.masks[str(i)] = mask
    return out


def masking_tool(da):
    data_node = pp.Node(da)

    masking_node = pp.Node(_apply_masks, data_node)

    fig = pp.imagefigure(masking_node, norm="log")

    r = RectangleTool(
        figure=fig,
        input_node=data_node,
        func=define_rect_mask,
        destination=masking_node,
    )
    v = VspansTool(
        figure=fig,
        input_node=data_node,
        func=define_vspan_mask,
        destination=masking_node,
    )

    import ipywidgets as ipw

    fig.top_bar.add(ipw.HBox([], layout={"width": "120px"}))
    fig.top_bar.add(r)
    fig.top_bar.add(v)

    for t in (r, v):
        t._tool._on_change.clear()
        t._tool.on_vertex_release(r.update_node)
        t._tool.on_drag_release(r.update_node)

    return fig
