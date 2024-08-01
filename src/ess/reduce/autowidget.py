import ipywidgets as ipw
from IPython.display import display


class AutoWidget(ipw.ValueWidget, ipw.AppLayout):
    def __init__(
        self,
        tp,
        tp_args,
        tp_kwargs=None,
        *,
        display_output=True,
        arg_layout=ipw.VBox,
        kwarg_layout=ipw.VBox,
        **kwargs,
    ):
        '''Creates a widget from a constructor `tp` and a range of argument widgets

        Example usage:
            bound_widget = AutoWidget(
                lambda a, b: (sc.scalar(a), sc.scalar(b)),
                ipw.FloatText(description="left"),
                ipw.FloatText(description="right")
            )
            range_widget = AutoWidget(
                sc.linspace,
                dim=ipywidgets.Text(value='m'),
                start=ipywidgets.FloatText(),
                stop=ipywidgets.FloatText(),
                num=ipywidgets.IntText(value=50),
                unit=ipywidgets.Text(value='angstrom')
            )
        '''
        if tp_kwargs is None:
            tp_kwargs = {}
        self.tp = tp
        self.args = tp_args
        self.kwargs = tp_kwargs
        self.out = ipw.Output()

        form = kwarg_layout(
            [
                arg_layout(tp_args),
                *(ipw.HBox([ipw.Label(value=k), v]) for k, v in tp_kwargs.items()),
            ]
        )
        layout = (
            dict(left_sidebar=form, center=self.out)  # noqa: C408
            if display_output
            else dict(center=form)  # noqa: C408
        )
        if 'description' in kwargs:
            layout["header"] = ipw.Label(value=kwargs['description'])

        super().__init__(
            **kwargs,
            **layout,
        )
        for a in (*tp_args, *tp_kwargs.values()):
            a.observe(self._recompute, names='value')

        self.observe(self._set_output, names='value')
        self._recompute()

    def _set_output(self, _=None):
        self.out.clear_output()
        with self.out:
            display(self.value)

    def _recompute(self, _=None):
        self.value = self.tp(
            *(v.value for v in self.args),
            **{k: v.value for k, v in self.kwargs.items()},
        )


def extract_children(widget):
    'Extracts underlying data from nested widget'
    if not hasattr(widget, 'children') or widget.children == ():
        if hasattr(widget, 'value'):
            return widget.value
        else:
            return ()
    return tuple(extract_children(child) for child in widget.children)


def insert_children(widget, children):
    'Inserts underlying data into nested widget'
    if not hasattr(widget, 'children') or widget.children == ():
        if hasattr(widget, 'value'):
            widget.value = children
            return
        else:
            return
    for wid, child in zip(widget.children, children, strict=True):
        insert_children(wid, child)
