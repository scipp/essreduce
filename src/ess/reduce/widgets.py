import ipywidgets as ipw
from IPython.display import display


class AutoWidget(ipw.ValueWidget, ipw.AppLayout):
    def __init__(self, tp, *args, **kwargs):
        '''Creates a widget from a constructor `tp` and a range of argument widgets

        Example usage:
            bound_widget = AutoWidget(
                lambda a, b: (sc.scalar(a), sc.scalar(b)),
                ipywidgets.FloatText(),
                ipywidgets.FloatText()
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
        self.tp = tp
        self.args = args
        self.kwargs = kwargs
        self.out = ipw.Output()

        super().__init__(
            center=ipw.VBox(
                [
                    *args,
                    *(ipw.HBox([ipw.Label(value=k), v]) for k, v in kwargs.items()),
                    self.out,
                ]
            ),
        )
        for a in (*args, *kwargs.values()):
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
