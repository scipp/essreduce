# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import Any, NewType

import pytest
import sciline as sl
import scipp as sc
from ipywidgets import FloatText, IntText

from ess.reduce.parameter import (
    BinEdgesParameter,
    FilenameParameter,
    MultiFilenameParameter,
    Parameter,
    Vector3dParameter,
    parameter_registry,
)
from ess.reduce.ui import ResultBox, WorkflowWidget, workflow_widget
from ess.reduce.widgets import OptionalWidget, SwitchWidget, create_parameter_widget
from ess.reduce.widgets._base import WidgetWithFieldsProtocol, get_fields, set_fields
from ess.reduce.workflow import register_workflow, workflow_registry

SwitchableInt = NewType('SwitchableInt', int)
SwitchableFloat = NewType('SwitchableFloat', float)
OptionalInt = int | None
OptionalFloat = float | None


class IntParam(Parameter): ...


class FloatParam(Parameter): ...


parameter_registry[SwitchableInt] = IntParam('_', '_', 1, switchable=True)
parameter_registry[SwitchableFloat] = FloatParam('_', '_', 2.0, switchable=True)
parameter_registry[int] = IntParam('_', '_', 1)
parameter_registry[float] = FloatParam('_', '_', 2.0)
parameter_registry[OptionalInt] = IntParam('_', '_', 1, optional=True)
parameter_registry[OptionalFloat] = FloatParam('_', '_', 2.0, optional=True)


@create_parameter_widget.register(IntParam)
def _(param: IntParam) -> IntText:
    return IntText(value=param.default, description=param.name)


@create_parameter_widget.register(FloatParam)
def _(param: FloatParam) -> FloatText:
    return FloatText(value=param.default, description=param.name)


def _refresh_widget_parameter(
    *, widget: WorkflowWidget, output_selections: list[type]
) -> None:
    widget.output_selection_box.typical_outputs_widget.value = output_selections
    widget.parameter_box.parameter_refresh_button.click()


def _ready_widget(
    *,
    providers: list[Callable] | None = None,
    params: dict[type, Any] | None = None,
    output_selections: list[type],
    result_registry: dict[type, Any] | None = None,
) -> WorkflowWidget:
    widget = WorkflowWidget(
        sl.Pipeline(providers or [], params=params or {}),
        result_registry=result_registry,
    )
    _refresh_widget_parameter(widget=widget, output_selections=output_selections)
    return widget


def strict_provider(a: int, b: float) -> str:
    return f"{a} + {b}"


def provider_with_switch(a: SwitchableInt, b: SwitchableFloat) -> str:
    return f"{a} + {b}"


def provider_with_optional(a: OptionalInt, b: OptionalFloat) -> str:
    parts = [] if a is None else [str(a)]
    return ' + '.join([*parts] if b is None else [*parts, str(b)])


def _get_param_widget(widget: WorkflowWidget, param_type: type) -> Any:
    return widget.parameter_box._input_widgets[param_type]


def test_parameter_default_value_test() -> None:
    widget = _ready_widget(providers=[strict_provider], output_selections=[str])
    assert _get_param_widget(widget, int).value == 1
    assert _get_param_widget(widget, float).value == 2.0


def test_parameter_registry() -> None:
    assert isinstance(create_parameter_widget(IntParam('_a', '_a', 1)), IntText)
    assert isinstance(create_parameter_widget(FloatParam('_b', '_b', 2.0)), FloatText)


def test_run_not_allowed_when_parameter_not_refreshed_after_output_selected() -> None:
    widget = _ready_widget(providers=[strict_provider], output_selections=[str])
    # Clear the value of the output selection box
    widget.output_selection_box.typical_outputs_widget.value = []
    assert widget.result_box.run_button.disabled
    # Click the refresh button
    widget.parameter_box.parameter_refresh_button.click()
    assert not widget.result_box.run_button.disabled
    # Add a value to the parameter
    widget.output_selection_box.typical_outputs_widget.value = [str]
    assert widget.result_box.run_button.disabled
    # Click the refresh button again
    widget.parameter_box.parameter_refresh_button.click()
    assert not widget.result_box.run_button.disabled


def test_result_registry() -> None:
    registry = {}
    widget = _ready_widget(
        providers=[strict_provider], output_selections=[str], result_registry=registry
    )
    _get_param_widget(widget, int).value = 2
    _get_param_widget(widget, float).value = 0.1
    assert registry == {}
    widget.result_box.run_button.click()
    assert registry == {str: '2 + 0.1'}


def test_switchable_widget_dispatch() -> None:
    switchable_param = Parameter('a', 'a', 1, switchable=True)
    assert isinstance(create_parameter_widget(switchable_param), SwitchWidget)
    non_switchable_param = Parameter('b', 'b', 2, switchable=False)
    assert not isinstance(create_parameter_widget(non_switchable_param), SwitchWidget)


def test_switchable_parameter_switch_widget() -> None:
    widget = _ready_widget(providers=[provider_with_switch], output_selections=[str])

    int_widget = _get_param_widget(widget, SwitchableInt)
    float_widget = _get_param_widget(widget, SwitchableFloat)

    assert isinstance(int_widget, SwitchWidget)
    assert isinstance(float_widget, SwitchWidget)

    assert not float_widget.enabled
    assert not int_widget.enabled


def test_collect_values_from_disabled_switchable_widget() -> None:
    widget = _ready_widget(providers=[provider_with_switch], output_selections=[str])

    assert not _get_param_widget(widget, SwitchableFloat).enabled
    assert not _get_param_widget(widget, SwitchableInt).enabled
    assert widget.parameter_box.value == {}


def test_collect_values_from_enabled_switchable_widget() -> None:
    widget = _ready_widget(providers=[provider_with_switch], output_selections=[str])

    float_widget = _get_param_widget(widget, SwitchableFloat)
    float_widget.enabled = True
    float_widget.value = 0.2

    assert widget.parameter_box.value == {SwitchableFloat: 0.2}


def test_switchable_optional_parameter_switchable_first() -> None:
    dummy_param = Parameter('a', 'a', 1, switchable=True, optional=True)
    dummy_widget = create_parameter_widget(dummy_param)
    assert isinstance(dummy_widget, SwitchWidget)
    assert isinstance(dummy_widget.wrapped, OptionalWidget)


def test_optional_widget_set_value_get_fields() -> None:
    optional_param = IntParam('a', 'a', 1, optional=True)
    optional_widget = create_parameter_widget(optional_param)
    assert isinstance(optional_widget, WidgetWithFieldsProtocol)
    assert isinstance(optional_widget, OptionalWidget)
    # Check initial state
    assert optional_widget._option_box.value is None
    assert get_fields(optional_widget) == {'essreduce-opted-out': True, 'value': 1}
    # Update the value of the wrapped widget and check the fields
    set_fields(optional_widget, {'value': 2})
    assert optional_widget.value is None  # Opted-out is not changed
    assert get_fields(optional_widget) == {'essreduce-opted-out': True, 'value': 2}
    optional_widget.value = 3
    assert get_fields(optional_widget) == {'essreduce-opted-out': False, 'value': 3}


def test_optional_widget_set_fields_get_fields() -> None:
    optional_param = Vector3dParameter(
        'a', 'a', sc.vector([1, 2, 3], unit='m'), optional=True
    )
    optional_widget = create_parameter_widget(optional_param)
    assert isinstance(optional_widget, WidgetWithFieldsProtocol)
    assert isinstance(optional_widget, OptionalWidget)
    # Check initial state
    assert optional_widget._option_box.value is None
    expected = {'essreduce-opted-out': True, 'x': 1, 'y': 2, 'z': 3, 'unit': 'm'}
    assert optional_widget.get_fields() == expected
    # Update the value of the wrapped widget
    optional_widget.set_fields(
        {'essreduce-opted-out': True, 'x': 4, 'y': 5, 'z': 6, 'unit': 'm'}
    )
    assert optional_widget.value is None  # Opted-out is not changed
    optional_widget.set_fields({'essreduce-opted-out': False})
    assert optional_widget.value == sc.vector([4, 5, 6], unit='m')
    # Check the fields and the option box value
    expected = {'essreduce-opted-out': False, 'x': 4, 'y': 5, 'z': 6, 'unit': 'm'}
    assert optional_widget.get_fields() == expected
    assert optional_widget._option_box.value == optional_param.name


def test_optional_widget_dispatch() -> None:
    optional_param = Parameter('a', 'a', 1, optional=True)
    assert isinstance(create_parameter_widget(optional_param), OptionalWidget)
    non_optional_param = Parameter('b', 'b', 2, optional=False)
    assert not isinstance(create_parameter_widget(non_optional_param), OptionalWidget)


def test_optional_parameter_optional_widget() -> None:
    widget = _ready_widget(providers=[provider_with_optional], output_selections=[str])

    int_widget = _get_param_widget(widget, OptionalInt)
    float_widget = _get_param_widget(widget, OptionalFloat)

    assert isinstance(int_widget, OptionalWidget)
    assert isinstance(float_widget, OptionalWidget)

    assert float_widget.value is None
    assert int_widget.value is None


def test_collect_values_from_optional_widget() -> None:
    widget = _ready_widget(providers=[provider_with_optional], output_selections=[str])

    float_widget = _get_param_widget(widget, OptionalFloat)
    float_widget.value = 0.2

    assert widget.parameter_box.value == {OptionalFloat: 0.2, OptionalInt: None}


def test_collect_values_from_optional_widget_compute_result() -> None:
    result_registry = {}
    widget = _ready_widget(
        providers=[provider_with_optional],
        output_selections=[str],
        result_registry=result_registry,
    )

    float_widget = _get_param_widget(widget, OptionalFloat)
    float_widget.value = 0.2
    widget.result_box.run_button.click()

    assert result_registry == {str: '0.2'}

    int_widget = _get_param_widget(widget, OptionalInt)
    int_widget.value = 2
    widget.result_box.run_button.click()

    assert result_registry == {str: '2 + 0.2'}


def dummy_workflow_constructor() -> sl.Pipeline:
    return sl.Pipeline([strict_provider])


@contextmanager
def temporary_workflow_registry(
    *constructors: Callable[[], sl.Pipeline],
) -> Generator[None, None, None]:
    existance_flags = {
        constructor: constructor in workflow_registry for constructor in constructors
    }
    for constructor in constructors:
        register_workflow(constructor)
    yield
    for constructor, flag in existance_flags.items():
        if not flag:
            workflow_registry.discard(constructor)


def test_register_workflow() -> None:
    with temporary_workflow_registry(dummy_workflow_constructor):
        assert dummy_workflow_constructor in workflow_registry

    assert dummy_workflow_constructor not in workflow_registry


def _get_selection_widget(widget):
    return widget.children[0].children[0]


def test_workflow_registry_applied_to_selector() -> None:
    expected_constructor_pair = (
        'dummy_workflow_constructor',
        dummy_workflow_constructor,
    )
    with temporary_workflow_registry(dummy_workflow_constructor):
        selection_widget = _get_selection_widget(workflow_widget())
        assert expected_constructor_pair in selection_widget.options

    selection_widget = _get_selection_widget(workflow_widget())
    assert expected_constructor_pair not in selection_widget.options


def dummy_second_workflow_constructor() -> sl.Pipeline:
    return sl.Pipeline([provider_with_switch])


def test_workflow_selection() -> None:
    # Prepare
    with temporary_workflow_registry(
        dummy_workflow_constructor, dummy_second_workflow_constructor
    ):
        widget = workflow_widget()
        selection_widget = _get_selection_widget(widget)
        # Before selection
        assert len(widget.children[1].children) == 0
        # Select first workflow
        selection_widget.value = dummy_workflow_constructor
        assert len(widget.children[1].children) == 1
        # Test created WorkflowWidget
        first_widget = widget.children[1].children[0]
        assert isinstance(first_widget, WorkflowWidget)
        assert first_widget.output_selection_box.typical_outputs_widget.options == (
            ('str', str),
        )
        _refresh_widget_parameter(widget=first_widget, output_selections=[str])
        assert first_widget.parameter_box._input_widgets.keys() == {int, float}
        # Select second workflow
        selection_widget.value = dummy_second_workflow_constructor
        second_widget = widget.children[1].children[0]
        _refresh_widget_parameter(widget=second_widget, output_selections=[str])
        assert second_widget.parameter_box._input_widgets.keys() == {
            SwitchableInt,
            SwitchableFloat,
        }


WavelengthBins = NewType('WavelengthBins', sc.Variable)
WavelengthBinsWithUnit = NewType('WavelengthBinsWithUnit', sc.Variable)
QBins = NewType('QBins', sc.Variable)
TemperatureBinsWithUnits = NewType('TemperatureBinsWithUnits', sc.Variable)


parameter_registry[WavelengthBins] = BinEdgesParameter(WavelengthBins, dim='wavelength')
parameter_registry[WavelengthBinsWithUnit] = BinEdgesParameter(
    WavelengthBinsWithUnit, dim='wavelength', unit='m'
)
parameter_registry[QBins] = BinEdgesParameter(
    QBins, dim='Q', start=0.01, stop=0.6, nbins=150
)
parameter_registry[TemperatureBinsWithUnits] = BinEdgesParameter(
    TemperatureBinsWithUnits, dim='temperature', unit=('Kelvin', 'Celsius', 'Degrees')
)


def wavelength_bins_print_provider(bins: WavelengthBins) -> str:
    return str(bins)


def wavelength_bins_with_unit_print_provider(bins: WavelengthBinsWithUnit) -> str:
    return str(bins)


def q_bins_print_provider(bins: QBins) -> str:
    return str(bins)


def temperature_bins_print_provider(bins: TemperatureBinsWithUnits) -> str:
    return str(bins)


def test_bin_edges_widget_simple() -> None:
    widget = _ready_widget(
        providers=[wavelength_bins_print_provider], output_selections=[str]
    )
    param_widget = _get_param_widget(widget, WavelengthBins)
    assert sc.identical(
        param_widget.value,
        sc.linspace(dim='wavelength', start=0.0, stop=0.0, num=2, unit='angstrom'),
    )
    assert set(param_widget.fields['unit'].options) == {'angstrom', 'nm'}
    assert param_widget.fields['nbins'].value == 1
    assert param_widget.fields['spacing'].value == 'linear'
    # Modify the values
    param_widget.fields['start'].value = 1.0
    param_widget.fields['stop'].value = 200.0
    param_widget.fields['nbins'].value = 10
    assert sc.identical(
        param_widget.value,
        sc.linspace(dim='wavelength', start=1.0, stop=200.0, num=11, unit='angstrom'),
    )


def test_bin_edges_widget_override_unit() -> None:
    widget = _ready_widget(
        providers=[wavelength_bins_with_unit_print_provider], output_selections=[str]
    )
    param_widget = _get_param_widget(widget, WavelengthBinsWithUnit)
    assert sc.identical(
        param_widget.value,
        sc.linspace(dim='wavelength', start=0.0, stop=0.0, num=2, unit='m'),
    )
    assert set(param_widget.fields['unit'].options) == {'m'}
    assert param_widget.fields['nbins'].value == 1


def test_bin_edges_widget_override_unit_multiple() -> None:
    widget = _ready_widget(
        providers=[temperature_bins_print_provider], output_selections=[str]
    )
    param_widget = _get_param_widget(widget, TemperatureBinsWithUnits)
    assert sc.identical(
        param_widget.value,
        sc.linspace(dim='temperature', start=0.0, stop=0.0, num=2, unit='Kelvin'),
    )
    assert set(param_widget.fields['unit'].options) == {'Kelvin', 'Celsius', 'Degrees'}


def test_bin_edges_widget_log_spacing() -> None:
    widget = _ready_widget(
        providers=[wavelength_bins_print_provider], output_selections=[str]
    )
    param_widget = _get_param_widget(widget, WavelengthBins)
    param_widget.fields['spacing'].value = 'log'
    param_widget.fields['start'].value = 1.0e3
    param_widget.fields['stop'].value = 1.0e5
    param_widget.fields['nbins'].value = 7
    assert sc.identical(
        param_widget.value,
        sc.geomspace(dim='wavelength', start=1.0e3, stop=1.0e5, num=8, unit='angstrom'),
    )
    assert param_widget.fields['spacing'].value == 'log'


def test_bin_edges_widget_with_default_values() -> None:
    widget = _ready_widget(providers=[q_bins_print_provider], output_selections=[str])
    param_widget = _get_param_widget(widget, QBins)
    assert sc.identical(
        param_widget.value,
        sc.linspace(dim='Q', start=0.01, stop=0.6, num=151, unit='1/angstrom'),
    )
    assert set(param_widget.fields['unit'].options) == {'1/angstrom', '1/nm'}
    assert param_widget.fields['start'].value == 0.01
    assert param_widget.fields['stop'].value == 0.6
    assert param_widget.fields['nbins'].value == 150
    assert param_widget.fields['spacing'].value == 'linear'


@pytest.mark.parametrize(
    'output',
    [
        (sc.scalar(1), sc.scalar(2)),
        'Test with a string',
        sc.data.binned_xy(100, 10, 10),
    ],
)
def test_result_box_can_handle_different_outputs(output):
    was_called = False

    def run_workflow():
        nonlocal was_called
        was_called = True
        return dict(enumerate(output))

    ResultBox(run_workflow).run_button.click()
    assert was_called


def test_switchable_widget_set_values() -> None:
    param = IntParam('a', 'a', 1, switchable=True)
    widget = create_parameter_widget(param)
    assert isinstance(widget, SwitchWidget)
    assert not widget.enabled
    widget.set_fields({'enabled': True})
    assert widget.enabled
    widget.set_fields({'enabled': False})
    assert not widget.enabled
    widget.set_fields({'enabled': True, 'value': 2})
    assert widget.enabled
    assert widget.value == 2
    widget.set_fields({'enabled': False, 'value': 3})
    assert not widget.enabled
    assert widget.value == 3
    widget.set_fields({'value': 4})
    assert not widget.enabled
    assert widget.value == 4


def test_switchable_widget_get_fields_only_value() -> None:
    param = IntParam('a', 'a', 1, switchable=True)
    widget = create_parameter_widget(param)
    assert isinstance(widget, SwitchWidget)
    assert widget.get_fields() == {'enabled': False, 'value': 1}
    widget.enabled = True
    assert widget.get_fields() == {'enabled': True, 'value': 1}
    widget.value = 2
    assert widget.get_fields() == {'enabled': True, 'value': 2}
    widget.enabled = False
    assert widget.get_fields() == {'enabled': False, 'value': 2}


def test_switchable_widget_set_fields() -> None:
    param = Vector3dParameter('a', 'a', sc.vector([1, 2, 3], unit='m'), switchable=True)
    widget = create_parameter_widget(param)
    assert isinstance(widget, SwitchWidget)
    assert not widget.enabled
    assert widget.value == sc.vector([1, 2, 3], unit='m')
    widget.set_fields({'enabled': True, 'x': 4, 'y': 5, 'z': 6, 'unit': 'm'})
    assert widget.enabled
    assert widget.value == sc.vector([4, 5, 6], unit='m')
    widget.set_fields({'x': 7, 'y': 8})
    assert widget.enabled
    assert widget.value == sc.vector([7, 8, 6], unit='m')


def test_switchable_widget_get_fields_sub_fields() -> None:
    param = Vector3dParameter('a', 'a', sc.vector([1, 2, 3], unit='m'), switchable=True)
    widget = create_parameter_widget(param)
    assert isinstance(widget, SwitchWidget)
    assert widget.get_fields() == {
        'enabled': False,
        'x': 1,
        'y': 2,
        'z': 3,
        'unit': 'm',
    }
    widget.enabled = True
    assert widget.get_fields() == {'enabled': True, 'x': 1, 'y': 2, 'z': 3, 'unit': 'm'}
    widget.set_fields({'enabled': False, 'x': 4, 'y': 5, 'unit': 'mm'})
    assert widget.get_fields() == {
        'enabled': False,
        'x': 4,
        'y': 5,
        'z': 3,
        'unit': 'mm',
    }


FilenameSampleRun = NewType('FilenameSampleRun', str)
parameter_registry[FilenameSampleRun] = FilenameParameter.from_type(
    FilenameSampleRun, default="SampleRun.hdf"
)
MultiFilenameSampleRun = NewType('MultiFilenameSampleRun', list[str])
parameter_registry[MultiFilenameSampleRun] = MultiFilenameParameter.from_type(
    MultiFilenameSampleRun, default=["file1.hdf", "file2.hdf"]
)
MultiFilenameBackgroundRun = NewType('MultiFilenameBackgroundRun', list[str])
parameter_registry[MultiFilenameBackgroundRun] = MultiFilenameParameter.from_type(
    MultiFilenameBackgroundRun, default="background.hdf"
)


def filename_print_provider(fname: FilenameSampleRun) -> str:
    return str(fname)


def multi_filename_print_provider(fnames: MultiFilenameSampleRun) -> list[str]:
    return [str(fname) for fname in fnames]


def multi_filename_print_provider_bis(fnames: MultiFilenameBackgroundRun) -> list[str]:
    return [str(fname) for fname in fnames]


def test_filename_widget_single() -> None:
    widget = _ready_widget(providers=[filename_print_provider], output_selections=[str])
    param_widget = _get_param_widget(widget, FilenameSampleRun)
    assert param_widget.value == "SampleRun.hdf"


def test_filename_widget_multiple() -> None:
    widget = _ready_widget(
        providers=[multi_filename_print_provider], output_selections=[list[str]]
    )
    param_widget = _get_param_widget(widget, MultiFilenameSampleRun)
    assert param_widget.value == ("file1.hdf", "file2.hdf")


def test_filename_widget_multiple_initialized_with_single_string() -> None:
    widget = _ready_widget(
        providers=[multi_filename_print_provider_bis], output_selections=[list[str]]
    )
    param_widget = _get_param_widget(widget, MultiFilenameBackgroundRun)
    assert param_widget.value == ("background.hdf",)
