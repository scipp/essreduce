import ess
import ipywidgets as widgets
import sciline
import scipp as sc
from ess import loki
from ess.reduce import workflow
from ess.reduce.widget import create_parameter_widget
from ess.sans.types import (
    BackgroundRun,
    # BeamCenter,
    # CorrectForGravity,
    EmptyBeamRun,
    Filename,
    # QBins,
    # ReturnEvents,
    SampleRun,
    TransmissionRun,
)

# UncertaintyBroadcastMode,
# WavelengthBands,
# WavelengthBins,
from IPython import display
from ipywidgets import Layout, TwoByTwoLayout

workflows = workflow.workflow_registry


_style = {
    'description_width': 'auto',
    'value_width': 'auto',
    'button_width': 'auto',
}
workflow_select = widgets.Dropdown(
    options=workflows,
    description='Workflow:',
    value=None,
)

typical_outputs_widget = widgets.SelectMultiple(
    description='Outputs:', layout=Layout(width='80%', height='150px')
)

possible_outputs_widget = widgets.SelectMultiple(
    description='Extended Outputs:',
    style=_style,
    layout=Layout(width='80%', height='150px'),
)


def handle_workflow_select(change):
    workflow = change.new()
    typical_outputs_widget.options = workflow.typical_outputs
    possible_outputs_widget.options = workflow.possible_outputs


workflow_select.observe(handle_workflow_select, names='value')

generate_parameter_button = widgets.Button(
    description='Generate Parameters',
    disabled=False,
    button_style='info',
    tooltip='Generate Parameters',
)

reset_button = widgets.Button(
    description='Reset',
    disabled=True,
    button_style='info',
    tooltip='Reset',
)

parameter_box = widgets.VBox([])


def generate_parameter_widgets():
    workflow = workflow_select.value()
    outputs = possible_outputs_widget.value + typical_outputs_widget.value
    registry = workflow.parameters(outputs)

    for parameter in registry.values():
        temp_widget = create_parameter_widget(parameter)
        temp = widgets.HBox([temp_widget])
        parameter_box.children = (*parameter_box.children, temp)


def on_button_clicked(b):
    generate_parameter_widgets()
    generate_parameter_button.disabled = True
    reset_button.disabled = False
    run_button.disabled = False
    output.clear_output()


generate_parameter_button.on_click(on_button_clicked)


def reset_button_clicked(b):
    generate_parameter_button.disabled = False
    reset_button.disabled = True
    parameter_box.children = []
    output.clear_output()


reset_button.on_click(reset_button_clicked)

button_box = widgets.HBox([generate_parameter_button, reset_button])
workflow_box = widgets.VBox(
    [workflow_select, typical_outputs_widget, possible_outputs_widget, button_box]
)

run_button = widgets.Button(
    description='Run',
    disabled=True,
    button_style='success',
    tooltip='Run',
)

output = widgets.Output()


def run_workflow(b):
    workflow = workflow_select.value()
    outputs = possible_outputs_widget.value + typical_outputs_widget.value
    registry = workflow.parameters(outputs)

    for i, node in enumerate(registry.keys()):
        workflow[node] = parameter_box.children[i].children[0].value

    # workflow[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    workflow[ess.sans.types.NeXusDetectorName] = 'larmor_detector'
    workflow[ess.sans.types.Filename[ess.sans.types.SampleRun]] = (
        loki.data.loki_tutorial_sample_run_60339()
    )
    workflow[ess.sans.types.PixelMaskFilename] = (
        loki.data.loki_tutorial_mask_filenames()[0]
    )

    workflow[Filename[SampleRun]] = loki.data.loki_tutorial_sample_run_60339()
    workflow[Filename[BackgroundRun]] = loki.data.loki_tutorial_background_run_60393()
    workflow[Filename[TransmissionRun[SampleRun]]] = (
        loki.data.loki_tutorial_sample_transmission_run()
    )
    workflow[Filename[TransmissionRun[BackgroundRun]]] = (
        loki.data.loki_tutorial_run_60392()
    )
    workflow[Filename[EmptyBeamRun]] = loki.data.loki_tutorial_run_60392()

    workflow[ess.sans.types.BeamCenter] = sc.vector(value=[0.0, 0.0, 0.0], unit='m')
    workflow[ess.sans.types.DirectBeam] = None

    # Wavelength binning parameters
    # wavelength_min = sc.scalar(1.0, unit='angstrom')
    # wavelength_max = sc.scalar(13.0, unit='angstrom')
    # n_wavelength_bins = 50
    # n_wavelength_bands = 50

    # workflow[WavelengthBins] = sc.linspace(
    #     'wavelength', wavelength_min, wavelength_max, n_wavelength_bins + 1
    # )
    # workflow[WavelengthBands] = sc.linspace(
    #     'wavelength', wavelength_min, wavelength_max, n_wavelength_bands + 1
    # )

    # workflow[CorrectForGravity] = True
    # workflow[UncertaintyBroadcastMode] = UncertaintyBroadcastMode.upper_bound
    # workflow[ReturnEvents] = False

    # workflow[QBins] = sc.linspace(dim='Q', start=0.01, stop=0.3,
    # num=101, unit='1/angstrom')

    with output:
        compute_result = workflow.compute(
            outputs, scheduler=sciline.scheduler.NaiveScheduler()
        )
        for i in compute_result.values():
            display.display(i.plot())


run_button.on_click(run_workflow)

# To render this in a voila server, create a notebook which imports this
# layout and then render it.
# from ess.reduce.ui import layout
# layout
layout = TwoByTwoLayout(
    top_left=workflow_box,
    bottom_left=widgets.VBox([run_button, output]),
    #   bottom_left=run_button,
    bottom_right=parameter_box,
)
