import ess
import ipywidgets as widgets
import sciline
from ess.reduce import workflow
from ess.reduce.widget import create_parameter_widget
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
    wf = change.new()
    typical_outputs_widget.options = workflow.get_typical_outputs(wf)
    possible_outputs_widget.options = workflow.get_possible_outputs(wf)


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
    wf = workflow_select.value()
    outputs = possible_outputs_widget.value + typical_outputs_widget.value
    registry = workflow.get_parameters(wf, outputs)

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
    wf = workflow_select.value()
    outputs = possible_outputs_widget.value + typical_outputs_widget.value
    registry = workflow.get_parameters(wf, outputs)

    values = {}

    for i, node in enumerate(registry.keys()):
        values[node] = parameter_box.children[i].children[0].value

    values[ess.sans.types.DirectBeam] = None

    wf = workflow.assign_parameter_values(wf, values)

    with output:
        compute_result = wf.compute(
            outputs, scheduler=sciline.scheduler.NaiveScheduler()
        )
        results.clear()
        results.update(compute_result)
        for i in compute_result.values():
            display.display(display.HTML(i._repr_html_()))


run_button.on_click(run_workflow)

# To render this in a voila server, create a notebook which imports this
# layout and then render it.
#     from ess.reduce import ui
#     ui.layout
# Results will be in ui.results
layout = TwoByTwoLayout(
    top_left=workflow_box,
    bottom_left=widgets.VBox([run_button, output]),
    #   bottom_left=run_button,
    bottom_right=parameter_box,
)
results = {}
