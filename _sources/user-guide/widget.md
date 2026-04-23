# Workflow Widget in Jupyter Notebook

Users can run a workflow with our widget in a jupyter notebook.

## Widget with All Workflows

``workflow_widget`` will build a widget that you can select workflows among ``ess.reduce.workflow.workflow_registry``.

**In order to select workflows from other ess packages, you will need to import the module that has the workflow constructor like the [Voila Example](#deploying-voila-application).**

## Deploying Voila Application
You need a jupyter notebook that contains these lines.


```python
from ess import loki  # loki module register workflows itself
from ess.reduce.ui import workflow_widget

ess_widget = workflow_widget()
```
And you can deploy the notebook as an application using voila command.
```bash
voila {PATH_TO_THE_NOTEBOOK}
```

## Widget from a Workflow in a Notebook
```python
%matplotlib widget
from ess.reduce.ui import workflow_widget

workflow_widget()
```
