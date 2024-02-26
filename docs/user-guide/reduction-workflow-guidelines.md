# Reduction Workflow Guidelines

## About

- Version: 1
- Last update: 2024-02-26

## Introduction

This document contains guidelines for writing reduction workflows for ESS based on Scipp and Sciline.
The guidelines are intended to ensure that the workflows are consistent (both for developers and users, across instruments and techniques), maintainable, and efficient.

## To be included in future version

We plan to include the following in future versions of the guidelines:

- Naming conventions
- Type conventions
  - Example: Filenames
- Package and module structure:
  - Where to place types?
    What goes where?
- Loading from SciCat vs. local files
  - Example: Define run ID, choose provider that either converts to local path, or uses service to get file and return path
- Should we have default params set in workflows?
  - Avoid unless good reason.
  - Can have widgets that generate dict of params and values, widgets can have defaults
- How to define parameters, such that we can, e.g., auto generate widgets for user input (names, description, limits, default values, ...)
  - Range checks / validators
  - If part of pipeline then UX and writing providers is more cumbersome
  - Default values?
- Requires experimentation with how Sciline handles param tables, and transformations of task graphs:
  - Multiple banks, multiple files, chunking (file-based + stream-based)
  - How to handle optional steps
  - Structure for masking any dim or transformed dim, in various steps
    - Could be handled as a task-graph transform?
- How to handle optional inputs?
  - Can we find a way to minimize the occasions where we need this?
  - Can we avoid mutually exclusive parameters?


## Nomenclature

- *Provider*: A callable step in a workflow writing with Sciline.

## D: Documentation

### D.1: Document math and references in docstrings

**Reason**
Documentation should be as close to the code as possible, to decrease the chance that it runs out of sync.
This includes mathematical formulas and references to literature.

**Note**
We have previously documented math and references in Jupyter notebooks.
This is not sufficient, as the documentation is not close to the code.

## N: Naming

## P: Performance

### P.1: Runtime and memory use of workflows shall be tested with large data

**Reason**
We want to ensure that the workflows are efficient and do not consume excessive memory.

**Note**
This is often not apparent from small test data, as the location of performance bottlenecks may depend on the size of the data.

## S: Structure

### S.1: Workflows shall be able to return final results as event data

**Reason**
- Required for polarization analysis, which wraps a base workflow.
- Required for subsequent filtering, unless part of the workflow.

**Note**
There should be a workflow parameter (flag) to select whether to return event data or not.

### S.2: Split loading and handling of monitors and detectors from loading of auxiliary data and metadata

**Reason**
- Allows for more efficient parallelism and reduction in memory use.
- Avoids loading large data into memory that is not needed for the reduction.
- Avoids keeping large data alive in memory if output metadata extraction depends on auxiliary input data or input metadata.

### S.3: Avoid dependencies of output metadata on large data

**Reason**
Adding dependencies on large data to the output metadata extraction may lead to large data being kept alive in memory.

**Note**
Most of this is avoided by following S.2.
A bad example would be writing the total raw counts to the output metadata, as this would require keeping the large data alive in memory, unless it is ensured that the task runs early.

### S.4: Preserve floating-point precision of input data and coordinates

**Reason**
Single-precision may be sufficient for most data.
By writing workflows transparently for single- and double-precision, we avoid future changes if we either want to use single-precision for performance reasons or double-precision for accuracy reasons.

**Note**
This affects coordinates and data values independently.
- If input counts are single-precision, the reduced intensity should be single-precision, and equivalently for double-precision.
- If input coordinates are single-precision, derived coordinates should be single-precision, and equivalently for double-precision.

**Note**
This will allow for changing the precision of the entire workflow by choosing a precision when loading the input data.

**Example**
- If time-of-flight is single-precision, wavelength and momentum transfer should be single-precision.
- If counts are single-precision, reduced intensity should be single-precision.

### S.5: Switches to double-precision shall be deliberate, explicit, and documented

**Reason**
Some workflows may require switching to double-precision at a certain point in the workflow.
This should be a deliberate choice, and the reason for the switch should be documented.

### S.6: Propagation of uncertainties in broadcast operations should support "drop" and "upper-bound" strategies, "upper-bound" shall be the default

**Reason**
Unless explicitly computed, the exact propagation of uncertainties in broadcast operations is not tractable.
Dropping uncertainties is not desirable in general, as it may lead to underestimation of the uncertainties, but we realize that the upper-bound approach may not be suitable in all cases.
We should therefore support two strategies, "drop" and "upper-bound", and "upper-bound" should be the default.

**Note**
See [Systematic underestimation of uncertainties by widespread neutron-scattering data-reduction software](http://dx.doi.org/10.3233/JNR-220049) for a discussion of the topic.
TODO Add reference to upper-bound approach.

### S.7: Do not write files or make write requests to services such as SciCat in providers

**Reason**
Providers should be side-effect free, and should not write files or make write requests to services.

**Note**
Workflows may run many times, or in parallel, or tasks may be retried after failure, and we want to avoid side-effects in these cases.
This will, e.g., avoid unintentional overwriting of a user's files.

### S.8: Detector banks shall be loaded with their logical dimensions, if possible

**Reason**
Using logical dims (instead of a flat list of pixels) allows for simpler indexing and slicing of the data, reductions over a subset of dimensions, and masking of physical components.

**Note**
This is not always possible, as some detectors have an irregular structure and cannot be reshaped to a (multi-dimensional) array.

## T: Testing

### T.1: Adherence to the guidelines shall be tested, and the guideline ID shall be referenced in the test name

**Reason**
We want to ensure that the guidelines are followed, and that this remains the case as the code base evolves.
Referencing the guideline ID in the test name makes it easier to find the relevant guideline (or vice-versa), or remove the test if the guideline is removed.

**Note**
Not all guidelines are testable.

### T.2: Write unit tests for providers

**Reason** Unit tests for providers are easier to write and maintain than for entire workflows.

**Note** This does not mean that we should not write tests for entire workflows, but that we should also write tests for providers.