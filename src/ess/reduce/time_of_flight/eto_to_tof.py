# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Time-of-flight workflow for unwrapping the time of arrival of the neutron at the
detector.
This workflow is used to convert raw detector data with event_time_zero and
event_time_offset coordinates to data with a time-of-flight coordinate.
"""

from collections.abc import Callable

import numpy as np
import scipp as sc
import scippneutron as scn
import scippnexus as snx
from scippneutron._utils import elem_unit

try:
    from .interpolator_numba import Interpolator as InterpolatorImpl
except ImportError:
    from .interpolator_scipy import Interpolator as InterpolatorImpl

from ..nexus.types import (
    EmptyDetector,
    EmptyMonitor,
    GravityVector,
    MonitorType,
    Position,
    RawDetector,
    RawMonitor,
    RunType,
)
from .resample import rebin_strictly_increasing
from .types import (
    DetectorLtotal,
    MonitorLtotal,
    PulseStrideOffset,
    TimeOfFlightLookupTable,
    ToaDetector,
    TofDetector,
    TofMonitor,
)


class TofInterpolator:
    def __init__(self, lookup: sc.DataArray, distance_unit: str, time_unit: str):
        self._distance_unit = distance_unit
        self._time_unit = time_unit

        self._time_edges = (
            lookup.coords["event_time_offset"]
            .to(unit=self._time_unit, copy=False)
            .values
        )
        self._distance_edges = (
            lookup.coords["distance"].to(unit=distance_unit, copy=False).values
        )

        self._interpolator = InterpolatorImpl(
            time_edges=self._time_edges,
            distance_edges=self._distance_edges,
            values=lookup.data.to(unit=self._time_unit, copy=False).values,
        )

    def __call__(
        self,
        ltotal: sc.Variable,
        event_time_offset: sc.Variable,
        pulse_period: sc.Variable,
        pulse_index: sc.Variable | None = None,
    ) -> sc.Variable:
        if ltotal.unit != self._distance_unit:
            raise sc.UnitError(
                f"ltotal must have unit: {self._distance_unit}, "
                f"but got unit: {ltotal.unit}."
            )
        if event_time_offset.unit != self._time_unit:
            raise sc.UnitError(
                f"event_time_offset must have unit: {self._time_unit}, "
                f"but got unit: {event_time_offset.unit}."
            )
        out_dims = event_time_offset.dims
        ltotal = ltotal.values
        event_time_offset = event_time_offset.values

        return sc.array(
            dims=out_dims,
            values=self._interpolator(
                times=event_time_offset,
                distances=ltotal,
                pulse_index=pulse_index.values if pulse_index is not None else None,
                pulse_period=pulse_period.value,
            ),
            unit=self._time_unit,
        )


def _time_of_flight_data_histogram(
    da: sc.DataArray, lookup: sc.DataArray, ltotal: sc.Variable
) -> sc.DataArray:
    # In NeXus, 'time_of_flight' is the canonical name in NXmonitor, but in some files,
    # it may be called 'tof' or 'frame_time'.
    key = next(iter(set(da.coords.keys()) & {"time_of_flight", "tof", "frame_time"}))
    raw_eto = da.coords[key].to(dtype=float, copy=False)
    eto_unit = raw_eto.unit
    pulse_period = lookup.coords["pulse_period"].to(unit=eto_unit)

    # In histogram mode, because there is a wrap around at the end of the pulse, we
    # need to insert a bin edge at that exact location to avoid having the last bin
    # with one finite left edge and a NaN right edge (it becomes NaN as it would be
    # outside the range of the lookup table).
    new_bins = sc.sort(
        sc.concat([raw_eto, sc.scalar(0.0, unit=eto_unit), pulse_period], dim=key),
        key=key,
    )
    rebinned = da.rebin({key: new_bins})
    etos = rebinned.coords[key]

    # Create linear interpolator
    interp = TofInterpolator(lookup, distance_unit=ltotal.unit, time_unit=eto_unit)

    # Compute time-of-flight of the bin edges using the interpolator
    tofs = interp(
        ltotal=ltotal.broadcast(sizes=etos.sizes),
        event_time_offset=etos,
        pulse_period=pulse_period,
    )

    return rebinned.assign_coords(tof=tofs).drop_coords(
        list({key} & {"time_of_flight", "frame_time"})
    )


def _guess_pulse_stride_offset(
    pulse_index: sc.Variable,
    ltotal: sc.Variable,
    event_time_offset: sc.Variable,
    pulse_period: sc.Variable,
    pulse_stride: int,
    interp: TofInterpolator,
) -> int:
    """
    Using the minimum ``event_time_zero`` to calculate a reference time when computing
    the time-of-flight for the neutron events makes the workflow depend on when the
    first event was recorded. There is no straightforward way to know if we started
    recording at the beginning of a frame, or half-way through a frame, without looking
    at the chopper logs. This can be manually corrected using the pulse_stride_offset
    parameter, but this makes automatic reduction of the data difficult.
    See https://github.com/scipp/essreduce/issues/184.

    Here, we perform a simple guess for the ``pulse_stride_offset`` if it is not
    provided.
    We choose a few random events, compute the time-of-flight for every possible value
    of pulse_stride_offset, and return the value that yields the least number of NaNs
    in the computed time-of-flight.

    Parameters
    ----------
    pulse_index:
        Pulse index for every event.
    ltotal:
        Total length of the flight path from the source to the detector for each event.
    event_time_offset:
        Time of arrival of the neutron at the detector for each event.
    pulse_period:
        Period of the source pulses, i.e., time between consecutive pulse starts.
    pulse_stride:
        Stride of used pulses.
    interp:
        Interpolator for the lookup table.
    """
    tofs = {}
    # Choose a few random events to compute the time-of-flight
    inds = np.random.choice(
        len(event_time_offset), min(5000, len(event_time_offset)), replace=False
    )
    pulse_index = sc.array(
        dims=pulse_index.dims,
        values=pulse_index.values[inds],
        unit=pulse_index.unit,
    )
    ltotal = sc.array(dims=ltotal.dims, values=ltotal.values[inds], unit=ltotal.unit)
    etos = sc.array(
        dims=event_time_offset.dims,
        values=event_time_offset.values[inds],
        unit=event_time_offset.unit,
    )
    for i in range(pulse_stride):
        pulse_inds = (pulse_index + i) % pulse_stride
        tofs[i] = interp(
            ltotal=ltotal,
            event_time_offset=etos,
            pulse_index=pulse_inds,
            pulse_period=pulse_period,
        )
    # Find the entry in the list with the least number of nan values
    return sorted(tofs, key=lambda x: sc.isnan(tofs[x]).sum())[0]


def _prepare_tof_interpolation_inputs(
    da: sc.DataArray,
    lookup: sc.DataArray,
    ltotal: sc.Variable,
    pulse_stride_offset: int | None,
) -> dict:
    """
    Prepare the inputs required for the time-of-flight interpolation.
    This function is used when computing the time-of-flight for event data, and for
    computing the time-of-arrival for event data (as they both require guessing the
    pulse_stride_offset if not provided).

    Parameters
    ----------
    da:
        Data array with event data.
    lookup:
        Lookup table giving time-of-flight as a function of distance and time of
        arrival.
    ltotal:
        Total length of the flight path from the source to the detector.
    pulse_stride_offset:
        When pulse-skipping, the offset of the first pulse in the stride. This is
        typically zero but can be a small integer < pulse_stride.
        If None, a guess is made.
    """
    etos = da.bins.coords["event_time_offset"].to(dtype=float, copy=False)
    eto_unit = elem_unit(etos)

    # Create linear interpolator
    interp = TofInterpolator(lookup, distance_unit=ltotal.unit, time_unit=eto_unit)

    # Operate on events (broadcast distances to all events)
    ltotal = sc.bins_like(etos, ltotal).bins.constituents["data"]
    etos = etos.bins.constituents["data"]

    pulse_index = None
    pulse_period = lookup.coords["pulse_period"].to(unit=eto_unit)
    pulse_stride = lookup.coords["pulse_stride"].value

    if pulse_stride > 1:
        # Compute a pulse index for every event: it is the index of the pulse within a
        # frame period. The index ranges from zero to pulse_stride - 1.
        etz_unit = 'ns'
        etz = (
            da.bins.coords["event_time_zero"]
            .bins.constituents["data"]
            .to(unit=etz_unit, copy=False)
        )
        pulse_period_ns = pulse_period.to(unit=etz_unit, dtype=int)
        frame_period = pulse_period_ns * pulse_stride
        # Define a common reference time using epoch as a base, but making sure that it
        # is aligned with the pulse_period and the frame_period.
        # We need to use a global reference time instead of simply taking the minimum
        # event_time_zero because the events may arrive in chunks, and the first event
        # may not be the first event of the first pulse for all chunks. This would lead
        # to inconsistent pulse indices.
        epoch = sc.datetime(0, unit=etz_unit)
        diff_to_epoch = (etz.min() - epoch) % pulse_period_ns
        # Here we offset the reference by half a pulse period to avoid errors from
        # fluctuations in the event_time_zeros in the data. They are triggered by the
        # neutron source, and may not always be exactly separated by the pulse period.
        # While fluctuations will exist, they will be small, and offsetting the times
        # by half a pulse period is a simple enough fix.
        reference = epoch + diff_to_epoch - (pulse_period_ns // 2)
        # Use in-place operations to avoid large allocations
        pulse_index = etz - reference
        pulse_index %= frame_period
        pulse_index //= pulse_period_ns

        # Apply the pulse_stride_offset
        if pulse_stride_offset is None:
            pulse_stride_offset = _guess_pulse_stride_offset(
                pulse_index=pulse_index,
                ltotal=ltotal,
                event_time_offset=etos,
                pulse_period=pulse_period,
                pulse_stride=pulse_stride,
                interp=interp,
            )
        pulse_index += pulse_stride_offset
        pulse_index %= pulse_stride

    return {
        "eto": etos,
        "pulse_index": pulse_index,
        "pulse_period": pulse_period,
        "interp": interp,
        "ltotal": ltotal,
    }


def _time_of_flight_data_events(
    da: sc.DataArray,
    lookup: sc.DataArray,
    ltotal: sc.Variable,
    pulse_stride_offset: int | None,
) -> sc.DataArray:
    inputs = _prepare_tof_interpolation_inputs(
        da=da,
        lookup=lookup,
        ltotal=ltotal,
        pulse_stride_offset=pulse_stride_offset,
    )

    # Compute time-of-flight for all neutrons using the interpolator
    tofs = inputs["interp"](
        ltotal=inputs["ltotal"],
        event_time_offset=inputs["eto"],
        pulse_index=inputs["pulse_index"],
        pulse_period=inputs["pulse_period"],
    )

    parts = da.bins.constituents
    parts["data"] = tofs
    result = da.bins.assign_coords(tof=sc.bins(**parts, validate_indices=False))
    return result.bins.drop_coords("event_time_offset")


def detector_ltotal_from_straight_line_approximation(
    detector: EmptyDetector[RunType],
    source_position: Position[snx.NXsource, RunType],
    sample_position: Position[snx.NXsample, RunType],
    gravity: GravityVector,
) -> DetectorLtotal[RunType]:
    """Compute Ltotal for the detector pixels.

    This is a naive straight-line approximation to Ltotal based on basic component
    positions.

    Parameters
    ----------
    detector:
        Data array with detector positions.
    source_position:
        Position of the neutron source.
    sample_position:
        Position of the sample.
    gravity:
        Gravity vector.
    """
    graph = {
        **scn.conversion.graph.beamline.beamline(scatter=True),
        'source_position': lambda: source_position,
        'sample_position': lambda: sample_position,
        'gravity': lambda: gravity,
    }
    return DetectorLtotal[RunType](
        detector.transform_coords(
            "Ltotal", graph=graph, keep_intermediate=False
        ).coords["Ltotal"]
    )


def monitor_ltotal_from_straight_line_approximation(
    monitor_beamline: EmptyMonitor[RunType, MonitorType],
) -> MonitorLtotal[RunType, MonitorType]:
    """
    Compute Ltotal for the monitor.
    This is a naive straight-line approximation to Ltotal based on basic component
    positions.

    Parameters
    ----------
    monitor_beamline:
        Beamline data for the monitor that contains the positions necessary to compute
        the straight-line approximation to Ltotal (source and monitor positions).
    """
    graph = scn.conversion.graph.beamline.beamline(scatter=False)
    return MonitorLtotal[RunType, MonitorType](
        monitor_beamline.transform_coords(
            "Ltotal", graph=graph, keep_intermediate=False
        ).coords["Ltotal"]
    )


def _compute_tof_data(
    da: sc.DataArray,
    lookup: sc.DataArray,
    ltotal: sc.Variable,
    pulse_stride_offset: int,
) -> sc.DataArray:
    if da.bins is None:
        data = _time_of_flight_data_histogram(da=da, lookup=lookup, ltotal=ltotal)
        return rebin_strictly_increasing(data, dim='tof')
    else:
        return _time_of_flight_data_events(
            da=da,
            lookup=lookup,
            ltotal=ltotal,
            pulse_stride_offset=pulse_stride_offset,
        )


def detector_time_of_flight_data(
    detector_data: RawDetector[RunType],
    lookup: TimeOfFlightLookupTable,
    ltotal: DetectorLtotal[RunType],
    pulse_stride_offset: PulseStrideOffset,
) -> TofDetector[RunType]:
    """
    Convert the time-of-arrival data to time-of-flight data using a lookup table.
    The output data will have a time-of-flight coordinate.

    Parameters
    ----------
    da:
        Raw detector data loaded from a NeXus file, e.g., NXdetector containing
        NXevent_data.
    lookup:
        Lookup table giving time-of-flight as a function of distance and time of
        arrival.
    ltotal:
        Total length of the flight path from the source to the detector.
    pulse_stride_offset:
        When pulse-skipping, the offset of the first pulse in the stride. This is
        typically zero but can be a small integer < pulse_stride.
    """
    return TofDetector[RunType](
        _compute_tof_data(
            da=detector_data,
            lookup=lookup,
            ltotal=ltotal,
            pulse_stride_offset=pulse_stride_offset,
        )
    )


def monitor_time_of_flight_data(
    monitor_data: RawMonitor[RunType, MonitorType],
    lookup: TimeOfFlightLookupTable,
    ltotal: MonitorLtotal[RunType, MonitorType],
    pulse_stride_offset: PulseStrideOffset,
) -> TofMonitor[RunType, MonitorType]:
    """
    Convert the time-of-arrival data to time-of-flight data using a lookup table.
    The output data will have a time-of-flight coordinate.

    Parameters
    ----------
    da:
        Raw monitor data loaded from a NeXus file, e.g., NXmonitor containing
        NXevent_data.
    lookup:
        Lookup table giving time-of-flight as a function of distance and time of
        arrival.
    ltotal:
        Total length of the flight path from the source to the monitor.
    pulse_stride_offset:
        When pulse-skipping, the offset of the first pulse in the stride. This is
        typically zero but can be a small integer < pulse_stride.
    """
    return TofMonitor[RunType, MonitorType](
        _compute_tof_data(
            da=monitor_data,
            lookup=lookup,
            ltotal=ltotal,
            pulse_stride_offset=pulse_stride_offset,
        )
    )


def detector_time_of_arrival_data(
    detector_data: RawDetector[RunType],
    lookup: TimeOfFlightLookupTable,
    ltotal: DetectorLtotal[RunType],
    pulse_stride_offset: PulseStrideOffset,
) -> ToaDetector[RunType]:
    """
    Convert the time-of-flight data to time-of-arrival data using a lookup table.
    The output data will have a time-of-arrival coordinate.
    The time-of-arrival is the time since the neutron was emitted from the source.
    It is basically equal to event_time_offset + pulse_index * pulse_period.

    Parameters
    ----------
    da:
        Raw detector data loaded from a NeXus file, e.g., NXdetector containing
        NXevent_data.
    lookup:
        Lookup table giving time-of-flight as a function of distance and time of
        arrival.
    ltotal:
        Total length of the flight path from the source to the detector.
    pulse_stride_offset:
        When pulse-skipping, the offset of the first pulse in the stride. This is
        typically zero but can be a small integer < pulse_stride.
    """
    if detector_data.bins is None:
        raise NotImplementedError(
            "Computing time-of-arrival in histogram mode is not implemented yet."
        )
    inputs = _prepare_tof_interpolation_inputs(
        da=detector_data,
        lookup=lookup,
        ltotal=ltotal,
        pulse_stride_offset=pulse_stride_offset,
    )
    parts = detector_data.bins.constituents
    parts["data"] = inputs["eto"]
    # The pulse index is None if pulse_stride == 1 (i.e., no pulse skipping)
    if inputs["pulse_index"] is not None:
        parts["data"] = parts["data"] + inputs["pulse_index"] * inputs["pulse_period"]
    result = detector_data.bins.assign_coords(
        toa=sc.bins(**parts, validate_indices=False)
    )
    return result


def providers() -> tuple[Callable]:
    """
    Providers of the time-of-flight workflow.
    """
    return (
        detector_time_of_flight_data,
        monitor_time_of_flight_data,
        detector_ltotal_from_straight_line_approximation,
        monitor_ltotal_from_straight_line_approximation,
        detector_time_of_arrival_data,
    )
