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
from scipp._scipp.core import _bins_no_validate
from scippneutron._utils import elem_unit

try:
    from .interpolator_numba import Interpolator as InterpolatorImpl
except ImportError:
    from .interpolator_scipy import Interpolator as InterpolatorImpl

from ..nexus.types import (
    CalibratedBeamline,
    CalibratedMonitor,
    DetectorData,
    MonitorData,
    MonitorType,
    RunType,
)
from .to_events import to_events
from .types import (
    DetectorLtotal,
    DetectorTofData,
    DistanceResolution,
    LookupTableRelativeErrorThreshold,
    LtotalRange,
    MonitorLtotal,
    MonitorTofData,
    PulsePeriod,
    PulseStride,
    PulseStrideOffset,
    ResampledDetectorTofData,
    ResampledMonitorTofData,
    SimulationResults,
    TimeOfFlightLookupTable,
    TimeResolution,
)


def _mask_large_uncertainty(table: sc.DataArray, error_threshold: float):
    """
    Mask regions with large uncertainty with NaNs.
    The values are modified in place in the input table.

    Parameters
    ----------
    table:
        Lookup table with time-of-flight as a function of distance and time-of-arrival.
    error_threshold:
        Threshold for the relative standard deviation (coefficient of variation) of the
        projected time-of-flight above which values are masked.
    """
    # Finally, mask regions with large uncertainty with NaNs.
    relative_error = sc.stddevs(table.data) / sc.values(table.data)
    mask = relative_error > sc.scalar(error_threshold)
    # Use numpy for indexing as table is 2D
    table.values[mask.values] = np.nan


def _compute_mean_tof_in_distance_range(
    simulation: SimulationResults,
    distance_bins: sc.Variable,
    time_bins: sc.Variable,
    distance_unit: str,
    time_unit: str,
    frame_period: sc.Variable,
    time_bins_half_width: sc.Variable,
) -> sc.DataArray:
    """
    Compute the mean time-of-flight inside event_time_offset bins for a given range of
    distances.

    Parameters
    ----------
    simulation:
        Results of a time-of-flight simulation used to create a lookup table.
    distance_bins:
        Bin edges for the distance axis in the lookup table.
    time_bins:
        Bin edges for the event_time_offset axis in the lookup table.
    distance_unit:
        Unit of the distance axis.
    time_unit:
        Unit of the event_time_offset axis.
    frame_period:
        Period of the source pulses, i.e., time between consecutive pulse starts.
    time_bins_half_width:
        Half width of the time bins in the event_time_offset axis.
    """
    simulation_distance = simulation.distance.to(unit=distance_unit)
    distances = sc.midpoints(distance_bins)
    # Compute arrival and flight times for all neutrons
    toas = simulation.time_of_arrival + (distances / simulation.speed).to(
        unit=time_unit, copy=False
    )
    dist = distances + simulation_distance
    tofs = dist * (sc.constants.m_n / sc.constants.h) * simulation.wavelength

    data = sc.DataArray(
        data=sc.broadcast(simulation.weight, sizes=toas.sizes),
        coords={
            "toa": toas,
            "tof": tofs.to(unit=time_unit, copy=False),
            "distance": dist,
        },
    ).flatten(to="event")

    # Add the event_time_offset coordinate, wrapped to the frame_period
    data.coords['event_time_offset'] = data.coords['toa'] % frame_period

    # Because we staggered the mesh by half a bin width, we want the values above
    # the last bin edge to wrap around to the first bin.
    # Technically, those values should end up between -0.5*bin_width and 0, but
    # a simple modulo also works here because even if they end up between 0 and
    # 0.5*bin_width, we are (below) computing the mean between -0.5*bin_width and
    # 0.5*bin_width and it yields the same result.
    # data.coords['event_time_offset'] %= pulse_period - time_bins_half_width
    data.coords['event_time_offset'] %= frame_period - time_bins_half_width

    binned = data.bin(
        distance=distance_bins + simulation_distance, event_time_offset=time_bins
    )

    # Weighted mean of tof inside each bin
    mean_tof = (
        binned.bins.data * binned.bins.coords["tof"]
    ).bins.sum() / binned.bins.sum()
    # Compute the variance of the tofs to track regions with large uncertainty
    variance = (
        binned.bins.data * (binned.bins.coords["tof"] - mean_tof) ** 2
    ).bins.sum() / binned.bins.sum()

    mean_tof.variances = variance.values
    return mean_tof


def compute_tof_lookup_table(
    simulation: SimulationResults,
    ltotal_range: LtotalRange,
    distance_resolution: DistanceResolution,
    time_resolution: TimeResolution,
    pulse_period: PulsePeriod,
    pulse_stride: PulseStride,
    error_threshold: LookupTableRelativeErrorThreshold,
) -> TimeOfFlightLookupTable:
    """
    Compute a lookup table for time-of-flight as a function of distance and
    time-of-arrival.

    Parameters
    ----------
    simulation:
        Results of a time-of-flight simulation used to create a lookup table.
        The results should be a flat table with columns for time-of-arrival, speed,
        wavelength, and weight.
    ltotal_range:
        Range of total flight path lengths from the source to the detector.
    distance_resolution:
        Resolution of the distance axis in the lookup table.
    time_resolution:
        Resolution of the time-of-arrival axis in the lookup table. Must be an integer.
    pulse_period:
        Period of the source pulses, i.e., time between consecutive pulse starts.
    pulse_stride:
        Stride of used pulses. Usually 1, but may be a small integer when
        pulse-skipping.
    error_threshold:
        Threshold for the relative standard deviation (coefficient of variation) of the
        projected time-of-flight above which values are masked.

    Notes
    -----

    Below are some details about the binning and wrapping around frame period in the
    time dimension.

    We have some simulated ``toa`` (events) from a Tof/McStas simulation.
    Those are absolute ``toa``, unwrapped.
    First we compute the usual ``event_time_offset = toa % frame_period``.

    Now, we want to ensure periodic boundaries. If we make a bin centered around 0,
    and a bin centered around 71ms: the first bin will use events between 0 and
    ``0.5 * dt`` (where ``dt`` is the bin width).
    The last bin will use events between ``frame_period - 0.5*dt`` and
    ``frame_period + 0.5 * dt``. So when we compute the mean inside those two bins,
    they will not yield the same results.
    It is as if the first bin is missing the events it should have between
    ``-0.5 * dt`` and 0 (because of the modulo we computed above).

    To fix this, we do not make a last bin around 71ms (the bins stop at
    ``frame_period - 0.5*dt``). Instead, we compute modulo a second time,
    but this time using ``event_time_offset %= (frame_period - 0.5*dt)``.
    (we cannot directly do ``event_time_offset = toa % (frame_period - 0.5*dt)`` in a
    single step because it would introduce a gradual shift,
    as the pulse number increases).

    This second modulo effectively takes all the events that would have gone in the
    last bin (between ``frame_period - 0.5*dt`` and ``frame_period``) and puts them in
    the first bin. Instead of placing them between ``-0.5*dt`` and 0,
    it places them between 0 and ``0.5*dt``, but this does not really matter,
    because we then take the mean inside the first bin.
    Whether the events are on the left or right side of zero does not matter.

    Finally, we make a copy of the left edge, and append it to the right of the table,
    thus ensuring that the values on the right edge are strictly the same as on the
    left edge.
    """
    distance_unit = "m"
    time_unit = simulation.time_of_arrival.unit
    res = distance_resolution.to(unit=distance_unit)
    pulse_period = pulse_period.to(unit=time_unit)
    frame_period = pulse_period * pulse_stride

    min_dist, max_dist = (
        x.to(unit=distance_unit) - simulation.distance.to(unit=distance_unit)
        for x in ltotal_range
    )
    # We need to bin the data below, to compute the weighted mean of the wavelength.
    # This results in data with bin edges.
    # However, the 2d interpolator expects bin centers.
    # We want to give the 2d interpolator a table that covers the requested range,
    # hence we need to extend the range by at least half a resolution in each direction.
    # Then, we make the choice that the resolution in distance is the quantity that
    # should be preserved. Because the difference between min and max distance is
    # not necessarily an integer multiple of the resolution, we need to add a pad to
    # ensure that the last bin is not cut off. We want the upper edge to be higher than
    # the maximum distance, hence we pad with an additional 1.5 x resolution.
    pad = 2.0 * res
    distance_bins = sc.arange('distance', min_dist - pad, max_dist + pad, res)

    # Create some time bins for event_time_offset.
    # We want our final table to strictly cover the range [0, frame_period].
    # However, binning the data associates mean values inside the bins to the bin
    # centers. Instead, we stagger the mesh by half a bin width so we are computing
    # values for the final mesh edges (the bilinear interpolation needs values on the
    # edges/corners).
    nbins = int(frame_period / time_resolution.to(unit=time_unit)) + 1
    time_bins = sc.linspace(
        'event_time_offset', 0.0, frame_period.value, nbins + 1, unit=pulse_period.unit
    )
    time_bins_half_width = 0.5 * (time_bins[1] - time_bins[0])
    time_bins -= time_bins_half_width

    # To avoid a too large RAM usage, we compute the table in chunks, and piece them
    # together at the end.
    ndist = len(distance_bins) - 1
    max_size = 2e7
    total_size = ndist * len(simulation.time_of_arrival)
    nchunks = total_size / max_size
    chunk_size = int(ndist / nchunks) + 1
    pieces = []
    for i in range(int(nchunks) + 1):
        dist_edges = distance_bins[i * chunk_size : (i + 1) * chunk_size + 1]

        pieces.append(
            _compute_mean_tof_in_distance_range(
                simulation=simulation,
                distance_bins=dist_edges,
                time_bins=time_bins,
                distance_unit=distance_unit,
                time_unit=time_unit,
                frame_period=frame_period,
                time_bins_half_width=time_bins_half_width,
            )
        )

    table = sc.concat(pieces, 'distance')
    table.coords["distance"] = sc.midpoints(table.coords["distance"])
    table.coords["event_time_offset"] = sc.midpoints(table.coords["event_time_offset"])

    # Copy the left edge to the right to create periodic boundary conditions
    table = sc.DataArray(
        data=sc.concat(
            [table.data, table.data['event_time_offset', 0]], dim='event_time_offset'
        ),
        coords={
            "distance": table.coords["distance"],
            "event_time_offset": sc.concat(
                [table.coords["event_time_offset"], frame_period],
                dim='event_time_offset',
            ),
            "pulse_period": pulse_period,
            "pulse_stride": sc.scalar(pulse_stride, unit=None),
            "distance_resolution": table.coords["distance"][1]
            - table.coords["distance"][0],
            "time_resolution": table.coords["event_time_offset"][1]
            - table.coords["event_time_offset"][0],
            "error_threshold": sc.scalar(error_threshold),
        },
    )

    # In-place masking for better performance
    _mask_large_uncertainty(table, error_threshold)

    return TimeOfFlightLookupTable(table)


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
    # it may be called 'tof'.
    key = next(iter(set(da.coords.keys()) & {"time_of_flight", "tof"}))
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

    return rebinned.assign_coords(tof=tofs)


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


def _time_of_flight_data_events(
    da: sc.DataArray,
    lookup: sc.DataArray,
    ltotal: sc.Variable,
    pulse_stride_offset: int,
) -> sc.DataArray:
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

    # Compute time-of-flight for all neutrons using the interpolator
    tofs = interp(
        ltotal=ltotal,
        event_time_offset=etos,
        pulse_index=pulse_index,
        pulse_period=pulse_period,
    )

    parts = da.bins.constituents
    parts["data"] = tofs
    return da.bins.assign_coords(tof=_bins_no_validate(**parts))


def detector_ltotal_from_straight_line_approximation(
    detector_beamline: CalibratedBeamline[RunType],
) -> DetectorLtotal[RunType]:
    """
    Compute Ltotal for the detector pixels.
    This is a naive straight-line approximation to Ltotal based on basic component
    positions.

    Parameters
    ----------
    detector_beamline:
        Beamline data for the detector that contains the positions necessary to compute
        the straight-line approximation to Ltotal (source, sample, and detector
        positions).
    """
    graph = scn.conversion.graph.beamline.beamline(scatter=True)
    return DetectorLtotal[RunType](
        detector_beamline.transform_coords(
            "Ltotal", graph=graph, keep_intermediate=False
        ).coords["Ltotal"]
    )


def monitor_ltotal_from_straight_line_approximation(
    monitor_beamline: CalibratedMonitor[RunType, MonitorType],
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
        return _time_of_flight_data_histogram(da=da, lookup=lookup, ltotal=ltotal)
    else:
        return _time_of_flight_data_events(
            da=da,
            lookup=lookup,
            ltotal=ltotal,
            pulse_stride_offset=pulse_stride_offset,
        )


def detector_time_of_flight_data(
    detector_data: DetectorData[RunType],
    lookup: TimeOfFlightLookupTable,
    ltotal: DetectorLtotal[RunType],
    pulse_stride_offset: PulseStrideOffset,
) -> DetectorTofData[RunType]:
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
    return DetectorTofData[RunType](
        _compute_tof_data(
            da=detector_data,
            lookup=lookup,
            ltotal=ltotal,
            pulse_stride_offset=pulse_stride_offset,
        )
    )


def monitor_time_of_flight_data(
    monitor_data: MonitorData[RunType, MonitorType],
    lookup: TimeOfFlightLookupTable,
    ltotal: MonitorLtotal[RunType, MonitorType],
    pulse_stride_offset: PulseStrideOffset,
) -> MonitorTofData[RunType, MonitorType]:
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
    return MonitorTofData[RunType, MonitorType](
        _compute_tof_data(
            da=monitor_data,
            lookup=lookup,
            ltotal=ltotal,
            pulse_stride_offset=pulse_stride_offset,
        )
    )


def _resample_tof_data(da: sc.DataArray) -> sc.DataArray:
    """
    Histogrammed data that has been converted to `tof` will typically have
    unsorted bin edges (due to either wrapping of `time_of_flight` or wavelength
    overlap between subframes).
    This function re-histograms the data to ensure that the bin edges are sorted.
    It makes use of the ``to_events`` helper which generates a number of events in each
    bin with a uniform distribution. The new events are then histogrammed using a set of
    sorted bin edges.

    WARNING:
    This function is highly experimental, has limitations and should be used with
    caution. It is a workaround to the issue that rebinning data with unsorted bin
    edges is not supported in scipp.
    As such, this function is not part of the default set of providers, and needs to be
    inserted manually into the workflow.

    Parameters
    ----------
    da:
        Histogrammed data with the time-of-flight coordinate.
    """
    dim = next(iter(set(da.dims) & {"time_of_flight", "tof"}))
    data = da.rename_dims({dim: "tof"}).drop_coords(
        [name for name in da.coords if name != "tof"]
    )
    events = to_events(data, "event")

    # Define a new bin width, close to the original bin width.
    # TODO: this could be a workflow parameter
    coord = da.coords["tof"]
    bin_width = (coord[dim, 1:] - coord[dim, :-1]).nanmedian()
    rehist = events.hist(tof=bin_width)
    return rehist.assign_coords(
        {key: var for key, var in da.coords.items() if dim not in var.dims}
    )


def resample_detector_time_of_flight_data(
    da: DetectorTofData[RunType],
) -> ResampledDetectorTofData[RunType]:
    """
    Resample the detector time-of-flight data to ensure that the bin edges are sorted.
    """
    return ResampledDetectorTofData(_resample_tof_data(da))


def resample_monitor_time_of_flight_data(
    da: MonitorTofData[RunType, MonitorType],
) -> ResampledMonitorTofData[RunType, MonitorType]:
    """
    Resample the monitor time-of-flight data to ensure that the bin edges are sorted.
    """
    return ResampledMonitorTofData(_resample_tof_data(da))


def default_parameters() -> dict:
    """
    Default parameters of the time-of-flight workflow.
    """
    return {
        PulsePeriod: 1.0 / sc.scalar(14.0, unit="Hz"),
        PulseStride: 1,
        PulseStrideOffset: None,
        DistanceResolution: sc.scalar(0.1, unit="m"),
        TimeResolution: sc.scalar(250.0, unit='us'),
        LookupTableRelativeErrorThreshold: 0.1,
    }


def providers() -> tuple[Callable]:
    """
    Providers of the time-of-flight workflow.
    """
    return (
        compute_tof_lookup_table,
        detector_time_of_flight_data,
        monitor_time_of_flight_data,
        detector_ltotal_from_straight_line_approximation,
        monitor_ltotal_from_straight_line_approximation,
    )
