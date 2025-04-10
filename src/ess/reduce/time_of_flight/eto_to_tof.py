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
from scipp._scipp.core import _bins_no_validate
from scippneutron._utils import elem_unit

try:
    from .interpolator_numba import Interpolator as InterpolatorImpl
except ImportError:
    from .interpolator_scipy import Interpolator as InterpolatorImpl
from .to_events import to_events
from .types import (
    DistanceResolution,
    LookupTableRelativeErrorThreshold,
    Ltotal,
    LtotalRange,
    PulsePeriod,
    PulseStride,
    PulseStrideOffset,
    RawData,
    ResampledTofData,
    SimulationResults,
    TimeOfFlightLookupTable,
    TimeResolution,
    TofData,
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
    time_bins_width = time_bins[1] - time_bins[0]
    # Add an additional bin edge at the end to cover half a bin width above the
    # frame_period.
    time_bins = sc.concat(
        [time_bins, time_bins[-1] + time_bins_width], dim=time_bins.dim
    )
    time_bins -= 0.5 * time_bins_width

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
            )
        )

    table = sc.concat(pieces, 'distance')
    table.coords["distance"] = sc.midpoints(table.coords["distance"])
    table.coords["event_time_offset"] = sc.midpoints(table.coords["event_time_offset"])

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
        self, ltotal: sc.Variable, event_time_offset: sc.Variable
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
            values=self._interpolator(times=event_time_offset, distances=ltotal),
            unit=self._time_unit,
        )


def _time_of_flight_data_histogram(
    da: sc.DataArray,
    lookup: sc.DataArray,
    ltotal: sc.Variable,
    pulse_period: sc.Variable,
) -> sc.DataArray:
    # In NeXus, 'time_of_flight' is the canonical name in NXmonitor, but in some files,
    # it may be called 'tof'.
    key = next(iter(set(da.coords.keys()) & {"time_of_flight", "tof"}))
    raw_eto = da.coords[key].to(dtype=float, copy=False)
    eto_unit = raw_eto.unit
    pulse_period = pulse_period.to(unit=eto_unit)

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

    # In histogram mode, the lookup table cannot have a pulse dimension because we
    # cannot know in the histogrammed data which pulse the events belong to.
    # So we merge the pulse dimension in the lookup table. A quick way to do this
    # is to take the mean of the data along the pulse dimension (there should
    # only be regions that are NaN in one pulse and finite in the other).
    merged = lookup.data.nanmean('pulse')
    dim = merged.dims[0]
    lookup = sc.DataArray(
        data=merged.fold(dim=dim, sizes={'pulse': 1, dim: merged.sizes[dim]}),
        coords={
            'pulse': sc.arange('pulse', 1.0),
            'distance': lookup.coords['distance'],
            'event_time_offset': lookup.coords['event_time_offset'],
        },
    )
    pulse_index = sc.zeros(sizes=etos.sizes)

    # Create linear interpolator
    interp = TofInterpolator(lookup, distance_unit=ltotal.unit, time_unit=eto_unit)

    # Compute time-of-flight of the bin edges using the interpolator
    tofs = interp(
        pulse_index=pulse_index,
        ltotal=ltotal.broadcast(sizes=etos.sizes),
        event_time_offset=etos,
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
    pulse_period = pulse_period.to(unit=etos.unit)
    for i in range(pulse_stride):
        pulse_inds = (pulse_index + i) % pulse_stride
        tofs[i] = interp(
            ltotal=ltotal, event_time_offset=etos + pulse_inds * pulse_period
        )
    # Find the entry in the list with the least number of nan values
    return sorted(tofs, key=lambda x: sc.isnan(tofs[x]).sum())[0]


def _time_of_flight_data_events(
    da: sc.DataArray,
    lookup: sc.DataArray,
    ltotal: sc.Variable,
    pulse_period: sc.Variable,
    pulse_stride: int,
    pulse_stride_offset: int,
) -> sc.DataArray:
    etos = da.bins.coords["event_time_offset"].to(dtype=float, copy=False)
    eto_unit = elem_unit(etos)

    # Create linear interpolator
    interp = TofInterpolator(lookup, distance_unit=ltotal.unit, time_unit=eto_unit)

    # Operate on events (broadcast distances to all events)
    ltotal = sc.bins_like(etos, ltotal).bins.constituents["data"]
    etos = etos.bins.constituents["data"]

    # Compute a pulse index for every event: it is the index of the pulse within a
    # frame period. When there is no pulse skipping, those are all zero. When there is
    # pulse skipping, the index ranges from zero to pulse_stride - 1.
    if pulse_stride == 1:
        pulse_index = sc.zeros(sizes=etos.sizes)
    else:
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
        event_time_offset=etos + pulse_index * pulse_period.to(unit=eto_unit),
    )

    parts = da.bins.constituents
    parts["data"] = tofs
    return da.bins.assign_coords(tof=_bins_no_validate(**parts))


def time_of_flight_data(
    da: RawData,
    lookup: TimeOfFlightLookupTable,
    ltotal: Ltotal,
    pulse_period: PulsePeriod,
    pulse_stride: PulseStride,
    pulse_stride_offset: PulseStrideOffset,
) -> TofData:
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
    pulse_period:
        Period of the source pulses, i.e., time between consecutive pulse starts.
    pulse_stride:
        Stride of used pulses. Usually 1, but may be a small integer when
        pulse-skipping.
    pulse_stride_offset:
        When pulse-skipping, the offset of the first pulse in the stride. This is
        typically zero but can be a small integer < pulse_stride.
    """

    if da.bins is None:
        out = _time_of_flight_data_histogram(
            da=da, lookup=lookup, ltotal=ltotal, pulse_period=pulse_period
        )
    else:
        out = _time_of_flight_data_events(
            da=da,
            lookup=lookup,
            ltotal=ltotal,
            pulse_period=pulse_period,
            pulse_stride=pulse_stride,
            pulse_stride_offset=pulse_stride_offset,
        )
    return TofData(out)


def resample_tof_data(da: TofData) -> ResampledTofData:
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
    return ResampledTofData(
        rehist.assign_coords(
            {key: var for key, var in da.coords.items() if dim not in var.dims}
        )
    )


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
    return (compute_tof_lookup_table, time_of_flight_data)
