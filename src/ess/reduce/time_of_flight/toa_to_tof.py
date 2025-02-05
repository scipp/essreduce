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


def extract_ltotal(da: RawData) -> Ltotal:
    """
    Extract the total length of the flight path from the source to the detector from the
    detector data.

    Parameters
    ----------
    da:
        Raw detector data loaded from a NeXus file, e.g., NXdetector containing
        NXevent_data.
    """
    return Ltotal(da.coords["Ltotal"])


def compute_tof_lookup_table(
    simulation: SimulationResults,
    ltotal_range: LtotalRange,
    distance_resolution: DistanceResolution,
    time_resolution: TimeResolution,
    pulse_period: PulsePeriod,
    pulse_stride: PulseStride,
    pulse_stride_offset: PulseStrideOffset,
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
    pulse_stride_offset:
        When pulse-skipping, the offset of the first pulse in the stride.
    error_threshold:
        Threshold for the relative standard deviation (coefficient of variation) of the
        projected time-of-flight above which values are masked.
    """
    distance_unit = "m"
    time_unit = simulation.time_of_arrival.unit
    res = distance_resolution.to(unit=distance_unit)
    simulation_distance = simulation.distance.to(unit=distance_unit)
    pulse_period = pulse_period.to(unit=time_unit)
    frame_period = pulse_period * pulse_stride

    min_dist, max_dist = (
        x.to(unit=distance_unit) - simulation_distance for x in ltotal_range
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
    dist_edges = sc.array(
        dims=["distance"],
        values=np.arange((min_dist - pad).value, (max_dist + pad).value, res.value),
        unit=distance_unit,
    )

    # Create some time bins for event_time_offset.
    # We want our final table to strictly cover the range [0, pulse_period].
    # However, binning the data associates mean values inside the bins to the bin
    # centers. Instead, we stagger the mesh by half a bin width so we are computing
    # values for the final mesh edges (the bilinear interpolation needs values on the
    # edges/corners).
    time_bins = sc.linspace(
        'event_time_offset',
        0.0,
        pulse_period.value,
        time_resolution + 1,
        unit=pulse_period.unit,
    )
    time_bins_half_width = 0.5 * (time_bins[1] - time_bins[0])
    time_bins -= time_bins_half_width

    # To avoid a too large RAM usage, we compute the table in chunks, and piece them
    # together at the end.
    ndist = len(dist_edges) - 1
    max_size = 2e7
    total_size = ndist * len(simulation.time_of_arrival)
    nchunks = total_size / max_size
    chunk_size = int(ndist / nchunks) + 1
    pieces = []
    for i in range(int(nchunks) + 1):
        edges = dist_edges[i * chunk_size : (i + 1) * chunk_size + 1]
        distances = sc.midpoints(edges)

        toas = simulation.time_of_arrival + (distances / simulation.speed).to(
            unit=time_unit, copy=False
        )

        # Compute time-of-flight for all neutrons
        wavs = sc.broadcast(
            simulation.wavelength.to(unit="m"), sizes=toas.sizes
        ).flatten(to="event")
        dist = sc.broadcast(distances + simulation_distance, sizes=toas.sizes).flatten(
            to="event"
        )
        tofs = dist * (sc.constants.m_n / sc.constants.h)
        tofs *= wavs

        data = sc.DataArray(
            data=sc.broadcast(simulation.weight, sizes=toas.sizes).flatten(to="event"),
            coords={
                "toa": toas.flatten(to="event"),
                "tof": tofs.to(unit=time_unit, copy=False),
                "distance": dist,
            },
        )

        # Add the event_time_offset and pulse index coordinate to the data.
        data.coords['event_time_offset'] = data.coords['toa'] % pulse_period
        pulse_index = (data.coords['toa'] % frame_period) // pulse_period

        # pulse_index = data.coords['toa'] % frame_period
        # pulse_index %= frame_period - time_bins_half_width
        # pulse_index = pulse_index // pulse_period

        pulse_index += pulse_stride_offset
        pulse_index %= pulse_stride
        data.coords['pulse'] = pulse_index

        # TODO: Use a sc.lookup for the pulse_index instead.
        edges = (
            sc.arange('pulse', 1, 3)
            * sc.concat([pulse_period - time_bins_half_width, pulse_period], 'toa')
        ).flatten(to='toa')
        edges = sc.concat([sc.scalar(0.0, unit=time_unit), edges], 'toa')
        pulse_lookup = sc.DataArray(
            data=sc.array(
                dims=['toa'], values=np.roll(np.repeat(np.arange(pulse_stride), 2), -1)
            ),
            coords={'toa': edges},
        )

        # Because we staggered the mesh by half a bin width, we want the values above
        # the last bin edge to wrap around to the first bin.
        # Technically, those values should end up between -0.5*bin_width and 0, but
        # a simple modulo also works here because even if they end up between 0 and
        # 0.5*bin_width, we are (below) computing the mean between -0.5*bin_width and
        # 0.5*bin_width and it yields the same result.
        data.coords['event_time_offset'] %= pulse_period - time_bins_half_width

        binned = data.group('pulse').bin(
            distance=edges + simulation_distance, event_time_offset=time_bins
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
        mean_tof.coords["distance"] = sc.midpoints(mean_tof.coords["distance"])
        pieces.append(mean_tof)

    table = sc.concat(pieces, 'distance')

    table.coords["event_time_offset"] = sc.midpoints(table.coords["event_time_offset"])
    # We are still missing the upper edge of the table in the event_time_offset axis
    # (at pulse_period). Because the event_time_offset is periodic, we can simply copy
    # the left edge over to the right edge.
    # Note that the
    left_edge = table['event_time_offset', 0].copy()
    left_edge.coords['event_time_offset'] = pulse_period
    out = sc.concat([table, left_edge], dim='event_time_offset')

    # Finally, mask regions with large uncertainty with NaNs.
    relative_error = sc.stddevs(out.data) / sc.values(out.data)
    mask = relative_error > sc.scalar(error_threshold)
    # Use numpy for indexing as table is 2D
    out.values[mask.values] = np.nan

    return TimeOfFlightLookupTable(out)


def _make_tof_interpolator(
    lookup: sc.DataArray, distance_unit: str, time_unit: str
) -> Callable:
    from scipy.interpolate import RegularGridInterpolator

    # TODO: to make use of multi-threading, we could write our own interpolator.
    # This should be simple enough as we are making the bins linspace, so computing
    # bin indices is fast.

    # In the pulse dimension, it could be that for a given event_time_offset and
    # distance, a tof value is finite in one pulse and NaN in the other.
    # When using the bilinear interpolation, even if the value of the requested point is
    # exactly 0 or 1 (in the case of pulse_stride=2), the interpolator will still
    # use all 4 corners surrounding the point. This means that if one of the corners
    # is NaN, the result will be NaN.
    # Here, we use a trick where we duplicate the lookup values in the 'pulse' dimension
    # so that the interpolator has values on bin edges for that dimension.
    # The interpolator raises an error if axes coordinates are not strictly monotonic,
    # so we cannot use e.g. [-0.5, 0.5, 0.5, 1.5] in the case of pulse_stride=2.
    # Instead we use [-0.25, 0.25, 0.75, 1.25].
    base_grid = np.arange(float(lookup.sizes["pulse"]))
    return RegularGridInterpolator(
        (
            np.sort(np.concatenate([base_grid - 0.25, base_grid + 0.25])),
            lookup.coords["distance"].to(unit=distance_unit, copy=False).values,
            lookup.coords["event_time_offset"].to(unit=time_unit, copy=False).values,
        ),
        np.repeat(lookup.data.to(unit=time_unit, copy=False).values, 2, axis=0),
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
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
    eto_unit = da.coords[key].unit
    pulse_period = pulse_period.to(unit=eto_unit)

    # In histogram mode, because there is a wrap around at the end of the pulse, we
    # need to insert a bin edge at that exact location to avoid having the last bin
    # with one finite left edge and a NaN right edge (it becomes NaN as it would be
    # outside the range of the lookup table).
    new_bins = sc.sort(
        sc.concat(
            [da.coords[key], sc.scalar(0.0, unit=eto_unit), pulse_period], dim=key
        ),
        key=key,
    )
    rebinned = da.rebin({key: new_bins})
    etos = rebinned.coords[key]

    # In histogram mode, the lookup table cannot have a pulse dimension because we
    # cannot know in the histogrammed data which pulse the events belong to.
    # So we merge the pulse dimension in the lookup table. A quick way to do this
    # is to take the mean of the data along the pulse dimension (there should
    # mainly be regions that are NaN in one pulse and finite in the other).
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

    # Create 2D interpolator
    interp = _make_tof_interpolator(
        lookup, distance_unit=ltotal.unit, time_unit=eto_unit
    )

    # Compute time-of-flight of the bin edges using the interpolator
    tofs = sc.array(
        dims=etos.dims,
        values=interp((pulse_index.values, ltotal.values, etos.values)),
        unit=eto_unit,
    )

    return rebinned.assign_coords(tof=tofs)


def _time_of_flight_data_events(
    da: sc.DataArray,
    lookup: sc.DataArray,
    ltotal: sc.Variable,
    pulse_period: sc.Variable,
    pulse_stride: int,
) -> sc.DataArray:
    etos = da.bins.coords["event_time_offset"]
    eto_unit = elem_unit(etos)
    pulse_period = pulse_period.to(unit=eto_unit)
    frame_period = pulse_period * pulse_stride

    # Compute a pulse index for every event: it is the index of the pulse within a
    # frame period. When there is no pulse skipping, those are all zero. When there is
    # pulse skipping, the index ranges from zero to pulse_stride - 1.
    etz = da.bins.concat().value.coords['event_time_zero']
    tmin = etz.min()
    pulse_index = (
        ((da.bins.coords['event_time_zero'] - tmin) + 0.5 * pulse_period) % frame_period
    ) // pulse_period

    # Create 2D interpolator
    interp = _make_tof_interpolator(
        lookup, distance_unit=ltotal.unit, time_unit=eto_unit
    )

    # Operate on events (broadcast distances to all events)
    ltotal = sc.bins_like(etos, ltotal).bins.constituents["data"]
    etos = etos.bins.constituents["data"]
    pulse_index = pulse_index.bins.constituents["data"]

    # Compute time-of-flight for all neutrons using the interpolator
    tofs = sc.array(
        dims=etos.dims,
        values=interp((pulse_index.values, ltotal.values, etos.values)),
        unit=eto_unit,
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
    events = to_events(da.rename_dims({dim: "tof"}), "event")

    # Define a new bin width, close to the original bin width.
    # TODO: this could be a workflow parameter
    coord = da.coords["tof"]
    bin_width = (coord[dim, 1:] - coord[dim, :-1]).nanmedian()
    rehist = events.hist(tof=bin_width)
    for key, var in da.coords.items():
        if dim not in var.dims:
            rehist.coords[key] = var
    return ResampledTofData(rehist)


def default_parameters() -> dict:
    """
    Default parameters of the time-of-flight workflow.
    """
    return {
        PulsePeriod: 1.0 / sc.scalar(14.0, unit="Hz"),
        PulseStride: 1,
        PulseStrideOffset: 0,
        DistanceResolution: sc.scalar(0.1, unit="m"),
        TimeResolution: 256,
        LookupTableRelativeErrorThreshold: 0.1,
    }


def providers() -> tuple[Callable]:
    """
    Providers of the time-of-flight workflow.
    """
    return (compute_tof_lookup_table, extract_ltotal, time_of_flight_data)
