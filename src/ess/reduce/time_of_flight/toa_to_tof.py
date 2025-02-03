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
    # FramePeriod,
    LookupTableRelativeErrorThreshold,
    Ltotal,
    LtotalRange,
    MaskedTimeOfFlightLookupTable,
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

# def frame_period(pulse_period: PulsePeriod, pulse_stride: PulseStride) -> FramePeriod:
#     """
#     Return the period of a frame, which is defined by the pulse period times the pulse
#     stride.

#     Parameters
#     ----------
#     pulse_period:
#         Period of the source pulses, i.e., time between consecutive pulse starts.
#     pulse_stride:
#         Stride of used pulses. Usually 1, but may be a small integer when
#         pulse-skipping.
#     """
#     return FramePeriod(pulse_period * pulse_stride)


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
    toa_resolution:
        Resolution of the time-of-arrival axis in the lookup table.
    """
    import time

    start = time.time()
    distance_unit = "m"
    res = distance_resolution.to(unit=distance_unit)
    simulation_distance = simulation.distance.to(unit=distance_unit)

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
    distances = sc.midpoints(dist_edges)

    time_unit = simulation.time_of_arrival.unit
    toas = simulation.time_of_arrival + (distances / simulation.speed).to(
        unit=time_unit, copy=False
    )

    print(f"Time to compute toas: {time.time() - start}")
    start = time.time()

    # Compute time-of-flight for all neutrons
    wavs = sc.broadcast(simulation.wavelength.to(unit="m"), sizes=toas.sizes).flatten(
        to="event"
    )
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

    # Add the event_time_offset coordinate to the data.
    pulse_period = pulse_period.to(unit=time_unit)
    data.coords['event_time_offset'] = data.coords['toa'] % pulse_period
    print(f"Time to compute tofs: {time.time() - start}")
    start = time.time()

    # Create some time bins for event_time_offset.
    # For the bilinear interpolation, we need to have values on the edges of the bins.
    # So we offsets the bins by half a resolution to compute the means inside the bins,
    # and later convert the axes to bin midpoints.
    # The table need to cover exactly the range [0, pulse_period].
    half_width = (pulse_period / time_resolution).value * 0.5
    time_bins = sc.linspace(
        'event_time_offset',
        -half_width,
        pulse_period.value + half_width,
        time_resolution + 2,  # nbins + 1 for bin edges, + 1 for the extra padding
        unit=pulse_period.unit,
    )

    frame_period = (pulse_period * pulse_stride).to(unit=time_unit)
    data.coords['pulse'] = (data.coords['toa'] % frame_period) // pulse_period

    binned = (
        data.group('pulse').bin(
            # toa=sc.arange('toa', pulse_stride + 2) * pulse_period,
            distance=dist_edges + simulation_distance,
            event_time_offset=time_bins,
        )
        # .rename_dims(toa='pulse')
        # .drop_coords('toa')
    )

    print(f"Time to bin data: {time.time() - start}")
    start = time.time()

    # Weighted mean of tof inside each bin
    mean_tof = (
        binned.bins.data * binned.bins.coords["tof"]
    ).bins.sum() / binned.bins.sum()
    # Compute the variance of the tofs to track regions with large uncertainty
    variance = (
        binned.bins.data * (binned.bins.coords["tof"] - mean_tof) ** 2
    ).bins.sum() / binned.bins.sum()

    mean_tof.variances = variance.values

    # Convert coordinates to midpoints
    mean_tof.coords["event_time_offset"] = sc.midpoints(
        mean_tof.coords["event_time_offset"]
    )
    mean_tof.coords["distance"] = sc.midpoints(mean_tof.coords["distance"])
    print(f"Time to compute lookup table: {time.time() - start}")

    return TimeOfFlightLookupTable(mean_tof)


def masked_tof_lookup_table(
    tof_lookup: TimeOfFlightLookupTable,
    error_threshold: LookupTableRelativeErrorThreshold,
) -> MaskedTimeOfFlightLookupTable:
    """
    Mask regions of the lookup table where the variance of the projected time-of-flight
    is larger than a given threshold.

    Parameters
    ----------
    tof_lookup:
        Lookup table giving time-of-flight as a function of distance and
        time-of-arrival.
    variance_threshold:
        Threshold for the variance of the projected time-of-flight above which regions
        are masked.
    """
    relative_error = sc.stddevs(tof_lookup.data) / sc.values(tof_lookup.data)
    mask = relative_error > sc.scalar(error_threshold)
    out = tof_lookup.copy()
    # Use numpy for indexing as table is 2D
    out.values[mask.values] = np.nan
    return MaskedTimeOfFlightLookupTable(out)


def time_of_flight_data(
    da: RawData,
    lookup: MaskedTimeOfFlightLookupTable,
    ltotal: Ltotal,
    pulse_period: PulsePeriod,
    # frame_period: FramePeriod,
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
    toas:
        Time of arrival of the neutron at the detector, folded by the frame period.
    """
    from scipy.interpolate import RegularGridInterpolator

    etos = da.bins.coords["event_time_offset"]
    eto_unit = elem_unit(etos)

    frame_period = (pulse_period * pulse_stride).to(unit=eto_unit)

    # Compute a pulse index for every event: it is the index of the pulse within a
    # frame period. When there is no pulse skipping, those are all zero. When there is
    # pulse skipping, the index ranges from zero to pulse_stride - 1.
    etz = da.bins.concat().value.coords['event_time_zero']
    tmin = etz.min()
    # pulse_period = (1.0 / sc.scalar(14., unit='Hz')).to(unit='us')
    # pulse_stride = 2
    # frame_period = pulse_period * pulse_stride
    pulse_period = pulse_period.to(unit=eto_unit)
    pulse_index = (
        ((da.bins.coords['event_time_zero'] - tmin) + 0.5 * pulse_period) % frame_period
    ) // pulse_period
    # pulse_index

    # TODO: to make use of multi-threading, we could write our own interpolator.
    # This should be simple enough as we are making the bins linspace, so computing
    # bin indices is fast.

    # Here, we use a trick where we duplicate the lookup values in the 'pulse' dimension
    # so that the interpolator has values on bin edges for that dimension.
    # The interpolator raises an error if axes coordinates are not stricly monotonic,
    # so we cannot use e.g. [-0.5, 0.5, 0.5, 1.5] in the case of pulse_stride=2.
    # Instead we use [-0.25, 0.25, 0.75, 1.25].
    base_grid = np.arange(float(pulse_stride))
    f = RegularGridInterpolator(
        (
            np.sort(np.concatenate([base_grid - 0.25, base_grid + 0.25])),
            lookup.coords["distance"].to(unit=ltotal.unit, copy=False).values,
            lookup.coords["event_time_offset"].to(unit=eto_unit, copy=False).values,
        ),
        np.repeat(lookup.data.to(unit=eto_unit, copy=False).values, 2, axis=0),  # .T,
        method="linear",
        bounds_error=False,
    )

    if da.bins is not None:
        ltotal = sc.bins_like(etos, ltotal).bins.constituents["data"]
        etos = etos.bins.constituents["data"]
        pulse_index = pulse_index.bins.constituents["data"]

    tofs = sc.array(
        dims=etos.dims,
        values=f((pulse_index.values, ltotal.values, etos.values)),
        unit=eto_unit,
    )

    if da.bins is not None:
        parts = da.bins.constituents
        parts["data"] = tofs
        out = da.bins.assign_coords(tof=_bins_no_validate(**parts))
    else:
        out = da.assign_coords(tof=tofs)

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
        TimeResolution: 1000,
        LookupTableRelativeErrorThreshold: 0.1,
    }


def providers() -> tuple[Callable]:
    """
    Providers of the time-of-flight workflow.
    """
    return (
        compute_tof_lookup_table,
        extract_ltotal,
        # frame_period,
        masked_tof_lookup_table,
        time_of_flight_data,
    )


# class TofWorkflow:
#     """
#     Helper class to build a time-of-flight workflow and cache the expensive part of the
#     computation: running the simulation and building the lookup table.

#     Parameters
#     ----------
#     simulated_neutrons:
#         Results of a time-of-flight simulation used to create a lookup table.
#         The results should be a flat table with columns for time-of-arrival, speed,
#         wavelength, and weight.
#     ltotal_range:
#         Range of total flight path lengths from the source to the detector.
#         This is used to create the lookup table to compute the neutron
#         time-of-flight.
#         Note that the resulting table will extend slightly beyond this range, as the
#         supplied range is not necessarily a multiple of the distance resolution.
#     pulse_stride:
#         Stride of used pulses. Usually 1, but may be a small integer when
#         pulse-skipping.
#     pulse_stride_offset:
#         Integer offset of the first pulse in the stride (typically zero unless we
#         are using pulse-skipping and the events do not begin with the first pulse in
#         the stride).
#     distance_resolution:
#         Resolution of the distance axis in the lookup table.
#         Should be a single scalar value with a unit of length.
#         This is typically of the order of 1-10 cm.
#     toa_resolution:
#         Resolution of the time of arrival axis in the lookup table.
#         Can be an integer (number of bins) or a sc.Variable (bin width).
#     error_threshold:
#         Threshold for the variance of the projected time-of-flight above which
#         regions are masked.
#     """

#     def __init__(
#         self,
#         simulated_neutrons: SimulationResults,
#         ltotal_range: LtotalRange,
#         pulse_stride: PulseStride | None = None,
#         pulse_stride_offset: PulseStrideOffset | None = None,
#         distance_resolution: DistanceResolution | None = None,
#         toa_resolution: TimeResolution | None = None,
#         error_threshold: LookupTableRelativeErrorThreshold | None = None,
#     ):
#         import sciline as sl

#         self.pipeline = sl.Pipeline(providers())
#         self.pipeline[SimulationResults] = simulated_neutrons
#         self.pipeline[LtotalRange] = ltotal_range

#         params = default_parameters()
#         self.pipeline[PulsePeriod] = params[PulsePeriod]
#         self.pipeline[PulseStride] = pulse_stride or params[PulseStride]
#         self.pipeline[PulseStrideOffset] = (
#             pulse_stride_offset or params[PulseStrideOffset]
#         )
#         self.pipeline[DistanceResolution] = (
#             distance_resolution or params[DistanceResolution]
#         )
#         self.pipeline[TimeResolution] = toa_resolution or params[TimeResolution]
#         self.pipeline[LookupTableRelativeErrorThreshold] = (
#             error_threshold or params[LookupTableRelativeErrorThreshold]
#         )

#     def cache_results(
#         self,
#         results=(SimulationResults, MaskedTimeOfFlightLookupTable, FastestNeutron),
#     ) -> None:
#         """
#         Cache a list of (usually expensive to compute) intermediate results of the
#         time-of-flight workflow.

#         Parameters
#         ----------
#         results:
#             List of results to cache.
#         """
#         for t in results:
#             self.pipeline[t] = self.pipeline.compute(t)
