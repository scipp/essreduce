# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""
Time-of-flight workflow for unwrapping the time of arrival of the neutron at the
detector.
This workflow is used to convert raw detector data with event_time_zero and
event_time_offset coordinates to data with a time-of-flight coordinate.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import scipp as sc
from scipp._scipp.core import _bins_no_validate
from scippneutron._utils import elem_unit

from .to_events import to_events
from .types import (
    DistanceResolution,
    FastestNeutron,
    FrameFoldedTimeOfArrival,
    FramePeriod,
    LookupTableRelativeErrorThreshold,
    Ltotal,
    LtotalRange,
    MaskedTimeOfFlightLookupTable,
    PivotTimeAtDetector,
    PulsePeriod,
    PulseStride,
    PulseStrideOffset,
    RawData,
    ResampledTofData,
    SimulationResults,
    TimeOfArrivalMinusPivotTimeModuloPeriod,
    TimeOfArrivalResolution,
    TimeOfFlightLookupTable,
    TofData,
    UnwrappedTimeOfArrival,
    UnwrappedTimeOfArrivalMinusPivotTime,
)


def frame_period(pulse_period: PulsePeriod, pulse_stride: PulseStride) -> FramePeriod:
    """
    Return the period of a frame, which is defined by the pulse period times the pulse
    stride.

    Parameters
    ----------
    pulse_period:
        Period of the source pulses, i.e., time between consecutive pulse starts.
    pulse_stride:
        Stride of used pulses. Usually 1, but may be a small integer when
        pulse-skipping.
    """
    return FramePeriod(pulse_period * pulse_stride)


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
    toa_resolution: TimeOfArrivalResolution,
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

    # Compute time-of-flight for all neutrons
    wavs = sc.broadcast(simulation.wavelength.to(unit="m"), sizes=toas.sizes).flatten(
        to="event"
    )
    dist = sc.broadcast(distances + simulation_distance, sizes=toas.sizes).flatten(
        to="event"
    )
    tofs = dist * sc.constants.m_n
    tofs *= wavs
    tofs /= sc.constants.h

    data = sc.DataArray(
        data=sc.broadcast(simulation.weight, sizes=toas.sizes).flatten(to="event"),
        coords={
            "toa": toas.flatten(to="event"),
            "tof": tofs.to(unit=time_unit, copy=False),
            "distance": dist,
        },
    )

    binned = data.bin(distance=dist_edges + simulation_distance, toa=toa_resolution)
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
    mean_tof.coords["toa"] = sc.midpoints(mean_tof.coords["toa"])
    mean_tof.coords["distance"] = sc.midpoints(mean_tof.coords["distance"])

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


def find_fastest_neutron(simulation: SimulationResults) -> FastestNeutron:
    """
    Find the fastest neutron in the simulation results.
    """
    ind = np.argmax(simulation.speed.values)
    return FastestNeutron(
        time_of_arrival=simulation.time_of_arrival[ind],
        speed=simulation.speed[ind],
        distance=simulation.distance,
    )


def pivot_time_at_detector(
    fastest_neutron: FastestNeutron, ltotal: Ltotal
) -> PivotTimeAtDetector:
    """
    Compute the pivot time at the detector, i.e., the time of the start of the frame at
    the detector.
    The assumption here is that the fastest neutron in the simulation results is the one
    that arrives at the detector first.
    One could have an edge case where a slightly slower neutron which is born earlier
    could arrive at the detector first, but this edge case is most probably uncommon,
    and the difference in arrival times is likely to be small.

    Parameters
    ----------
    fastest_neutron:
        Properties of the fastest neutron in the simulation results.
    ltotal:
        Total length of the flight path from the source to the detector.
    """
    dist = ltotal - fastest_neutron.distance.to(unit=ltotal.unit)
    toa = fastest_neutron.time_of_arrival + (dist / fastest_neutron.speed).to(
        unit=fastest_neutron.time_of_arrival.unit, copy=False
    )
    return PivotTimeAtDetector(toa)


def unwrapped_time_of_arrival(
    da: RawData, offset: PulseStrideOffset, pulse_period: PulsePeriod
) -> UnwrappedTimeOfArrival:
    """
    Compute the unwrapped time of arrival of the neutron at the detector.
    For event data, this is essentially ``event_time_offset + event_time_zero``.

    Parameters
    ----------
    da:
        Raw detector data loaded from a NeXus file, e.g., NXdetector containing
        NXevent_data.
    offset:
        Integer offset of the first pulse in the stride (typically zero unless we are
        using pulse-skipping and the events do not begin with the first pulse in the
        stride).
    pulse_period:
        Period of the source pulses, i.e., time between consecutive pulse starts.
    """
    if da.bins is None:
        # 'time_of_flight' is the canonical name in NXmonitor, but in some files, it
        # may be called 'tof'.
        key = next(iter(set(da.coords.keys()) & {"time_of_flight", "tof"}))
        toa = da.coords[key]
    else:
        # To unwrap the time of arrival, we want to add the event_time_zero to the
        # event_time_offset. However, we do not really care about the exact datetimes,
        # we just want to know the offsets with respect to the start of the run.
        # Hence we use the smallest event_time_zero as the time origin.
        time_zero = da.coords["event_time_zero"] - da.coords["event_time_zero"].min()
        coord = da.bins.coords["event_time_offset"]
        unit = elem_unit(coord)
        toa = (
            coord
            + time_zero.to(dtype=float, unit=unit, copy=False)
            - (offset * pulse_period).to(unit=unit, copy=False)
        )
    return UnwrappedTimeOfArrival(toa)


def unwrapped_time_of_arrival_minus_frame_pivot_time(
    toa: UnwrappedTimeOfArrival, pivot_time: PivotTimeAtDetector
) -> UnwrappedTimeOfArrivalMinusPivotTime:
    """
    Compute the time of arrival of the neutron at the detector, unwrapped at the pulse
    period, minus the start time of the frame.
    We subtract the start time of the frame so that we can use a modulo operation to
    wrap the time of arrival at the frame period in the case of pulse-skipping.

    Parameters
    ----------
    toa:
        Time of arrival of the neutron at the detector, unwrapped at the pulse period.
    pivot_time:
        Pivot time at the detector, i.e., the time of the start of the frame at the
        detector.
    """
    # Order of operation to preserve dimension order
    return UnwrappedTimeOfArrivalMinusPivotTime(
        -pivot_time.to(unit=elem_unit(toa), copy=False) + toa
    )


def time_of_arrival_minus_pivot_time_modulo_period(
    toa_minus_pivot_time: UnwrappedTimeOfArrivalMinusPivotTime,
    frame_period: FramePeriod,
) -> TimeOfArrivalMinusPivotTimeModuloPeriod:
    """
    Compute the time of arrival of the neutron at the detector, unwrapped at the pulse
    period, minus the start time of the frame, modulo the frame period.

    Parameters
    ----------
    toa_minus_pivot_time:
        Time of arrival of the neutron at the detector, unwrapped at the pulse period,
        minus the start time of the frame.
    frame_period:
        Period of the frame, i.e., time between the start of two consecutive frames.
    """
    return TimeOfArrivalMinusPivotTimeModuloPeriod(
        toa_minus_pivot_time
        % frame_period.to(unit=elem_unit(toa_minus_pivot_time), copy=False)
    )


def time_of_arrival_folded_by_frame(
    toa: TimeOfArrivalMinusPivotTimeModuloPeriod,
    pivot_time: PivotTimeAtDetector,
) -> FrameFoldedTimeOfArrival:
    """
    The time of arrival of the neutron at the detector, folded by the frame period.

    Parameters
    ----------
    toa:
        Time of arrival of the neutron at the detector, unwrapped at the pulse period,
        minus the start time of the frame, modulo the frame period.
    pivot_time:
        Pivot time at the detector, i.e., the time of the start of the frame at the
        detector.
    """
    return FrameFoldedTimeOfArrival(
        toa + pivot_time.to(unit=elem_unit(toa), copy=False)
    )


def time_of_flight_data(
    da: RawData,
    lookup: MaskedTimeOfFlightLookupTable,
    ltotal: Ltotal,
    toas: FrameFoldedTimeOfArrival,
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

    # TODO: to make use of multi-threading, we could write our own interpolator.
    # This should be simple enough as we are making the bins linspace, so computing
    # bin indices is fast.
    f = RegularGridInterpolator(
        (
            lookup.coords["toa"].to(unit=elem_unit(toas), copy=False).values,
            lookup.coords["distance"].to(unit=ltotal.unit, copy=False).values,
        ),
        lookup.data.to(unit=elem_unit(toas), copy=False).values.T,
        method="linear",
        bounds_error=False,
    )

    if da.bins is not None:
        ltotal = sc.bins_like(toas, ltotal).bins.constituents["data"]
        toas = toas.bins.constituents["data"]

    tofs = sc.array(
        dims=toas.dims, values=f((toas.values, ltotal.values)), unit=elem_unit(toas)
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
    events = to_events(da.rename_dims(time_of_flight="tof"), "event")

    # Define a new bin width, close to the original bin width.
    # TODO: this could be a workflow parameter
    coord = da.coords["tof"]
    bin_width = (coord["time_of_flight", 1:] - coord["time_of_flight", :-1]).nanmedian()
    rehist = events.hist(tof=bin_width)
    for key, var in da.coords.items():
        if "time_of_flight" not in var.dims:
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
        TimeOfArrivalResolution: 500,
        LookupTableRelativeErrorThreshold: 0.1,
    }


def providers() -> tuple[Callable]:
    """
    Providers of the time-of-flight workflow.
    """
    return (
        compute_tof_lookup_table,
        extract_ltotal,
        find_fastest_neutron,
        frame_period,
        masked_tof_lookup_table,
        pivot_time_at_detector,
        time_of_arrival_folded_by_frame,
        time_of_arrival_minus_pivot_time_modulo_period,
        time_of_flight_data,
        unwrapped_time_of_arrival,
        unwrapped_time_of_arrival_minus_frame_pivot_time,
    )


class TofWorkflow:
    """
    Helper class to build a time-of-flight workflow and cache the expensive part of
    the computation: running the simulation and building the lookup table.

    Parameters
    ----------
    simulated_neutrons:
        Results of a time-of-flight simulation used to create a lookup table.
        The results should be a flat table with columns for time-of-arrival, speed,
        wavelength, and weight.
    ltotal_range:
        Range of total flight path lengths from the source to the detector.
        This is used to create the lookup table to compute the neutron time-of-flight.
        Note that the resulting table will extend slightly beyond this range, as the
        supplied range is not necessarily a multiple of the distance resolution.
    pulse_stride:
        Stride of used pulses. Usually 1, but may be a small integer when
        pulse-skipping.
    pulse_stride_offset:
        Integer offset of the first pulse in the stride (typically zero unless we are
        using pulse-skipping and the events do not begin with the first pulse in the
        stride).
    distance_resolution:
        Resolution of the distance axis in the lookup table.
        Should be a single scalar value with a unit of length.
        This is typically of the order of 1-10 cm.
    toa_resolution:
        Resolution of the time of arrival axis in the lookup table.
        Can be an integer (number of bins) or a sc.Variable (bin width).
    error_threshold:
        Threshold for the variance of the projected time-of-flight above which regions
        are masked.
    """

    def __init__(
        self,
        simulated_neutrons: SimulationResults,
        ltotal_range: LtotalRange,
        pulse_stride: PulseStride | None = None,
        pulse_stride_offset: PulseStrideOffset | None = None,
        distance_resolution: DistanceResolution | None = None,
        toa_resolution: TimeOfArrivalResolution | None = None,
        error_threshold: LookupTableRelativeErrorThreshold | None = None,
    ):
        import sciline as sl

        self.pipeline = sl.Pipeline(providers())
        self.pipeline[SimulationResults] = simulated_neutrons
        self.pipeline[LtotalRange] = ltotal_range

        params = default_parameters()
        self.pipeline[PulsePeriod] = params[PulsePeriod]
        self.pipeline[PulseStride] = pulse_stride or params[PulseStride]
        self.pipeline[PulseStrideOffset] = (
            pulse_stride_offset or params[PulseStrideOffset]
        )
        self.pipeline[DistanceResolution] = (
            distance_resolution or params[DistanceResolution]
        )
        self.pipeline[TimeOfArrivalResolution] = (
            toa_resolution or params[TimeOfArrivalResolution]
        )
        self.pipeline[LookupTableRelativeErrorThreshold] = (
            error_threshold or params[LookupTableRelativeErrorThreshold]
        )

    def __getitem__(self, key):
        return self.pipeline[key]

    def __setitem__(self, key, value):
        self.pipeline[key] = value

    def persist(self) -> None:
        for t in (SimulationResults, MaskedTimeOfFlightLookupTable, FastestNeutron):
            self.pipeline[t] = self.pipeline.compute(t)

    def compute(self, *args, **kwargs) -> Any:
        return self.pipeline.compute(*args, **kwargs)

    def visualize(self, *args, **kwargs) -> Any:
        return self.pipeline.visualize(*args, **kwargs)

    def copy(self) -> TofWorkflow:
        out = self.__class__(None, None)
        out.pipeline = self.pipeline.copy()
        return out

    def insert(self, *args, **kwargs) -> None:
        self.pipeline.insert(*args, **kwargs)
