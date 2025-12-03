# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Normalization routines for neutron data reduction."""

from __future__ import annotations

import enum
import functools
import warnings

import scipp as sc

from .uncertainty import UncertaintyBroadcastMode, broadcast_uncertainties


def normalize_by_monitor_histogram(
    detector: sc.DataArray,
    *,
    monitor: sc.DataArray,
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
    skip_range_check: bool = False,
) -> sc.DataArray:
    """Normalize detector data by a normalized histogrammed monitor.

    This normalization accounts for both the (wavelength) profile of the incident beam
    and the integrated neutron flux, meaning measurement duration and source strength.

    - For *event* detectors, the monitor values are mapped to the detector
      using :func:`scipp.lookup`. That is, for detector event :math:`d_i`,
      :math:`m_i` is the monitor bin value at the same coordinate.
    - For *histogram* detectors, the monitor is generally rebinned using the detector
      binning using :func:`scipp.rebin`. Thus, detector value :math:`d_i` and
      monitor value :math:`m_i` correspond to the same bin.

        - In case the detector coordinate does not have a dimension in common with the
          monitor, :func:`scipp.lookup` is used as in the event case.

    In both cases, let :math:`x_i` be the lower bound of monitor bin :math:`i`
    and let :math:`\\Delta x_i = x_{i+1} - x_i` be the width of that bin.

    The detector is normalized according to

    .. math::

        d_i^\\text{Norm} = \\frac{d_i}{m_i} \\Delta x_i

    Parameters
    ----------
    detector:
        Input detector data.
        Must have a coordinate named ``monitor.dim``, that is, the single
        dimension name of the **monitor**.
    monitor:
        A histogrammed monitor.
        Must be one-dimensional and have a dimension coordinate, typically "wavelength".
    uncertainty_broadcast_mode:
        Choose how uncertainties of the monitor are broadcast to the sample data.
    skip_range_check:
        If false (default), the detector data must be within the range of the monitor
        coordinate. Set this to true to disable the check.
        The value of out-of-range bins / events is undefined in that case.

        This is useful when the detector contains data outside the monitor range, and it
        is difficult or impossible to slice the detector without also removing in-range
        data. In this case, the caller can mask those data points and skip the range
        check. ``normalize_by_monitor_histogram`` does not take masks into account when
        checking ranges as that is expensive to implement in a general case.

    Returns
    -------
    :
        ``detector`` normalized by ``monitor``.
        If the monitor has masks or contains non-finite values, the output has a mask
        called '_monitor_mask' constructed from the monitor masks and non-finite values.

    See also
    --------
    normalize_by_monitor_integrated:
        Normalize by an integrated monitor.
    """
    if not skip_range_check:
        _check_monitor_range_contains_detector(monitor=monitor, detector=detector)

    dim = monitor.dim

    match _HistogramNormalizationMode.deduce(detector, dim):
        case _HistogramNormalizationMode.Events:
            detector = _mask_detector_for_norm(detector=detector, monitor=monitor)
            norm = _histogram_monitor_term(
                monitor,
                dim,
                broadcast_to=detector,
                uncertainty_broadcast_mode=uncertainty_broadcast_mode,
            )
            if dim in detector.bins.coords:
                return detector.bins / sc.lookup(norm, dim=dim)
            else:
                return detector / norm

        case _HistogramNormalizationMode.BinsCommonDim:
            monitor = monitor.rebin({dim: detector.coords[dim]})
            detector = _mask_detector_for_norm(detector=detector, monitor=monitor)
            norm = _histogram_monitor_term(
                monitor,
                dim,
                broadcast_to=detector,
                uncertainty_broadcast_mode=uncertainty_broadcast_mode,
            )
            return detector / norm

        case _HistogramNormalizationMode.BinsDifferentDim:
            detector = _mask_detector_for_norm(detector=detector, monitor=monitor)
            # No broadcast here because there are no common dims, use lookup instead.
            norm = _histogram_monitor_term(
                monitor,
                dim,
                uncertainty_broadcast_mode=uncertainty_broadcast_mode,
            )
            return detector / sc.lookup(norm)[_compute_bin_centers(detector, dim)]


def normalize_by_monitor_integrated(
    detector: sc.DataArray,
    *,
    monitor: sc.DataArray,
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
    skip_range_check: bool = False,
) -> sc.DataArray:
    """Normalize detector data by an integrated monitor.

    This normalization accounts only for the integrated neutron flux,
    meaning measurement duration and source strength.
    It does *not* account for the (wavelength) profile of the incident beam.
    For that, see :func:`normalize_by_monitor_histogram`.

    Let :math:`d_i` be a detector event or the counts in a detector bin.
    The normalized detector is

    .. math::

        d_i^\\text{Norm} = \\frac{d_i}{\\sum_j\\, m_j}

    where :math:`m_j` is the monitor counts in bin :math:`j`.
    Note that this is not a true integral but only a sum over monitor events.

    The result depends on the range of the monitor but not its
    binning within that range.

    Parameters
    ----------
    detector:
        Input detector data.
    monitor:
        A histogrammed monitor.
        Must be one-dimensional and have a dimension coordinate, typically "wavelength".
    uncertainty_broadcast_mode:
        Choose how uncertainties of the monitor are broadcast to the sample data.
    skip_range_check:
        If false (default), the detector data must be within the range of the monitor
        coordinate. Set this to true to disable the check.
        The value of out-of-range bins / events is undefined in that case.

        This is useful when the detector contains data outside the monitor range, and it
        is difficult or impossible to slice the detector without also removing in-range
        data. In this case, the caller can mask those data points and skip the range
        check. ``normalize_by_monitor_histogram`` does not take masks into account when
        checking ranges as that is expensive to implement in a general case.

    Returns
    -------
    :
        `detector` normalized by a monitor.
        If the monitor has masks or contains non-finite values, the output has a mask
        called '_monitor_mask' constructed from the monitor masks and non-finite values.

    See also
    --------
    normalize_by_monitor_histogram:
        Normalize by a monitor histogram.
    """
    if not skip_range_check:
        _check_monitor_range_contains_detector(monitor=monitor, detector=detector)
    detector = _mask_detector_for_norm(detector=detector, monitor=monitor)
    norm = monitor.nansum().data
    norm = broadcast_uncertainties(
        norm, prototype=detector, mode=uncertainty_broadcast_mode
    )
    return detector / norm


def _check_monitor_range_contains_detector(
    *, monitor: sc.DataArray, detector: sc.DataArray
) -> None:
    dim = monitor.dim
    if not monitor.coords.is_edges(dim):
        raise sc.CoordError(
            f"Monitor coordinate '{dim}' must be bin-edges to integrate the monitor."
        )

    # Prefer a bin coord over an event coord because this makes the behavior for binned
    # and histogrammed data consistent. If we used an event coord, we might allow a
    # monitor range that is less than the detector bins which is fine for the events,
    # but would be wrong if the detector was subsequently histogrammed.
    if (det_coord := detector.coords.get(dim)) is not None:
        ...
    elif (det_coord := detector.bins.coords.get(dim)) is not None:
        ...
    else:
        raise sc.CoordError(
            f"Missing '{dim}' coordinate in detector for monitor normalization."
        )
    lo = det_coord.nanmin()
    hi = det_coord.nanmax()

    if monitor.coords[dim].min() > lo or monitor.coords[dim].max() < hi:
        raise ValueError(
            f"Cannot normalize by monitor: The {dim} range of the monitor "
            f"({monitor.coords[dim].min():c} to {monitor.coords[dim].max():c}) "
            f"is smaller than the range of the detector ({lo:c} to {hi:c})."
        )


def _mask_detector_for_norm(
    *, detector: sc.DataArray, monitor: sc.DataArray
) -> sc.DataArray:
    """Mask the detector where the monitor is masked.

    For performance, this applies the monitor mask to the detector bins.
    This can lead to masking more events than strictly necessary if we
    used an event mask.
    """
    dim = monitor.dim

    if (monitor_mask := _monitor_mask(monitor)) is None:
        return detector

    if (detector_coord := detector.coords.get(dim)) is not None:
        # Apply the mask to the bins or a dense detector.
        mask_da = sc.DataArray(monitor_mask, coords={dim: monitor.coords[dim]})
        if dim in detector_coord.dims:
            # Use rebin to reshape the mask to the detector.
            mask = mask_da.rebin({dim: detector_coord}).data != sc.scalar(0, unit=None)
            return detector.assign_masks(_monitor_mask=mask)
        # else: need to use lookup to apply mask at matching coord elements
        return detector.assign_masks(
            _monitor_mask=sc.lookup(mask_da)[_compute_bin_centers(detector, dim)]
        )

    # else: Apply the mask to the events.
    if dim not in detector.bins.coords:
        raise sc.CoordError(
            f"Detector must have coordinate '{dim}' to mask by monitor."
        )
    event_mask = sc.lookup(
        sc.DataArray(monitor_mask, coords={dim: monitor.coords[dim]})
    )[detector.bins.coords[dim]]
    return detector.bins.assign_masks(_monitor_mask=event_mask)


def _monitor_mask(monitor: sc.DataArray) -> sc.Variable | None:
    """Mask nonfinite and zero monitor values and combine all masks."""
    masks = list(monitor.masks.values())

    finite = sc.isfinite(monitor.data)
    nonzero = monitor.data != sc.scalar(0, unit=monitor.unit)
    valid = finite & nonzero
    if not valid.all():
        masks.append(~valid)

    if not masks:
        return None
    return functools.reduce(sc.logical_or, masks)


def _histogram_monitor_term(
    monitor: sc.DataArray,
    dim: str,
    *,
    broadcast_to: sc.DataArray | None = None,
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> sc.DataArray:
    if not monitor.coords.is_edges(dim, dim):
        raise sc.CoordError(
            f"Monitor coordinage {dim} must be bin-edges for normalization."
        )
    coord = monitor.coords[dim]
    delta_w = sc.DataArray(coord[1:] - coord[:-1], masks=monitor.masks)
    norm = monitor / delta_w

    if broadcast_to is not None:
        return broadcast_uncertainties(
            norm, prototype=broadcast_to, mode=uncertainty_broadcast_mode
        )

    match uncertainty_broadcast_mode:
        case UncertaintyBroadcastMode.fail:
            return norm
        case UncertaintyBroadcastMode.drop:
            return sc.values(norm)
        case _:
            warnings.warn(
                "Cannot broadcast uncertainties in this case.",
                UserWarning,
                stacklevel=3,
            )
            return norm


class _HistogramNormalizationMode(enum.Enum):
    Events = enum.auto()
    """Use an event coordinate to lookup monitor values."""
    BinsCommonDim = enum.auto()
    """Use a bin coordinate which contains the monitor dimension.

    The coordinate may be multi-dimensional but one dimension matches
    the dimension of the monitor.
    """
    BinsDifferentDim = enum.auto()
    """Use a bin coordinate which does not contain the monitor dimension.

    The coordinate may be multi-dimensions, e.g., in the DREAM powder workflow
    where it has dims (two_theta, dspacing [bin-edges]).
    """

    @classmethod
    def deduce(cls, detector: sc.DataArray, dim: str) -> _HistogramNormalizationMode:
        # Use an event-coord when available:
        if detector.bins is not None and dim in detector.bins.coords:
            return _HistogramNormalizationMode.Events
        # else: use a bin-coord.

        det_coord = detector.coords[dim]
        if dim in det_coord.dims:
            return _HistogramNormalizationMode.BinsCommonDim
        return _HistogramNormalizationMode.BinsDifferentDim


def _compute_bin_centers(da: sc.DataArray, name: str) -> sc.Variable:
    coord = da.coords[name]
    for dim in coord.dims:
        if da.coords.is_edges(name, dim):
            coord = sc.midpoints(coord, dim=dim)
    return coord
