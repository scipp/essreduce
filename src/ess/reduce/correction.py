# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Correction algorithms for neutron data reduction."""

import scipp as sc

from .uncertainty import UncertaintyBroadcastMode, broadcast_uncertainties


def normalize_by_monitor_histogram(
    detector: sc.DataArray,
    *,
    monitor: sc.DataArray,
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> sc.DataArray:
    """Normalize detector data by a histogrammed monitor.

    First, the monitor is clipped to the range of the detector

    .. math::

        \\bar{m}_i = m_i I(x_i, x_{i+1}),

    where :math:`m_i` is the monitor intensity in bin :math:`i`,
    :math:`x_i` is the lower bin edge of bin :math:`i`, and
    :math:`I(x_i, x_{i+1})` selects bins that are within the range of the detector.

    The detector bins :math:`d_i` are normalized according to

    .. math::

        d_i^\\text{Norm} = \\frac{d_i}{\\bar{m}_i} \\Delta x_i
        \\frac{\\sum_j\\,\\bar{m}_j}{\\sum_j\\,\\Delta x_j}

    where :math:`\\Delta x_i` is the width of monitor bin :math:`i` (see below).
    This normalization leads to a result that has the same
    unit as the input detector data.

    Monitor bin :math:`i` is chosen according to:

    - *Histogrammed detector*: The monitor is
      `rebinned <https://scipp.github.io/generated/functions/scipp.rebin.html>`_
      to the detector binning. This distributes the monitor weights to the
      detector bins.
    - *Binned detector*: The monitor value for bin :math:`i` is determined via
      :func:`scipp.lookup`. This means that for each event, the monitor value
      is obtained from the monitor histogram at that event coordinate value.

    This function is based on the implementation in
    `NormaliseToMonitor <https://docs.mantidproject.org/nightly/algorithms/NormaliseToMonitor-v1.html>`_
    of Mantid.

    Parameters
    ----------
    detector:
        Input detector data.
        Must have a coordinate named ``monitor.dim``, that is, the single
        dimension name of the monitor.
    monitor:
        A histogrammed monitor.
        Must be one-dimensional and have a dimension coordinate, typically "wavelength".
    uncertainty_broadcast_mode:
        Choose how uncertainties of the monitor are broadcast to the sample data.

    Returns
    -------
    :
        ``detector`` normalized by ``monitor``.

    See also
    --------
    normalize_by_monitor_integrated:
        Normalize by an integrated monitor.
    """
    dim = monitor.dim

    clipped = _clip_monitor_to_detector_range(monitor=monitor, detector=detector)
    coord = clipped.coords[dim]
    delta_w = coord[1:] - coord[:-1]
    total_monitor_weight = broadcast_uncertainties(
        clipped.sum() / delta_w.sum(),
        prototype=clipped,
        mode=uncertainty_broadcast_mode,
    )
    delta_w *= total_monitor_weight
    norm = broadcast_uncertainties(
        clipped / delta_w, prototype=detector, mode=uncertainty_broadcast_mode
    )

    if detector.bins is None:
        return detector / norm
    return detector.bins / sc.lookup(norm, dim=dim)


def normalize_by_monitor_integrated(
    detector: sc.DataArray,
    *,
    monitor: sc.DataArray,
    uncertainty_broadcast_mode: UncertaintyBroadcastMode,
) -> sc.DataArray:
    """Normalize detector data by an integrated monitor.

    The monitor is integrated according to

    .. math::

        M = \\sum_{i=0}^{N-1}\\, m_i (x_{i+1} - x_i) I(x_i, x_{i+1}),

    where :math:`m_i` is the monitor intensity in bin :math:`i`,
    :math:`x_i` is the lower bin edge of bin :math:`i`, and
    :math:`I(x_i, x_{i+1})` selects bins that are within the range of the detector.

    Parameters
    ----------
    detector:
        Input detector data.
    monitor:
        A histogrammed monitor.
        Must be one-dimensional and have a dimension coordinate, typically "wavelength".
    uncertainty_broadcast_mode:
        Choose how uncertainties of the monitor are broadcast to the sample data.

    Returns
    -------
    :
        `detector` normalized by a monitor.

    See also
    --------
    normalize_by_monitor_histogram:
        Normalize by a monitor histogram without integration.
    """
    clipped = _clip_monitor_to_detector_range(monitor=monitor, detector=detector)
    coord = clipped.coords[clipped.dim]
    norm = (clipped * (coord[1:] - coord[:-1])).data.sum()
    norm = broadcast_uncertainties(
        norm, prototype=detector, mode=uncertainty_broadcast_mode
    )
    return detector / norm


def _clip_monitor_to_detector_range(
    *, monitor: sc.DataArray, detector: sc.DataArray
) -> sc.DataArray:
    dim = monitor.dim
    if not monitor.coords.is_edges(dim):
        raise sc.CoordError(
            f"Monitor coordinate '{dim}' must be bin-edges to integrate the monitor."
        )

    # Prefer a bin coord over an event coord because this makes the behavior for binned
    # and histogrammed data consistent. If we used an event coord, we might allow a
    # monitor range that is less than the detector bins which is fine for the vents,
    # but would be wrong if the detector was subsequently histogrammed.
    if dim in detector.coords:
        det_coord = detector.coords[dim]

        # Mask zero-count bins, which are an artifact from the rectangular 2-D binning.
        # The wavelength of those bins must be excluded when determining the range.
        if detector.bins is None:
            mask = detector.data == sc.scalar(0.0, unit=detector.unit)
        else:
            mask = detector.data.bins.size() == sc.scalar(0.0, unit=None)
        lo = (
            sc.DataArray(det_coord[dim, :-1], masks={'zero_counts': mask}).nanmin().data
        )
        hi = sc.DataArray(det_coord[dim, 1:], masks={'zero_counts': mask}).nanmax().data

    elif dim in detector.bins.coords:
        det_coord = detector.bins.coords[dim]
        # No need to mask here because we have the exact event coordinate values.
        lo = det_coord.nanmin()
        hi = det_coord.nanmax()

    else:
        raise sc.CoordError(
            f"Missing '{dim}' coordinate in detector for monitor normalization."
        )

    if monitor.coords[dim].min() > lo or monitor.coords[dim].max() < hi:
        raise ValueError(
            f"Cannot normalize by monitor: The {dim} range of the monitor "
            f"({monitor.coords[dim].min().value} to {monitor.coords[dim].max().value}) "
            f"is smaller than the range of the detector ({lo.value} to {hi.value})."
        )

    if detector.bins is None:
        # If we didn't rebin to the detector coord here, then, for a finer monitor
        # binning than detector, the lookup table would extract one monitor value for
        # each detector bin and ignore other values lying in the same detector bin.
        # But integration would pick up all monitor bins.
        return monitor.rebin({dim: det_coord})
    return monitor[dim, lo:hi]
