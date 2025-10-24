# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Scipp contributors (https://github.com/scipp)
"""Normalization routines for neutron data reduction."""

import itertools

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

    where :math:`\\Delta x_i = x_{i+1} - x_i` is the width of
    monitor bin :math:`i` (see below).
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

    .. Attention::

        Masked bins in ``detector`` are ignored when clipping the monitor and therefore
        impact the normalization factor.
        The output's masked bins are normalized using the same factor and may
        be incorrect and even contain NaN.
        You should only drop masks after normalization if you know what you are doing.

    This function is based on the implementation of
    `NormaliseToMonitor <https://docs.mantidproject.org/nightly/algorithms/NormaliseToMonitor-v1.html>`_
    in Mantid.

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

    detector = _mask_detector_for_norm(detector=detector, monitor=monitor)
    clipped = _clip_monitor_to_detector_range(monitor=monitor, detector=detector)
    coord = clipped.coords[dim]
    delta_w = sc.DataArray(coord[1:] - coord[:-1], masks=clipped.masks)
    total_monitor_weight = broadcast_uncertainties(
        clipped.sum() / delta_w.sum(),
        prototype=clipped,
        mode=uncertainty_broadcast_mode,
    ).data
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

    .. Attention::

        Masked bins in ``detector`` are ignored when clipping the monitor and therefore
        impact the normalization factor.
        The output's masked bins are normalized using the same factor and may
        be incorrect and even contain NaN.
        You should only drop masks after normalization if you know what you are doing.

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
    detector = _mask_detector_for_norm(detector=detector, monitor=monitor)
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

    # Reduce with `all` instead of `any` to include bins in range calculations
    # that contain any unmasked data.
    masks = {
        name: mask.all(set(mask.dims) - {dim}) for name, mask in detector.masks.items()
    }

    # Prefer a bin coord over an event coord because this makes the behavior for binned
    # and histogrammed data consistent. If we used an event coord, we might allow a
    # monitor range that is less than the detector bins which is fine for the events,
    # but would be wrong if the detector was subsequently histogrammed.
    if (det_coord := detector.coords.get(dim)) is not None:
        lo = sc.DataArray(det_coord[dim, :-1], masks=masks).nanmin().data
        hi = sc.DataArray(det_coord[dim, 1:], masks=masks).nanmax().data
    elif (det_coord := detector.bins.coords.get(dim)) is not None:
        lo = sc.DataArray(det_coord, masks=masks).nanmin().data
        hi = sc.DataArray(det_coord, masks=masks).nanmax().data
    else:
        raise sc.CoordError(
            f"Missing '{dim}' coordinate in detector for monitor normalization."
        )

    if monitor.coords[dim].min() > lo or monitor.coords[dim].max() < hi:
        raise ValueError(
            f"Cannot normalize by monitor: The {dim} range of the monitor "
            f"({monitor.coords[dim].min():c} to {monitor.coords[dim].max():c}) "
            f"is smaller than the range of the detector ({lo:c} to {hi:c})."
        )

    if detector.bins is None:
        # If we didn't rebin to the detector coord here, then, for a finer monitor
        # binning than detector, the lookup table would extract one monitor value for
        # each detector bin and ignore other values lying in the same detector bin.
        # But integration would pick up all monitor bins.
        return monitor.rebin({dim: det_coord})
    return monitor[dim, lo:hi]


def _mask_detector_for_norm(
    *, detector: sc.DataArray, monitor: sc.DataArray
) -> sc.DataArray:
    """Mask the detector where the monitor is masked.

    For performance, this applies the monitor mask to the detector bins.
    This can lead to masking more events than strictly necessary if we
    used an event mask.
    """
    if (monitor_mask := _monitor_mask(monitor)) is None:
        return detector

    # Use rebin to reshape the mask to the detector:
    dim = monitor.dim
    mask = sc.DataArray(monitor_mask, coords={dim: monitor.coords[dim]}).rebin(
        {dim: detector.coords[dim]}
    ).data != sc.scalar(0, unit=None)
    return detector.assign_masks({"_monitor_mask": mask})


def _monitor_mask(monitor: sc.DataArray) -> sc.Variable | None:
    """Mask nonfinite monitor values and combine all masks."""
    masks = monitor.masks.values()

    finite = sc.isfinite(monitor.data)
    if not finite.all():
        masks = itertools.chain(masks, (~finite,))

    mask = None
    for m in masks:
        if mask is None:
            mask = m
        else:
            mask |= m

    return mask
