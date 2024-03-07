# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2024 Scipp contributors (https://github.com/scipp)

"""Logging tools for ess.reduce."""

import logging


def get_logger() -> logging.Logger:
    """Return the logger for ess.reduce.

    Returns
    -------
    :
        The requested logger.
    """
    return logging.getLogger('scipp.ess.reduce')
