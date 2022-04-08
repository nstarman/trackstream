# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Kalman Filter."""

__all__ = ["FirstOrderNewtonianKalmanFilter"]

##############################################################################
# IMPORTS

# LOCAL
from . import core, helper, plot  # noqa: F401, F403
from .core import FirstOrderNewtonianKalmanFilter
