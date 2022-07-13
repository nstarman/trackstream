# see LICENSE.rst

"""Fit a Stream."""

# LOCAL
from .fitter import FitterStreamArmTrack
from .kalman import FirstOrderNewtonianKalmanFilter
from .som import CartesianSelfOrganizingMap1D, UnitSphereSelfOrganizingMap1D
from .track import StreamArmTrack

__all__ = [
    "StreamArmTrack",  # fit result
    "FitterStreamArmTrack",  # overall fitter
    # SOM
    "CartesianSelfOrganizingMap1D",
    "UnitSphereSelfOrganizingMap1D",
    # Kalman
    "FirstOrderNewtonianKalmanFilter",
]
