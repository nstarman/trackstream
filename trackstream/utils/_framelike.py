# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Resolve Frame-Like."""

__all__ = [
    "resolve_framelike",
]


##############################################################################
# IMPORTS

# STDLIB
import typing as T

# THIRD PARTY
from astropy.coordinates import BaseCoordinateFrame, SkyCoord, sky_coordinate_parsers

# LOCAL
from trackstream._type_hints import FrameLikeType
from trackstream.config import conf

##############################################################################
# CODE
##############################################################################


def resolve_framelike(
    frame: T.Optional[FrameLikeType],
    error_if_not_type: bool = True,
):
    """Determine the frame and return a blank instance.

    Parameters
    ----------
    frame : frame-like instance or None (optional)
        If BaseCoordinateFrame, replicates without data.
        If str, uses astropy parsers to determine frame class
        If None (default), gets default frame name from config, and parses.

    error_if_not_type : bool
        Whether to raise TypeError if `frame` is not one of the allowed types.

    Returns
    -------
    frame : `~astropy.coordinates.BaseCoordinateFrame` instance
        Replicated without data.

    """
    # If no frame is specified, assume that the input footprint is in a
    # frame specified in the configuration
    if frame is None:
        frame: str = conf.default_frame

    if isinstance(frame, str):
        frame = sky_coordinate_parsers._get_frame_class(frame.lower())()
    elif isinstance(frame, BaseCoordinateFrame):
        frame = frame.replicate_without_data()
    elif isinstance(frame, SkyCoord):
        frame = frame.frame.replicate_without_data()

    elif error_if_not_type:
        raise TypeError(
            "Input coordinate frame must be an astropy "
            "coordinates frame subclass *instance*, not a "
            "'{}'".format(frame.__class__.__name__),
        )

    return frame


# /def


##############################################################################
# END
