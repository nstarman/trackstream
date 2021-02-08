# -*- coding: utf-8 -*-

"""Path are an affine-parameterized path.

.. todo::

    Move this elsewhere: utils? or systems.utils?

"""

__all__ = [
    "Path",
]


##############################################################################
# IMPORTS

# BUILT-IN
import copy
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.coordinates import BaseCoordinateFrame

# PROJECT-SPECIFIC
from .interpolate import InterpolatedUnivariateSplinewithUnits as IUSU
from .interpolated_coordinates import (
    InterpolatedCoordinateFrame,
    InterpolatedSkyCoord,
)
from trackstream.type_hints import FrameLikeType, QuantityType
from trackstream.utils._framelike import resolve_framelike

##############################################################################
# CODE
##############################################################################


class Path:
    """Path are an affine-parameterized path.

    Parameters
    ----------
    path : coord-like array or `BaseRepresentation` subclass instance
        The central path of the path.

        If `BaseRepresentation`, then need to pass `frame`.
        If not interpolated (`InterpolatedSkyCoord`,
        `InterpolatedCoordinateFrame` or `InterpolatedRepresentation`), then
        need to pass `affine`.
        The best is `InterpolatedSkyCoord` or `InterpolatedCoordinateFrame`

    width : `Quantity` scalar or array or callable
        The width around `path`.

        Must be in physical distance.
        If array, must match `path` length
        If callable, must accept `path` affine as 1st (and only mandatory)
        argument

        .. todo::

            allow angular distance and convert using path distance?
    name : str (optional, keyword-only)
        Name of the path

    Other Parameters
    ----------------
    affine : `Quantity` array (optional, keyword-only)
        Affine parameter along `path`.
        Only used if `path` is not already interpolated.

    frame : frame-like or None (optional, keyword-only)
        The preferred frame of the data (`path`)
        If None (default), taken from the config (``conf.default_frame``)
        unless `path` has a frame (is not `BaseRepresentation`).
        If `path` is `BaseRepresentation`, then it is assumed in this frame.

    Raises
    ------
    Exception
        if `path` is not already interpolated and affine is None

    """

    def __init__(
        self,
        path: T.Union[InterpolatedCoordinateFrame, InterpolatedSkyCoord],
        width: T.Union[QuantityType, T.Callable, None] = None,  # func(affine)
        *,
        name: str = None,
        affine: T.Optional[QuantityType] = None,
        frame: T.Optional[FrameLikeType] = None,
    ):
        super().__init__()

        # -----------------------
        # Frame, name, & metadata

        if frame is None:
            # unless `path` has a frame (is not `BaseRepresentation`).
            if isinstance(path, BaseCoordinateFrame):
                frame = path.replicate_without_data()
            elif hasattr(path, "frame"):  # things like SkyCoord
                frame = path.frame.replicate_without_data()

        self.name = name
        self._frame = resolve_framelike(frame)  # (an instance, not class)
        # self.meta.update(meta)

        # --------------
        # path

        self._original_path = path.copy()  # original path. For safekeeping.

        # options are: BaseRepresentation, InterpolatedRepresentation
        #              BaseCoordinateFrame, InterpolatedCoordinateFrame
        #              SkyCoord, InterpolatedSkyCoord
        # need to end up with a InterpolatedSkyCoord
        if isinstance(path, coord.BaseRepresentation):  # works for interp
            path = self.frame.realize_frame(path)

        if isinstance(path, coord.BaseCoordinateFrame):  # works for interp
            path = InterpolatedCoordinateFrame(path, affine=affine)

        path = InterpolatedSkyCoord(path, affine=affine)
        self._path = path.transform_to(self.frame)

        # --------------
        # Width
        # this needs to be in physical coordinates
        # the initialization is separated out so that base classes can pass
        # `None` here and outside do stuff like have angular widths.

        if width is not None:
            self._initialize_width(path, width)

    # /def

    @property  # read-only
    def frame(self):
        """The preferred frame (instance) of the Footprint."""
        return self._frame

    # /def

    @property
    def path(self):
        """The path, protected."""
        return self._path

    # /def

    @property
    def affine(self):
        """Affine parameter along ``path``."""
        return self.path.affine

    # /def

    # ---------------------------------------------------------------

    def _initialize_width(self, path, width):
        # TODO clean this up and stuff
        # this is separated out so that base classes
        # can do stuff like have angular widths

        if callable(width):
            # just testing
            _ws = width(path.affine)
            if _ws.unit.physical_type != "length":
                raise ValueError("width must have units of length")

            o_w = width

        else:
            # clean
            o_w = u.Quantity(width, copy=False)

            # check
            if o_w.unit.physical_type != "length":
                raise ValueError("width must have units of length")

            # interpolate
            # first check if need to broadcast
            _ws = np.ones(len(path)) * o_w if o_w.isscalar else o_w
            width = IUSU(path.affine, _ws)

        self._original_width = copy.deepcopy(o_w)
        self._width_fn = width

    # /def

    def width(self, affine: T.Optional[QuantityType] = None):
        if affine is None:
            affine = self.affine
        return self._width_fn(u.Quantity(affine, copy=False))

    # /def

    #################################################################
    # Math on the Track!

    def __call__(self, affine: QuantityType):
        """Call."""
        meanpath = self.path(affine)
        width = self.width(affine)
        # TODO allow for higher moments

        return meanpath, width  # TODO! see FootprintsPackage

    # /def

    # def separation(self, c):
    #     raise NotImplementedError("TODO")

    # def likelihood_distance(self, c, errs, method="kullback_leibler"):
    #     """the likelihood distance."""
    #     raise NotImplementedError("TODO")

    #################################################################
    # Miscellaneous

    # def _preferred_frame_resolve(self, frame):
    #     """Call `resolve_framelike`, but default to preferred frame.

    #     For frame is None ``resolve_framelike`` returns the default
    #     frame from the config file. Instead, we want the default
    #     frame of the footprint.

    #     Returns
    #     -------
    #     `BaseCoordinateFrame` subclass instance
    #         Has no data.

    #     """
    #     if frame is None:
    #         frame = self.frame

    #     return resolve_framelike(frame)

    # # /def


# /class

##############################################################################
# END
