# -*- coding: utf-8 -*-

"""Core Functions."""

__all__ = [
    "TrackStream",
    "StreamTrack",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import numpy as np
from astropy.table import QTable
from astropy.utils.misc import indent

# PROJECT-SPECIFIC
from .type_hints import (
    CoordinateType,
    FrameType,
    QuantityType,
    TableType,
)

##############################################################################
# PARAMETERS

##############################################################################
# CODE
##############################################################################


class TrackStream:
    """Track a Stream.

    When run, produces a StreamTrack.

    Parameters
    ----------
    data : |Table| or |CoordinateFrame| instance
        The stream data.
        Must be convertible to |CartesianRep|

    origin : :class:`~astropy.coordinates.ICRS`
        The origin point of the rotated reference frame.

    data_err : |QTable| (optional)
        The data_err must have (at least) column names
        ["x_err", "y_err", "z_err"]

    frame : |CoordinateFrame| or None (optional, keyword-only)
        The stream frame. Locally linearizes the data.
        If not None, need to fit for the frame (default).

    Notes
    -----
    This method brings together a few different classes and techniques.
    More granular control can be achieved by using each piece separately.

    """

    def __init__(
        self,
        data: T.Union[QTable, coord.BaseCoordinateFrame],
        origin: FrameType,
        data_err: T.Optional[TableType] = None,
        *,
        frame: T.Optional[CoordinateType] = None,
    ):
        super().__init__()

        self.origin: SkyCoordType = coord.SkyCoord(origin, copy=False)
        self.frame: T.Optional[FrameType] = frame

        # ----------
        # process the data
        # The data is stored as a Representation object, the error as a QTable
        # ----------
        # SOM

        self.SOM = None
        self.visit_order = None

        # Step 1) Convert to Representation, storing the Frame type
        # Step 2) Convert to CartesianRepresentation, storing the Rep type
        if isinstance(data, coord.BaseCoordinateFrame):
            self._data_frame = data.replicate_without_data()
            self._data_rep = data.representation_type
            self._data = data._data
        elif isinstance(data, coord.BaseRepresentation):
            self._data_frame = coord.BaseCoordinateFrame()
            self._data_rep = data.__class__
            self._data = data
        else:
            raise TypeError(f"`data` type <{type(data)}> is wrong.")
        # /if

        # Step 3) Convert to Cartesian Representation
        data = data.represent_as(coord.CartesianRepresentation)

        # Step 4) Get the errors
        # The data does not have errors, we need to construct it here.
        err_colnames = ["x_err", "y_err", "z_err"]
        if data_err is None:  # make a table matching data with errors of 0
            data_err = QTable(np.zeros((len(data), 3)), names=err_colnames)
        else:  # there are provided errors
            # check has correct columns
            if not set(err_colnames).issubset(data_err.colnames):
                raise ValueError(
                    f"data_err does not have columns {err_colnames}",
                )

        self.data_err = data_err
        self.orig_data = data
        self.data = data

        # ----------
        # SOM


    # /def

    # ----------------------------------------------------
    # Fit

    def fit(
        self,
        # frame fitting
        fit_frame_if_needed=True,
    ):
        """Fit a data to the data.

        Returns
        -------
        StreamTrack instance
            Also stores as ``.track``

        """
        # reconstruct data
        # TODO handle velocities
        rep = self._data_rep.from_cartesian(self.data)
        data = self._data_frame.realize_frame(rep)

        # get interpolation
        # this is the real meat of the project
        interpolation = None

        # construct interpolation
        self.track = StreamTrack(
            interpolation,
            stream_data=data,
            origin=self.origin,
        )

        return self.track

    # /def

    def predict(self, arc_length):
        """Predict from a fit.

        Returns
        -------
        StreamTrack instance

        """
        return self.track(arc_length)

    # /def

    def fit_predict(self, arc_length, **fit_kwargs):
        """Fit and Predict."""
        self.fit(**fit_kwargs)
        return self.predict(arc_length)

    # /def


# /class


# -------------------------------------------------------------------


class StreamTrack:
    """A stream track interpolation as function of arc length.

    The track is Callable, returning a Frame.

    Parameters
    ----------
    data : `~astropy.coordinates.BaseCoordinateFrame`
    interpolation : dict
        of `~scipy.interpolate.InterpolatedUnivariateSpline`
        Should have components that match the data representation type

    """

    def __init__(
        self,
        track,
        stream_data: T.Optional[CoordinateType],
        origin,
    ):
        super().__init__()

        self._track = track
        self.origin = origin

        if isinstance(stream_data, coord.BaseCoordinateFrame):
            self._data_frame = stream_data.replicate_without_data()
            self._data_rep = stream_data.representation_type
            self._data = stream_data._data
        elif isinstance(stream_data, coord.BaseRepresentation):
            self._data_frame = coord.BaseCoordinateFrame()
            self._data_rep = stream_data.__class__
            self._data = stream_data
        else:
            raise TypeError(
                f"`stream_data` type <{type(stream_data)}> is wrong.",
            )
        # /if

    # /def

    #######################################################
    # Math on the Track

    def __call__(self, arc_length: QuantityType) -> CoordinateType:
        """Get discrete points along interpolated stream track.

        .. todo::

            Implement astropy-style representation parsing

        Parameters
        ----------
        arc_length: Quantity

        Returns
        -------
        |CoordinateFrame|
            Realized from the ``.data`` attribute Frame and Representation
            from the interpolation (``.interp``).

        """
        rep = self._data_rep(
            **{k: v(arc_length) for k, v in self._track.items()}
        )

        return self._data_frame.realize_frame(rep)

    # /def

    #######################################################
    # misc

    def __repr__(self):
        """String representation."""
        s = super().__repr__()

        frame_name = self._data_frame.__class__.__name__
        rep_name = self._data_rep.__name__
        s = s.replace("StreamTrack", f"StreamTrack ({frame_name}|{rep_name})")

        s += "\n" + indent(repr(self._data)[1:-1])

        return s

    # /def


# /class


##############################################################################
# END
