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
import astropy.units as u
import numpy as np
from astropy.table import QTable
from utilipy.utils import typing as ET

from .preprocess import RotatedFrameFitter

##############################################################################
# PARAMETERS

ICRSType = T.TypeVar("ICRSType", coord.ICRS, coord.ICRS)
QTableType = T.TypeVar("QTableType", QTable, QTable)


##############################################################################
# CODE
##############################################################################


class TrackStream:
    """Track a Stream.

    When run, produces a StreamTrack

    Parameters
    ----------
    data : |QTable| or |BaseRep| (or subclass) instance
        The stream data.
        If Representation instance, must be convertible to |CartesianRep|
        If QTable, split into a |CartesianRep| and QTable of errors
        (Note this operation copies the data).

    data_err : |QTable|, optional
        Only used if 'data' is Representation instance, since this cannot
        hold error information.
        The data_err must have (at least) column names
        ["x_err", "y_err", "z_err"]

    Notes
    -----
    This method brings together a few different classes and techniques.
    More granular control can be achieved by using each piece separately.

    ..
      RST SUBSTITUTIONS

    .. |QTable| replace:: `~astropy.table.QTable`
    .. |BaseRep| replace:: `~astropy.coordinates.BaseRepresentation`
    .. |CartesianRep| replace:: `~astropy.coordinates.CartesianRepresentation`

    """

    def __init__(
        self,
        data: T.Union[QTableType, coord.BaseCoordinateFrame],
        origin: ICRSType,
        data_err: T.Optional[QTableType] = None,
        rotated_frame: T.Optional[ET.CoordinateType] = None,
        SOM=None,
    ):
        super().__init__()

        self.origin = origin
        self.rotated_frame = rotated_frame

        self.SOM = SOM

        # ----------
        # process the data
        # The data is stored as a Representation object, the error as a QTable

        if isinstance(data, coord.BaseCoordinateFrame):
            # This works, but it doesn't have errors,
            # so we need to deal with that first.

            err_colnames = ["x_err", "y_err", "z_err"]
            if data_err is None:  # make a table matching data with errors of 0
                data_err = QTable(np.zeros((len(data), 3)), names=err_colnames)
            else:  # there are provided errors
                # check has correct columns
                if not set(err_colnames).issubset(data_err.colnames):
                    raise ValueError(
                        f"data_err does not have columns {err_colnames}"
                    )

            self.data_err = data_err
            self.orig_data = data
            self.data = data

        elif isinstance(data, QTable):  # will ignore data_err
            self.orig_data = data
            self.data = coord.CartesianRepresentation(data["x", "y", "z"])
            self.data_err = data[err_colnames]

        else:
            raise TypeError("data is not QTable or BaseRepresentation")

        # ----------

        self.visit_order = np.arange(len(self.data))

    # /def

    # ----------------------------------------------------
    # Preprocessing

    def fit_rotated_frame(self, reorder=True, **kwargs):
        #  store kwargs used in fitting
        self._frame_fit_kwargs = kwargs  # TODO do better with kwargs

        # make instance of fitter class
        fitter = RotatedFrameFitter(
            data=self.data, origin=self.origin, **kwargs
        )

        result = fitter.fit(rot0=0 * u.deg)  # TODO, override defaults?

        self._fit_result = result
        self.rotated_frame = result.frame
        self.data = result.data

        if reorder:
            self.visit_order = result.lon_order
            self.data = self.data[result.lon_order]

    # /def

    # ----------------------------------------------------

    def fit_single_SOM(self, start_point):
        """Fit SOM.

        .. todo::

            More logical organization




        """

        raise NotImplementedError

    # /def

    def fit_SOM(
        self,
        start_point,
        N_repeats,
        iterations: int = int(3e3),
        learning_rate: float = 1.5,
        sigma: float = 15,
    ):
        """Fit SOM, with cross-validation.

        .. todo::

            More logical organization


        S


        """

        raise NotImplementedError

    # /def

    # ----------------------------------------------------
    # Kalman

    def fit_kalman(self):
        """Fit Kalman Filter"""

        raise NotImplementedError

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
        if self.rotated_frame is None and fit_frame_if_needed:
            self.fit_rotated_frame()

        self.track = StreamTrack(self.data)

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

    # ----------------------------------------------------


# /class


# -------------------------------------------------------------------


class StreamTrack(object):
    """A stream track interpolation as function of arc length.

    Parameters
    ----------
    data : `~astropy.coordinates.BaseCoordinateFrame`
    interpolations : dict
        of `~scipy.interpolate.InterpolatedUnivariateSpline`
        Should have components that match the data representation type


    """

    def __init__(self, data, interpolations=None):
        super().__init__()

        self._data = data
        self._interp = interpolations

    # /def

    def __call__(self, arc_length):
        """Get discrete points along interpolated stream track.

        .. todo::

            Implement astropy-style representation parsing

        """
        rep = self._data.representation_type(
            **{k: v(arc_length) for k, v in self._interp.items()}
        )

        return self._data.realize_frame(rep)

    # /def

    # ---------------------

    def plot_data(self,):
        # THIRD PARTY
        import matplotlib.pyplot as plt

        plt.scatter(self._data.lon, self._data.lat)

    # /def


# /class


##############################################################################
# END
