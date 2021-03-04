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
import warnings

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.table import QTable, Table, hstack
from astropy.utils.metadata import MetaAttribute, MetaData
from astropy.utils.misc import indent
from scipy.linalg import block_diag

# PROJECT-SPECIFIC
from . import type_hints as TH
from trackstream.preprocess import som
from trackstream.preprocess.rotated_frame import RotatedFrameFitter
from trackstream.process.kalman import KalmanFilter
from trackstream.process.utils import make_dts, make_F, make_H, make_Q, make_R
from trackstream.utils._framelike import resolve_framelike
from trackstream.utils.path import Path
from trackstream.utils.utils import intermix_arrays

##############################################################################
# PARAMETERS

##############################################################################
# CODE
##############################################################################


class TrackStream:
    """Track a Stream in ICRS coordinates.

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
        origin: TH.FrameType,
        data_err: T.Optional[TH.TableType] = None,
        *,
        frame: T.Optional[TH.CoordinateType] = None,
    ):
        super().__init__()

        self.origin: TH.SkyCoordType = coord.SkyCoord(origin, copy=False)
        self.frame: T.Optional[TH.FrameType] = frame

        # the cache
        self._cache: T.Dict[str, object] = {}

        # ----------
        # process the data

        # properties of the original data
        self._data_frame: T.Optional[TH.FrameType] = None  # frame type
        self._data_rep: T.Optional[TH.RepresentationType] = None
        # original data (representation )
        self._data: T.Optional[TH.PositionType] = None
        # TODO, store the actual original data

        # processed data -> QTable
        self.data = None
        self._parse_data(data=data, data_err=data_err)

        # ----------
        # SOM

        self.SOM = None
        self.visit_order = None

    # /def

    @property
    def points(self):
        return self.data["coord"]

    # /def

    @property
    def original_data(self):
        # TODO, don't make new one each time
        return self._data_frame.realize_frame(self._data)

    # /def

    # ----------------------------------------------------

    def _parse_data(
        self,
        data: T.Union[
            Table, coord.BaseCoordinateFrame, coord.BaseRepresentation,
        ],
        data_err: T.Optional[TH.TableType] = None,
    ):
        """Parse the data table.

        - the frame is stored in ``_data_frame``
        - the representation is stored in ``_data_rep``
        - the original data representation  is in ``_data``

        .. todo::

            break this function out?

        Parameters
        ----------
        data : |Table| or |CoordinateFrame| or instance
            The stream data.

        data_err : |Table| or |CoordinateFrame| or instance
            It must be in Cartesian coordinates!

        Raises
        ------
        TypeError
            if `data` is not |Table| or |CoordinateFrame|

        """
        # ----------
        # 1) the data

        # Step 1) Convert to Representation, storing the Frame type
        # Step 2) Convert to CartesianRepresentation, storing the Rep type
        if isinstance(data, (coord.BaseCoordinateFrame, coord.SkyCoord)):
            self._data_frame = data.replicate_without_data()
            self._data_rep = data.representation_type
            self._data = data.data
        elif isinstance(data, Table):
            _data = coord.SkyCoord.guess_from_table(data)
            self._data_frame = _data.replicate_without_data()
            self._data_rep = _data.representation_type
            self._data = _data.data
        else:
            raise TypeError(f"`data` type {type(data)} is wrong.")

        # Step 3) Convert to Cartesian Representation
        _data = self._data.represent_as(coord.CartesianRepresentation)

        # ----------
        # 2) the error

        err_cols = ["x_err", "y_err", "z_err"]

        # first munge the data
        if data_err is not None:
            pass
        # try to extract from data table
        elif isinstance(data, Table) and set(err_cols).issubset(data.colnames):
            data_err = data[err_cols]
        else:
            warnings.warn(
                "No data_err and cannot extract from `data`."
                "Assuming errors of 0.",
            )
            data_err = QTable(np.zeros((len(data), 3)), names=err_cols)

        if isinstance(data_err, Table):
            data_err = data_err[err_cols]  # extract columns
        elif isinstance(data_err, coord.BaseCoordinateFrame):
            if not isinstance(data_err.data, coord.CartesianRepresentation):
                raise TypeError("data_err must be Cartesian.")
            rep = data_err.data
            data_err = QTable(dict(x_err=rep.x, y_err=rep.y, z_err=rep.z))
        elif isinstance(data_err, coord.BaseRepresentation):
            if not isinstance(data_err, coord.CartesianRepresentation):
                raise TypeError("data_err must be Cartesian.")
            data_err = QTable(dict(x_err=rep.x, y_err=rep.y, z_err=rep.z))
        else:
            raise TypeError(f"`data_err` type <{type(data_err)}> is wrong.")

        # ----------
        # 3) Return

        sc = coord.SkyCoord(self._data_frame.realize_frame(_data))

        table = hstack([QTable(dict(coord=sc)), data_err])
        self.data = table

    # /def

    #################################################################
    # Fit

    def _fit_rotated_frame(
        self,
        rot0: T.Optional[u.Quantity] = 0 * u.deg,
        bounds: T.Optional[T.Sequence] = None,
        **kwargs,
    ):
        """Fit a rotated frame in ICRS coordinates.

        Parameters
        ----------
        rot0 : |Quantity| or None.
            Initial guess for rotation.
        bounds : array-like or None, optional
            Parameter bounds. If None, these are automatically constructed.
            ::
                [[rot_low, rot_up],
                 [lon_low, lon_up],
                 [lat_low, lat_up]]

        Other Parameters
        ----------------
        rot_lower, rot_upper : |Quantity|, (optional, keyword-only)
            The lower and upper bounds in degrees.
            Default is (-180, 180] degree.
        origin_lim : |Quantity|, (optional, keyword-only)
            The symmetric lower and upper bounds on origin in degrees.
            Default is 0.005 degree.

        fix_origin : bool or None (optional, keyword-only)
            Whether to fix the origin point. Default is False.
        use_lmfit : bool or None (optional, keyword-only)
            Whether to use ``lmfit`` package.
            None (default) falls back to config file.
        leastsquares : bool or None (optional, keyword-only)
            If `use_lmfit` is False, whether to to use
            :func:`~scipy.optimize.least_square` or
            :func:`~scipy.optimize.minimize`
            Default is False

        align_v : bool or None (optional, keyword-only)
            Whether to align velocity to be in positive direction

        Raises
        ------
        TypeError
            If ``_data_frame`` is None

        """
        fitter = RotatedFrameFitter(
            data=self.points.icrs, origin=self.origin, **kwargs,
        )

        fit = fitter.fit(rot0=rot0, bounds=bounds)

        # cache
        self._cache["frame"] = fit.frame  # SkyOffsetICRS
        self._cache["frame_fit"] = fit

        return fit.frame, fit

    # /def

    # -------------------------------------------

    def _fit_SOM(
        self,
        data,
        *,
        learning_rate: float = 2.0,
        sigma: float = 20.0,
        iterations: int = 10000,
        random_seed: T.Optional[int] = None,
        reorder: T.Optional[int] = None,
        plot: bool = False,
    ):
        """Reorder data by SOM.

        .. todo::

            - iterative training

        """
        # if self.frame is not None:
        #     frame = self.frame
        # elif self._cache.get("frame", None) is not None:
        #     frame = self._cache["frame"]
        # else:
        #     frame = self.points.frame.replicate_without_data()

        # data = self.points.transform_to(frame)

        # # -------------------

        visit_order, SOM = som.apply_SOM(  # TODO iterative training
            data,
            learning_rate=learning_rate,
            sigma=sigma,
            iterations=iterations,
            random_seed=random_seed,
            reorder=reorder,
            plot=plot,
            return_som=True,
        )

        # cache
        self._cache["visit_order"] = visit_order
        self._cache["SOM"] = SOM

        return visit_order, SOM

    # /def

    def _run_kalman_filter(self, data: TH.SkyCoordType, w0=None):
        """Fit data with Kalman filter

        Parameters
        ----------
        w0

        """
        print(data[:4])
        arr = data.cartesian.xyz.T.value
        dts = make_dts(arr, dt0=0.5, N=6, axis=1, plot=False)

        if w0 is None:
            x = arr[0]  # fist point
            v = [0, 0, 0]
        w0 = intermix_arrays(x, v)

        # TODO! as options
        p = np.array([[0.0001, 0], [0, 1]])
        P0 = block_diag(p, p, p)

        H0 = make_H()
        R0 = make_R([0.05, 0.05, 0.003])[0]  # TODO! actual errors

        self.kalman_filter = kf = KalmanFilter(
            w0,
            P0,
            F0=make_F,
            Q0=make_Q,
            H0=H0,
            R0=R0,
            q_kw=dict(var=0.01, n_dims=3),  # TODO! as options
        )

        mean_path = kf.run(arr, dts, method="stepupdate", use_filterpy=None)

        self._cache["mean_path"] = mean_path
        self._cache["kalman"] = kf

        return mean_path, kf

    # /def

    # -------------------------------------------

    def fit(
        self,
        *,
        fit_frame_if_needed: bool = True,
        rotated_frame_fit_kw: T.Optional[dict] = None,
        fit_SOM_if_needed: bool = True,
        som_fit_kw: T.Optional[dict] = None,
        kalman_fit_kw: T.Optional[dict] = None,
    ):
        """Fit a data to the data.

        Parameters
        ----------
        fit_frame : bool
            Only fits frame if ``self.frame`` is None
            The fit frame is ICRS always.

            .. todo::

                make fitting work in the frame of the data

        Returns
        -------
        StreamTrack instance
            Also stores as ``.track``

        """
        # -------------------
        # fit rotated frame

        # 1) already provided
        frame = self.frame  # can be None
        frame_fit = None  # no fit info

        # 2) try to get from cache
        if frame is None:
            frame = self._cache.get("frame", None)
            frame_fit = self._cache.get("frame_fit", None)

        # 3) fit (& cache), if still None
        if frame is None and fit_frame_if_needed:
            rotated_frame_fit_kw = rotated_frame_fit_kw or {}
            frame, frame_fit = self._fit_rotated_frame(**rotated_frame_fit_kw)

        # 4) if it's still None, give up
        if frame is None:
            frame = self.data["coord"].frame.replicate_without_data()
            frame_fit = None

        # now rotate the data to the correct frame
        data = self.points.transform_to(frame)

        # -------------------
        # Self-organizing Map

        # 1) already provided
        visit_order = self.visit_order
        SOM = self.SOM

        # 2) try to get from cache
        if visit_order is None:
            visit_order = self._cache.get("visit_order", None)
            SOM = self._cache.get("SOM", None)

        # 3) fit, if still None
        if visit_order is None and fit_SOM_if_needed:
            som_fit_kw = som_fit_kw or {}
            visit_order, SOM = self._fit_SOM(data, **som_fit_kw)

        # 4) if it's still None, give up
        if visit_order is None:
            visit_order = np.arange(0, len(data))

        # now rearrange the data
        data = data[visit_order]

        # -------------------
        # Kalman Filter

        kalman_fit_kw = kalman_fit_kw or {}
        mean_path, kalman_filter = self._run_kalman_filter(
            data, **kalman_fit_kw
        )

        # -------------------

        # mean_path = self.points.transform_to(self.frame)  # FIXME!

        # path = Path(
        #     path=mean_path,
        #     width=100 * u.pc,  # FIXME!
        #     affine=mean_path.spherical.lon,
        #     frame=self.frame,
        # )

        # # construct interpolation
        track = StreamTrack(
            mean_path,
            stream_data=self.data,
            origin=self.origin,
            frame=frame,
            # extra
            frame_fit=frame_fit,
            visit_order=visit_order,
            som=SOM,
            kalman=kalman_filter,
        )
        return track

    # /def

    #################################################################
    # Other methods

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
    path : `~trackstream.utils.path.Path`
    stream_data
        Original stream data
    origin
        of the coordinate system (often the progenitor)

    """

    meta = MetaData()

    def __init__(
        self,
        path: Path,
        stream_data: T.Union[TH.TableType, TH.CoordinateType, None],
        origin: TH.CoordinateType,
        frame: TH.FrameLikeType,
        **metadata,
    ):
        super().__init__()

        # type validation
        if not isinstance(path, Path):
            raise TypeError("`path` must be <Path>.")
        elif not isinstance(
            origin, (coord.SkyCoord, coord.BaseCoordinateFrame)
        ):
            raise TypeError(
                "`origin` must be <|SkyCoord|, |CoordinateFrame|>."
            )

        # assign
        self._path: Path = path
        self._origin = origin
        self._frame = resolve_framelike(frame)

        self._stream_data = stream_data

        # set the MetaAttribute(s)
        for attr in list(metadata):
            descr = getattr(self.__class__, attr, None)
            if isinstance(descr, MetaAttribute):
                setattr(self, attr, metadata.pop(attr))

        self.meta.update(metadata)

    # /def

    @property
    def path(self):
        return self._path

    # /def

    @property
    def track(self):
        """The path's central track."""
        return self._path.path

    @property
    def affine(self):
        return self._path.affine

    @property
    def stream_data(self):
        return self._stream_data

    @property
    def origin(self):
        return self._origin

    @property
    def frame(self):
        return self._frame

    frame_fit = MetaAttribute()
    visit_order = MetaAttribute()
    som = MetaAttribute()
    kalman = MetaAttribute()

    #######################################################
    # Math on the Track

    def __call__(self, arc_length: TH.QuantityType) -> TH.CoordinateType:
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
        return self.path(arc_length)

    # /def

    #######################################################
    # misc

    def __repr__(self):
        """String representation."""
        s = super().__repr__()

        frame_name = self.frame.__class__.__name__
        rep_name = self.track.representation_type.__name__
        s = s.replace("StreamTrack", f"StreamTrack ({frame_name}|{rep_name})")

        s += "\n" + indent(repr(self._stream_data)[1:-1])

        return s

    # /def


# /class


##############################################################################
# END
