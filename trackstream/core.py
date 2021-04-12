# -*- coding: utf-8 -*-

"""Core Functions."""

__all__ = [
    "Stream",
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
from astropy import table
from astropy.table import QTable, Table
from astropy.utils.decorators import lazyproperty
from astropy.utils.metadata import MetaAttribute, MetaData
from astropy.utils.misc import indent
from scipy.linalg import block_diag

# PROJECT-SPECIFIC
from . import type_hints as TH
from trackstream.preprocess.rotated_frame import RotatedFrameFitter
from trackstream.preprocess.som import apply_SOM
from trackstream.process.kalman import KalmanFilter
from trackstream.process.utils import make_dts, make_F, make_H, make_Q, make_R
from trackstream.utils._framelike import resolve_framelike
from trackstream.utils.path import Path
from trackstream.utils.utils import intermix_arrays

##############################################################################
# CODE
##############################################################################


class Stream:
    """A Stellar Stream.

    Parameters
    ----------
    data : `~astropy.table.Table`

    origin : `~astropy.coordinates.ICRS`
        The origin point of the rotated reference frame.

    data_err : `~astropy.table.QTable` (optional)
        The data_err must have (at least) column names
        ["x_err", "y_err", "z_err"]

    frame : `~astropy.coordinates.BaseCoordinateFrame` or None (optional, keyword-only)
        The stream frame. Locally linearizes the data.
        If not None, need to fit for the frame (default).

    """

    # -----------------------------------------------------

    def __init__(
        self,
        data: table.QTable,
        origin: TH.FrameType,
        data_err: T.Optional[table.Table] = None,
        *,
        frame: T.Optional[TH.CoordinateType] = None,
    ):
        super().__init__()

        self.origin: TH.SkyCoordType = coord.SkyCoord(origin, copy=False)
        self._system_frame: T.Optional[TH.FrameType] = frame

        self._cache = dict()  # TODO! improve

        # ----------
        # process the data

        # seed values set in functions
        self._original_data: TH.SkyCoordType = None

        # processed data -> QTable
        self.data = self._normalize_data(data)

    # /def

    @classmethod
    def from_arms(
        cls,
        arm1: table.Table,
        arm2: table.Table,
        origin: TH.FrameType,
        data_err: table.QTable = None,
        *,
        frame: T.Optional[TH.FrameType] = None,
    ) -> "Stream":
        """Data from arms.

        .. todo:: TODO!

        Parameters
        ----------
        arm1, arm2 : `~astropy.table.Table`
            The stream data.

        origin : `~astropy.coordinates.ICRS`
            The origin point of the rotated reference frame.

        data_err : `~astropy.table.QTable` (optional)
            The data_err must have (at least) column names
            ["x_err", "y_err", "z_err"]

        frame : frame-like or None (optional, keyword-only)
            The stream frame. Locally linearizes the data.
            If not None, need to fit for the frame (default).

        Returns
        -------
        stream : `~trackstream.core.Stream`

        """
        raise NotImplementedError("TODO!")  # TODO!

        self = super().__new__(cls)
        data = None

        # Step 1) Convert to Representation, storing the Frame type
        if isinstance(data, (coord.BaseCoordinateFrame, coord.SkyCoord)):
            self._data_frame = data.replicate_without_data()
            self._data_rep = data.representation_type
            self._data = data.data

        elif isinstance(data, table.Table):
            _data = coord.SkyCoord.guess_from_table(data)
            self._data_frame = _data.replicate_without_data()
            self._data_rep = _data.representation_type
            self._data = _data.data

        else:
            raise TypeError(f"`data` type {type(data)} is wrong.")

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
            data_err = table.QTable(np.zeros((len(data), 3)), names=err_cols)

        if isinstance(data_err, Table):
            data_err = data_err[err_cols]  # extract columns
        elif isinstance(data_err, coord.BaseCoordinateFrame):
            if not isinstance(data_err.data, coord.CartesianRepresentation):
                raise TypeError("data_err must be Cartesian.")
            rep = data_err.data
            data_err = table.QTable(
                dict(x_err=rep.x, y_err=rep.y, z_err=rep.z),
            )
        elif isinstance(data_err, coord.BaseRepresentation):
            if not isinstance(data_err, coord.CartesianRepresentation):
                raise TypeError("data_err must be Cartesian.")
            data_err = table.QTable(
                dict(x_err=rep.x, y_err=rep.y, z_err=rep.z),
            )
        else:
            raise TypeError(f"`data_err` type <{type(data_err)}> is wrong.")

    # /def

    # -----------------------------------------------------

    @property
    def system_frame(self) -> coord.BaseCoordinateFrame:
        """A system-centric frame.

        Determined from the argument ``frame`` at initialization.
        If None (default) and the method ``fit`` has been called,
        than a system frame has been found and cached.

        """
        frame: coord.BaseCoordinateFrame = (
            self._system_frame
            if self._system_frame is not None
            else self._cache.get("frame", None)
        )
        return frame

    # /def

    @property
    def frame(self) -> coord.BaseCoordinateFrame:
        """Alias for ``system_frame``."""
        return self.system_frame

    # /def

    @lazyproperty
    def number_of_tails(self) -> int:
        """Number of tidal tails.

        Returns
        -------
        number_of_tails : int
            There can only be 1, or 2 tidal tails.

        """
        return 2 if (self.has_arm1 and self.has_arm2) else 1

    # /def

    @lazyproperty
    def has_arm1(self) -> bool:
        # get flags from data
        flags: table.Column = np.unique(self.data["tail"])
        has_arm: bool = "arm_1" in flags
        return has_arm

    # /def

    @lazyproperty
    def has_arm2(self) -> bool:
        # get flags from data
        flags: table.Column = np.unique(self.data["tail"])
        has_arm: bool = "arm_2" in flags
        return has_arm

    # /def

    @lazyproperty
    def coords(self) -> coord.SkyCoord:
        """Coordinates."""
        frame = (
            self.system_frame
            if self.system_frame is not None
            else self.data_frame
        )
        return self.data_coords.transform_to(frame)

    # /def

    @property
    def coords_arm1(self) -> coord.SkyCoord:
        """The coordinates of the first arm.

        If the data isn't labelled, everything is assumed to be in this arm.

        """
        arm1: coord.SkyCoord
        if self.number_of_tails == 2:
            arm1 = self.coords[self.data["tail"] == "arm_1"]
        elif not self.has_arm2:
            arm1 = self.coords

        return arm1

    # /def

    @property
    def coords_arm2(self) -> T.Optional[coord.SkyCoord]:
        arm2: T.Optional[coord.SkyCoord] = None
        if self.has_arm2:
            arm2: coord.SkyCoord = self.coords[self.data["tail"] == "arm_2"]

        return arm2

    # /def

    # ===============================================================

    @property
    def original_data(self) -> TH.FrameType:
        """Original data as passed."""
        # TODO! don't make new one each time
        return self._data_frame.realize_frame(self._data)

    # /def

    @property
    def data_coords(self) -> TH.SkyCoordType:
        """Get ``coord`` from data table."""
        return self.data["coord"]

    # /def

    @property
    def data_frame(self) -> TH.FrameType:
        """The frame of the data."""
        return self.data["coord"].frame.replicate_without_data()

    # /def

    @property
    def data_representation_type(self) -> TH.RepresentationType:
        """The representation type of the data."""
        return self.data["coord"].representation_type

    # /def

    # -----------------------------------------------------
    # Data normalization

    def _normalize_data(self, original: table.Table) -> table.QTable:
        """Normalize data table.

        Just calls other functions.

        .. todo::

            allow argument specifying column names.

        Parameters
        ----------
        original : :class:`~astropy.table.Table`

        Returns
        -------
        data : :class:`~astropy.table.QTable`

        """
        data = QTable()

        # 1) data probability
        self._normalize_data_probability(original, data, default_weight=1)

        # 2) coordinates. `data` modded in-place
        self._normalize_data_coordinates(original, data)

        # --------------
        # Metadata

        # TODO? selective, or just copy over?
        data.meta = original.meta.copy()  # TODO? deepcopy?

        return data

    # /def

    def _normalize_data_probability(
        self,
        original: table.Table,
        data: TH.QTableType,
        default_weight: T.Union[float, TH.QuantityType] = 1.0,
    ) -> None:
        """Data probability. Units of percent. Default is 100%.

        Parameters
        ----------
        original : |Table|
            The original data.
        data : |QTable|
            The normalized data.
        default_weight : float
            The default membership probability.
            If float, then range 0-1 maps to 0-100%.
            If has unit of percent, then unchanged

        """
        colns = [n.lower() for n in original.colnames]

        if "pmemb" in colns:
            Pmemb = original[original.colnames[colns.index("pmemb")]]
        else:
            Pmemb = np.ones(len(original)) * default_weight  # non-scalar

        data["Pmemb"] = u.Quantity(Pmemb).to(u.percent)  # in %

    # /def

    def _normalize_data_coordinates(
        self,
        original: table.Table,
        data: TH.QTableType,
    ):
        """Parse the data table.

        - the frame is stored in ``_data_frame``
        - the representation is stored in ``_data_rep``
        - the original data representation  is in ``_data``

        Parameters
        ----------
        data : |Table|
            The stream data.

        data_err : |Table| or |Frame| or instance
            It must be in Cartesian coordinates!

        Raises
        ------
        TypeError
            if `data` is not |Table| or |Frame|

        """
        # TODO!!! better

        data["tail"] = original["tail"]

        # ----------
        # 1) the data

        # First look for a column "coord"
        if "coord" in original.colnames:
            sc = osc = coord.SkyCoord(original["coord"], copy=False)
        else:
            sc = osc = coord.SkyCoord.guess_from_table(original)

        self._original_data = osc

        # Convert frame and representation type
        frame = (
            self.system_frame
            if self.system_frame is not None
            else osc.frame.replicate_without_data()
        )
        sc = sc.transform_to(frame)
        sc.representation_type = coord.CartesianRepresentation

        # it's now clean and can be added
        data["coord"] = sc

        # also want to store the components, for plotting
        compnames = sc.representation_component_names.keys()
        for n in compnames:
            data[n] = getattr(sc, n)

        # ----------
        # 2) the error
        # TODO! want errors in frame of the data

        err_cols = ["x_err", "y_err", "z_err"]
        for n in err_cols:
            data[n] = original[n]  # transfer

        # ----------

        data = data.group_by("tail")
        data.add_index("tail")

    # /def


# /class


##############################################################################


class TrackStream:
    """Track a Stream in ICRS coordinates.

    When run, produces a StreamTrack.

    Parameters
    ----------
    SOM1 : `~trackstream.preprocess.SelfOrganizingMap` or None (optional, keyword-only)
    SOM2 : `~trackstream.preprocess.SelfOrganizingMap` or None (optional, keyword-only)

    Notes
    -----
    This method brings together a few different classes and techniques.
    More granular control can be achieved by using each piece separately.

    """

    def __init__(self, *, SOM=None, SOM2=None):
        self._cache: T.Dict[str, object] = {}

        # ----------
        # SOM

        self._arm1_SOM = SOM
        self._arm2_SOM = SOM2

        # self.visit_order = None

    #################################################################
    # Fit

    def _fit_rotated_frame(
        self,
        stream,
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
            data=stream.data_coords,
            origin=stream.origin,
            **kwargs,
        )

        fit = fitter.fit(rot0=rot0, bounds=bounds)

        return fit.frame, fit

    # /def

    # -------------------------------------------

    def _fit_SOM(
        self,
        arm,
        *,
        learning_rate: float = 2.0,
        sigma: float = 1.0,
        iterations: int = 10000,
        random_seed: T.Optional[int] = None,
        reorder: T.Optional[int] = None,
        progress: bool = False,
    ):
        """Reorder data by SOM.

        .. todo::

            - iterative training
            - reuse SOM

        Parameters
        ----------
        arm : SkyCoord

        learning_rate : float
        sigma : float
        iterations : int
        random_seed : int or None
        reorder : int or None
        progress : bool
            Whether to show progress bar.

        """
        # we order the data by longitude
        # and ensure it's oriented from the origin, outward
        order_by_lon = np.argsort(arm.lon)
        if arm.lon[order_by_lon[-1]] < 0:
            order_by_lon = np.flip(order_by_lon)

        # TODO! iterative training
        # TODO! use cached SOM, if have
        visit_order, som = apply_SOM(
            arm[order_by_lon],
            learning_rate=learning_rate,
            sigma=sigma,
            iterations=iterations,
            random_seed=random_seed,
            reorder=reorder,
            progress=progress,
        )

        # then we need to change the visit order to
        # be applied to the original data, not ordered by lon
        visit_order = order_by_lon[visit_order]

        return visit_order, som

    # /def

    def _run_kalman_filter(self, data: TH.SkyCoordType, w0=None):
        """Fit data with Kalman filter.
        
        .. todo::
        
            allow more user control

        Parameters
        ----------
        stream : Stream
        w0 : array or None (optional)

        Returns
        -------
        mean_path
        kalman_filter

        """
        # TODO! it doesn't need to be cartesian?
        # but not small angle approx, need correct distance function
        arr = data.cartesian.xyz.T.value
        dts = make_dts(arr, dt0=0.5, N=6, axis=1, plot=False)

        if w0 is None:
            x = arr[0]  # fist point
            v = [0, 0, 0]  # guess for "velocity"
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

        _, smooth_mean_path = kf.run(
            arr,
            dts,
            method="stepupdate",
            use_filterpy=None,
        )

        return smooth_mean_path, kf

    # /def

    # -------------------------------------------

    def fit(
        self,
        stream,
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
        # Fit Rotated Frame
        # this step applies to all arms. In fact, it will perform better if both
        # arms are present, limiting the influence of the tails on the frame
        # orientation.

        # 1) Already provided or in cache.
        #    Either way, don't need to repeat the process.
        frame = stream._system_frame  # can be None
        frame_fit = self._cache.get("frame_fit", None)

        # 2) Fit (& cache), if still None.
        #    This can be turned off using `fit_frame_if_needed`, but is probably
        #    more important than the following step of the SOM.
        if frame is None and fit_frame_if_needed:
            rotated_frame_fit_kw = rotated_frame_fit_kw or {}
            frame, frame_fit = self._fit_rotated_frame(
                stream, **rotated_frame_fit_kw
            )

        # 3) if it's still None, give up
        if frame is None:
            frame = stream.data_coord.frame.replicate_without_data()
            frame_fit = None

        # Cache the fit frame on the stream. This is used for transforming the
        # coordinates into the system frame (if that wasn't provided to the
        # stream on initialization).
        self._cache["frame"] = frame  # SkyOffsetICRS
        self._cache["frame_fit"] = frame_fit
        stream._cache["frame"] = frame

        # get arms, in frame
        # do this after caching b/c coords_arm1 can use the cache
        arm1 = stream.coords_arm1.transform_to(frame)
        arm2 = stream.coords_arm2.transform_to(frame)

        # -------------------
        # Self-Organizing Map
        # Unlike the previous step, we must do this for both arms.

        som = self._arm1_SOM
        visit_order = None

        # 1) try to get from cache
        if som is None:
            visit_order = self._cache.get("arm1_visit_order", None)
            som = self._cache.get("arm1_SOM", None)
        # 2) fit, if still None
        if visit_order is None and fit_SOM_if_needed:
            som_fit_kw = som_fit_kw or {}
            visit_order, som = self._fit_SOM(arm1, **som_fit_kw)
        # 3) if it's still None, give up
        if visit_order is None:
            visit_order = np.argsort(arm1.lon)
        # now rearrange the data
        arm1 = arm1[visit_order]

        # cache (even if None)
        self._cache["arm1_visit_order"] = visit_order
        self._cache["arm1_SOM"] = som

        # Arm 2
        # -----
        som = self._arm2_SOM
        visit_order = None

        if arm2 is not None:

            # 1) try to get from cache
            if som is None:
                visit_order = self._cache.get("arm2_visit_order", None)
                som = self._cache.get("arm2_SOM", None)
            # 2) fit, if still None
            if visit_order is None and fit_SOM_if_needed:
                som_fit_kw = som_fit_kw or {}
                visit_order, som = self._fit_SOM(arm2, **som_fit_kw)
            # 3) if it's still None, give up
            if visit_order is None:
                visit_order = np.argsort(arm2.lon)
            # now rearrange the data
            arm2 = arm2[visit_order]

        # /if

        # cache (even if None)
        self._cache["arm2_visit_order"] = visit_order
        self._cache["arm2_SOM"] = som

        # -------------------
        # Kalman Filter

        # Arm 1  (never None)
        # -----
        kalman_fit_kw = kalman_fit_kw or {}
        mean_path1, kalman_filter1 = self._run_kalman_filter(
            arm1, **kalman_fit_kw
        )
        # cache (even if None)
        self._cache["arm1_mean_path"] = mean_path1
        self._cache["arm1_kalman"] = kalman_filter1

        # Arm 2
        # -----
        if arm2 is not None:
            mean_path2, kalman_filter2 = self._run_kalman_filter(
                arm2, **kalman_fit_kw
            )
        else:
            mean_path2 = kalman_filter2 = None
        # cache (even if None)
        self._cache["arm2_mean_path"] = mean_path2
        self._cache["arm2_kalman"] = kalman_filter2

        return mean_path1, kalman_filter1, mean_path2, kalman_filter2

        # -------------------
        # Combine together into a single path
        # TODO!

        # mean_path = self.points.transform_to(self.frame)  # FIXME!

        # path = Path(
        #     path=mean_path,
        #     width=100 * u.pc,  # FIXME!
        #     affine=mean_path.spherical.lon,
        #     frame=self.frame,
        # )

        # # construct interpolation
        track = StreamTrack(
            mean_path1,  # TODO!
            stream_data=stream.data,
            origin=stream.origin,
            frame=frame,
            # extra
            frame_fit=frame_fit,
            visit_order=visit_order,
            som=dict(
                arm1=self._cache.get("arm1_SOM", None),
                arm2=self._cache.get("arm2_SOM", None),
            ),
            kalman=dict(arm1=kalman_filter1, arm2=kalman_filter2),  # TODO!
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


#####################################################################


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
        stream_data: T.Union[table.Table, TH.CoordinateType, None],
        origin: TH.CoordinateType,
        frame: TH.FrameLikeType,
        **metadata,
    ):
        super().__init__()

        # validation of types
        if not isinstance(path, Path):
            raise TypeError("`path` must be <Path>.")
        elif not isinstance(
            origin,
            (coord.SkyCoord, coord.BaseCoordinateFrame),
        ):
            raise TypeError(
                "`origin` must be <|SkyCoord|, |Frame|>.",
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
        |Frame|
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
