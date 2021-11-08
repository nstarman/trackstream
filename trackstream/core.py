# -*- coding: utf-8 -*-

"""Core Functions."""

__all__ = [
    "TrackStream",
    "StreamTrack",
]


##############################################################################
# IMPORTS

# STDLIB
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy import table
from astropy.utils.metadata import MetaAttribute, MetaData
from astropy.utils.misc import indent
from scipy.linalg import block_diag

# LOCAL
from . import _type_hints as TH
from trackstream.preprocess.rotated_frame import RotatedFrameFitter
from trackstream.preprocess.som import order_data, prepare_SOM, reorder_visits
from trackstream.process.kalman import KalmanFilter
from trackstream.process.utils import make_dts, make_F, make_H, make_Q, make_R
from trackstream.utils._framelike import resolve_framelike
from trackstream.utils.path import Path
from trackstream.utils.utils import intermix_arrays


##############################################################################
# CODE
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
        som=None,
        *,
        learning_rate: float = 0.1,
        sigma: float = 1.0,
        iterations: int = 10000,
        random_seed: T.Optional[int] = None,
        reorder: T.Optional[int] = None,
        progress: bool = False,
        nlattice: T.Optional[int] = None,
        **kwargs,
    ):
        """Reorder data by SOM.

        .. todo::

            - iterative training
            - reuse SOM

        Parameters
        ----------
        arm : SkyCoord
        som : object or None (optional)
            The self-organizing map. If None, will be constructed.
        learning_rate : float (optional, keyword-only)
        sigma : float (optional, keyword-only)
        iterations : int (optional, keyword-only)
        random_seed : int or None (optional, keyword-only)
        reorder : int or None (optional, keyword-only)
        progress : bool (optional, keyword-only)
            Whether to show progress bar.

        """
        # # TODO! iterative training

        # we order the data by longitude
        # and ensure it's oriented from the origin, outward
        # order_by_lon = np.argsort(arm.lon)
        # if arm.lon[order_by_lon[-1]] < 0:
        #     order_by_lon = np.flip(order_by_lon)

        rep = arm.represent_as(coord.SphericalRepresentation)
        data = rep._values.view("f8").reshape(-1, len(rep.components))
        data[:, :2] *= u.rad.to(u.deg)  # rad -> deg
        # data = data[order_by_lon]

        if som is None:
            som = prepare_SOM(
                data=data,
                learning_rate=learning_rate,
                sigma=sigma,
                iterations=iterations,
                random_seed=random_seed,
                progress=progress,
                nlattice=nlattice,
                **kwargs,
            )

        som.train(
            data,
            iterations,
            verbose=False,
            random_order=False,
            progress=progress,
        )

        # get the ordering by "vote" of the Prototypes
        visit_order = order_data(som, data)

        # Reorder
        if reorder is not None:
            visit_order = reorder_visits(rep, visit_order, start_ind=reorder)

        # then we need to change the visit order to
        # be applied to the original data, not ordered by lon
        # visit_order = order_by_lon[visit_order]

        # ----------------------------
        # TODO! transition matrix

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
            The starting point of the Kalman filter.

        Returns
        -------
        mean_path
        kalman_filter

        """
        arr = data.cartesian.xyz.T.value
        dts = make_dts(arr, dt0=0.5, N=6, axis=1, plot=False)

        # starting point
        if w0 is None:  # need to determine a good starting point
            # Instead of choosing the first point as the starting point,
            # since the stream is in its frame, instead choose the locus of
            # points near the origin.
            x = arr[:3].mean(axis=0)  # fist point
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
            frame, frame_fit = self._fit_rotated_frame(stream, **rotated_frame_fit_kw)

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
            visit_order, som = self._fit_SOM(arm1, som=som, **som_fit_kw)
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
        mean_path1, kalman_filter1 = self._run_kalman_filter(arm1, **kalman_fit_kw)
        # cache (even if None)
        self._cache["arm1_mean_path"] = mean_path1
        self._cache["arm1_kalman"] = kalman_filter1

        # Arm 2
        # -----
        if arm2 is not None:
            mean_path2, kalman_filter2 = self._run_kalman_filter(arm2, **kalman_fit_kw)
        else:
            mean_path2 = kalman_filter2 = None
        # cache (even if None)
        self._cache["arm2_mean_path"] = mean_path2
        self._cache["arm2_kalman"] = kalman_filter2

        # FIXME!
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
#         rep = self._data_rep(**{k: v(arc_length) for k, v in self._track.items()})
#         return self._data_frame.realize_frame(rep)
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
