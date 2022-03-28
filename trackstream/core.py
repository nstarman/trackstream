# -*- coding: utf-8 -*-

"""Core Functions."""

__all__ = [
    "TrackStream",
    "StreamTrack",
]


##############################################################################
# IMPORTS

# STDLIB
from typing import Dict, Optional, Sequence, Union

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import BaseCoordinateFrame, CartesianRepresentation, SkyCoord
from astropy.table import Table
from astropy.utils.metadata import MetaAttribute, MetaData
from astropy.utils.misc import indent
from scipy.linalg import block_diag

# LOCAL
from ._type_hints import CoordinateType, FrameLikeType
from trackstream.kalman.core import FirstOrderNewtonianKalmanFilter, kalman_output
from trackstream.kalman.helper import make_F, make_H, make_Q, make_R
from trackstream.rotated_frame import FitResult, RotatedFrameFitter
from trackstream.som import SelfOrganizingMap1D
from trackstream.utils.misc import intermix_arrays
from trackstream.utils.path import Path, path_moments, concatenate_paths

##############################################################################
# CODE
##############################################################################


class TrackStream:
    """Track a Stream.

    When run, produces a StreamTrack.

    Parameters
    ----------
    arm1SOM, arm2SOM : `~trackstream.SelfOrganizingMap` or None (optional, keyword-only)
        Fiducial SOMs for stream arms 1 and 2, respectively.
    """

    def __init__(self, *, arm1SOM=None, arm2SOM=None):
        self._cache: Dict[str, object] = {}

        # SOM
        self._arm1_SOM = arm1SOM
        self._arm2_SOM = arm2SOM

    # ===============================================================
    # Fit

    def _fit_rotated_frame(
        self,
        stream: "Stream",
        rot0: Optional[u.Quantity] = 0 * u.deg,
        bounds: Optional[Sequence] = None,
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
        # Make and run fitter
        fitter = RotatedFrameFitter(
            data=stream.data_coords,
            origin=stream.origin,
            **kwargs,
        )
        fitted = fitter.fit(rot0=rot0, bounds=bounds)

        # Cache and return results
        self._cache["frame"] = fitted.frame  # SkyOffsetICRS
        self._cache["frame_fit"] = fitted

        return fitted.frame, fitted

    # -------------------------------------------

    def _fit_SOM(
        self,
        arm,
        som: Optional[SelfOrganizingMap1D] = None,
        origin: Optional[SkyCoord] = None,
        *,
        learning_rate: float = 0.1,
        sigma: float = 1.0,
        iterations: int = 10_000,
        random_seed: Optional[int] = None,
        # reorder: Optional[int] = None,
        progress: bool = False,
        nlattice: Optional[int] = None,
        **kwargs,
    ):
        """Reorder data by SOM.

        Parameters
        ----------
        arm : SkyCoord
        som : object or None, optional
            The self-organizing map. If None, will be constructed.
        learning_rate : float, optional keyword-only
        sigma : float, optional keyword-only
        iterations : int, optional keyword-only

        random_seed : int or None, optional keyword-only
        progress : bool, optional keyword-only
            Whether to show progress bar.

        Returns
        -------
        visit_order : array
        som : SelfOrganizingMap1D

        Other Parameters
        ----------------
        nlattice : int or None, optional, keyword-only
            Number of lattice points.
        """
        # The SOM
        if som is None:
            data_len = len(arm)
            nfeature = len(arm.representation_component_names)

            nlattice = data_len // 10 if nlattice is None else nlattice
            if nlattice == 0:
                raise ValueError

            som = SelfOrganizingMap1D(
                nlattice,
                nfeature,
                frame=arm.frame.replicate_without_data(),  # TODO! as args
                representation_type=arm.frame.representation_type,  # TODO! as args
                sigma=sigma,
                learning_rate=learning_rate,
                neighborhood_function="gaussian",
                activation_distance="euclidean",
                random_seed=random_seed,
            )

            # call method to initialize SOM weights
            som.binned_weights_init(arm, **kwargs)

        # get the ordering by "vote" of the Prototypes
        visit_order = som.fit_predict(
            arm,
            num_iteration=iterations,
            random_order=False,
            progress=progress,
            origin=origin,
        )

        # # Reorder
        # if reorder is not None:
        #     visit_order = reorder_visits(arm, visit_order, start_ind=reorder)

        # ----------------------------
        # TODO! optionally transition matrix, i.e. iterative training

        return visit_order, som

    def _fit_kalman_filter(
        self, data: SkyCoord, frame: BaseCoordinateFrame, **options
    ) -> Union[kalman_output, FirstOrderNewtonianKalmanFilter, np.ndarray]:
        """Fit data with Kalman filter.

        Parameters
        ----------
        stream : Stream
            In frame `frame`
        frame : BaseCoordinateFrame
        w0 : array or None, optional
            The starting point of the Kalman filter.

        Returns
        -------
        mean_path
        kalman_filter

        """
        # TODO! detect if want positions only, or also velocities
        arr = data.transform_to(frame).cartesian.xyz.T.value

        # starting point
        x0 = options.get("x0", None)
        if x0 is None:  # need to determine a good starting point
            # Instead of choosing the first point as the starting point,
            # since the stream is in its frame, instead choose the locus of
            # points near the origin.
            x0 = arr[:3].mean(axis=0)  # fist point

        # TODO! as options
        p = np.array([[0.0001, 0], [0, 1]])
        P0 = block_diag(p, p, p)

        R0 = make_R([0.05, 0.05, 0.003])[0]  # TODO! actual errors
        options["q_kw"] = dict(var=0.01, n_dims=3)  # TODO! overridable

        kf = FirstOrderNewtonianKalmanFilter(
            x0, P0, R0=R0, frame=frame, representation_type=CartesianRepresentation, **options  # TODO! from data
        )

        timesteps = kf.make_simple_timesteps(data, dt0=50 * u.pc, width=6, vmin=0.01 * u.pc)
        mean_path, path = kf.fit(
            arr,
            timesteps=timesteps,
            use_filterpy=None,
        )

        return mean_path, kf, path

    # -------------------------------------------

    def fit(
        self,
        stream,
        *,
        # frame: Optional[FrameLike] = None,
        tune_SOM: bool = True,
        rotated_frame_fit_kw: Optional[dict] = None,
        som_fit_kw: Optional[dict] = None,
        kalman_fit_kw: Optional[dict] = None,
    ):
        """Fit a data to the data.

        Parameters
        ----------
        tune_SOM : bool, optional keyword-only
        rotated_frame_fit_kw : dict or None, optional keyword-only
        som_fit_kw : dict or None, optional keyword-only
        kalman_fit_kw : dict or None, optional keyword-only

        Returns
        -------
        StreamTrack instance
            Also stores as ``.track``
        """
        # -------------------
        # Fit Rotated Frame
        # this step applies to all arms. In fact, it will perform better if
        # both arms are present, limiting the influence of the tails on the
        # frame orientation.

        frame: Optional[BaseCoordinateFrame]
        frame_fit: Optional[FitResult]

        # 1) Already provided or in cache.
        #    Either way, don't need to repeat the process.
        frame = stream.system_frame  # NOT .system_frame ?
        frame_fit = self._cache.get("frame_fit", None)

        # 2) Fit (& cache), if still None.
        if frame is None:
            kw: dict = rotated_frame_fit_kw or {}
            frame, frame_fit = self._fit_rotated_frame(stream, **kw)

        # Cache the fit frame on the stream. This is used for transforming the
        # coordinates into the system frame (if that wasn't provided to the
        # stream on initialization).
        # TODO! don't override cache if it's the exact same
        self._cache["frame"] = frame  # SkyOffsetICRS
        self._cache["frame_fit"] = frame_fit
        stream._cache["frame"] = frame

        # get arms, in frame
        # done after caching b/c coords can use the cache
        arm1: SkyCoord = stream.arm1.coords
        arm2: SkyCoord = stream.arm2.coords

        # -------------------
        # Self-Organizing Map
        # Unlike the previous step, we must do this for both arms.

        # -----
        # Arm 1
        som1 = self._arm1_SOM
        visit_order1 = None

        # 1) try to get from cache (e.g. first time fitting)
        if som1 is None:
            print("SOM is None")
            visit_order1 = self._cache.get("arm1_visit_order", None)
            som1 = self._cache.get("arm1_SOM", None)
        # 2) fit, if still None or force continued fit
        if visit_order1 is None:
            print("SOM is still None. kwargs are:", som_fit_kw)
            visit_order1, som1 = self._fit_SOM(
                arm1, som=som1, origin=stream.origin, **(som_fit_kw or {})
            )
        elif tune_SOM:
            print("SOM is tuning. kwargs are:", som_fit_kw)
            # arm1 = arm1[visit_order1]  # TODO!
            visit_order1, som1 = self._fit_SOM(
                arm1, som=som1, origin=stream.origin, **(som_fit_kw or {})
            )
        # 3) if it's still None, give up
        if visit_order1 is None:
            raise ValueError()

        # cache (even if None)
        self._cache["arm1_visit_order"] = visit_order1
        self._cache["arm1_SOM"] = som1

        arm1 = arm1[visit_order1]  # re-order

        # -----
        # Arm 2 (if not None)
        som2 = self._arm2_SOM
        visit_order2 = None

        if stream.arm2.has_data:

            # 1) try to get from cache (e.g. first time fitting)
            if som2 is None:
                print("SOM is None")
                visit_order2 = self._cache.get("arm2_visit_order2", None)
                som2 = self._cache.get("arm2_SOM", None)
            # 2) fit, if still None or force continued fit
            if visit_order2 is None:
                print("SOM is still None. kwargs are:", som_fit_kw)
                visit_order2, som2 = self._fit_SOM(
                    arm2, som=som2, origin=stream.origin, **(som_fit_kw or {})
                )
            elif tune_SOM:
                print("SOM is tuning. kwargs are:", som_fit_kw)
                # arm2 = arm2[visit_order2]  # TODO!
                visit_order2, som2 = self._fit_SOM(
                    arm2, som=som2, origin=stream.origin, **(som_fit_kw or {})
                )
            # 3) if it's still None, give up
            if visit_order2 is None:
                raise ValueError

        # cache (even if None)
        self._cache["arm2_visit_order"] = visit_order2
        self._cache["arm2_SOM"] = som2
        
        arm2 = arm2[visit_order2]  # re-order

        # -------------------
        # Kalman Filter
        # both arms start at 0 displacement wrt themselves, but not each other.
        # e.g. the progenitor is cut out. To address this the start of affine
        # is offset by epsilon = min(1e-10, 1e-10 * dp2p[0])

        # Arm 1  (never None)
        # -----
        kalman_fit_kw = kalman_fit_kw or {}
        mean1, kf1, path1 = self._fit_kalman_filter(arm1, frame=frame, **kalman_fit_kw)
        # cache
        self._cache["arm1_mean_path"] = mean1
        self._cache["arm1_kalman"] = kf1

        # Arm 2
        # -----
        if not stream.arm2.has_data:
            mean2 = kf2 = None
        else:
            mean2, kf2, path2 = self._fit_kalman_filter(arm2, frame=frame, **kalman_fit_kw)

        # cache (even if None)
        self._cache["arm2_mean_path"] = mean2
        self._cache["arm2_kalman"] = kf2

        # -------------------
        # Combine together into a single path
        # Need to reverse order of one arm to be indexed toward origin, not away

        path = concatenate_paths((path2, path1))  # TODO! which negative?

        # construct interpolation
        track = StreamTrack(
            path,
            stream_data=stream.data,
            origin=stream.origin,
            # frame=frame,
            # metadata
            frame_fit=frame_fit,
            # visit_order=visit_order,  # TODO! not combined
            som=dict(
                arm1=self._cache.get("arm1_SOM", None),  # TODO! fix ordering
                arm2=self._cache.get("arm2_SOM", None),
            ),
            kalman=dict(arm1=kf1, arm2=kf2),
        )
        return track

    # ===============================================================

    def predict(self, affine):
        """Predict from a fit.

        Returns
        -------
        StreamTrack instance

        """
        return self.track(affine)

    def fit_predict(self, stream, affine, **fit_kwargs):
        """Fit and Predict."""
        self.fit(stream, **fit_kwargs)
        return self.predict(affine)


##############################################################################


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

    frame_fit = MetaAttribute()
    visit_order = MetaAttribute()
    som = MetaAttribute()
    kalman = MetaAttribute()

    def __init__(
        self,
        path: Path,
        stream_data: Union[Table, CoordinateType, None],
        origin: CoordinateType,
        # frame: Optional[FrameLikeType] = None,
        **meta,
    ):
        # validation of types
        if not isinstance(path, Path):
            raise TypeError("`path` must be <Path>.")
        elif not isinstance(origin, (SkyCoord, BaseCoordinateFrame)):
            raise TypeError("`origin` must be <|SkyCoord|, |Frame|>.")

        # assign
        self._path: Path = path
        self._origin = origin
        # self._frame = resolve_framelike(frame)

        self._stream_data = stream_data

        # set the MetaAttribute(s)
        for attr in list(meta):
            descr = getattr(self.__class__, attr, None)
            if isinstance(descr, MetaAttribute):
                setattr(self, attr, meta.pop(attr))
        # and the meta
        self.meta.update(meta)

    @property
    def path(self):
        return self._path

    @property
    def track(self):
        """The path's central track."""
        return self._path.data

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
        return self._path.frame

    #######################################################
    # Math on the Track

    def __call__(
        self,
        affine: Optional[u.Quantity] = None,
        *,
        angular: bool = False,
    ) -> path_moments:
        """Get discrete points along interpolated stream track.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like or None, optional
            The affine interpolation parameter. If None (default), return
            path moments evaluated at all "tick" interpolation points.
        angular : bool, optional keyword-only
            Whether to compute on-sky or real-space.

        Returns
        -------
        `trackstream.utils.path.path_moments`
            Realized from the ``.path`` attribute.
        """
        return self.path(affine=affine, angular=angular)

    def probability(
        self,
        point: SkyCoord,
        background_model=None,
        *,
        angular: bool = False,
        affine: Optional[u.Quantity] = None,
    ):
        """Probability point is part of the stream.

        .. todo:: angular probability

        """
        # # Background probability
        # Pb = background_model(point) if background_model is not None else 0.0
        #
        # #
        # angular = False  # TODO: angular probability
        # afn = self._path.closest_affine_to_point(point, angular=False, affine=affine)
        # pt_w = getattr(self._path, "width_angular" if angular else "width")(afn)
        # sep = getattr(self._path, "separation" if angular else "separation_3d")(
        #     point,
        #     interpolate=False,
        #     affine=afn,
        # )

        # stats.norm.pdf(ps.separation_3d(point))  # FIXME! dimensionality
        raise NotImplementedError("TODO!")

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


##############################################################################
# END
