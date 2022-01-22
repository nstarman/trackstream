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
from astropy.table import Table
from astropy.utils.metadata import MetaAttribute, MetaData
from astropy.utils.misc import indent
from scipy import stats
from scipy.linalg import block_diag

# LOCAL
from . import _type_hints as TH
from .stream import Stream
from trackstream.preprocess.rotated_frame import FitResult, RotatedFrameFitter
from trackstream.preprocess.som import SelfOrganizingMap1D, order_data, reorder_visits
from trackstream.process.kalman import KalmanFilter
from trackstream.process.utils import make_dts, make_F, make_H, make_Q, make_R
from trackstream.utils import resolve_framelike
from trackstream.utils.misc import intermix_arrays
from trackstream.utils.path import Path, path_moments

##############################################################################
# CODE
##############################################################################


class TrackStream:
    """Track a Stream in ICRS coordinates.

    When run, produces a StreamTrack.

    Parameters
    ----------
    arm1SOM, arm2SOM : `~trackstream.preprocess.SelfOrganizingMap` or None (optional, keyword-only)
        Fiducial SOMs for stream arms 1 and 2, respectively.
    """

    def __init__(self, *, arm1SOM=None, arm2SOM=None):
        self._cache: T.Dict[str, object] = {}

        # SOM
        self._arm1_SOM = arm1SOM
        self._arm2_SOM = arm2SOM

    # ===============================================================
    # Fit

    def _fit_rotated_frame(
        self,
        stream: Stream,
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

        fitted = fitter.fit(rot0=rot0, bounds=bounds)

        return fitted.frame, fitted

    # -------------------------------------------

    def _fit_SOM(
        self,
        arm,
        som=None,
        *,
        learning_rate: float = 0.1,
        sigma: float = 1.0,
        iterations: int = 10_000,
        random_seed: T.Optional[int] = None,
        reorder: T.Optional[int] = None,
        progress: bool = False,
        nlattice: T.Optional[int] = None,
        **kwargs,
    ):
        """Reorder data by SOM.

        .. todo::

            - iterative training

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

        rep = arm.represent_as(coord.SphericalRepresentation)
        data = rep._values.view("f8").reshape(-1, len(rep.components))
        data[:, :2] *= u.rad.to(u.deg)  # rad -> deg

        if som is None:
            data_len, nfeature = data.shape
            if nlattice is None:
                nlattice = data_len // 10  # allows to be variable
            if nlattice == 0:
                raise ValueError

            som = SelfOrganizingMap1D(
                nlattice,
                nfeature,
                sigma=sigma,
                learning_rate=learning_rate,
                # decay_function=None,
                neighborhood_function="gaussian",
                activation_distance="euclidean",
                random_seed=random_seed,
            )

            # call method to initialize SOM weights
            weight_init_method = kwargs.get("weight_init_method", "binned_weights_init")
            getattr(som, weight_init_method)(data, **kwargs)

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

        # ----------------------------
        # TODO! transition matrix

        return visit_order, som

    def _fit_kalman_filter(self, data: coord.SkyCoord, w0=None):
        """Fit data with Kalman filter.

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

        kf = KalmanFilter(
            w0,
            P0,
            F0=make_F,
            Q0=make_Q,
            H0=H0,
            R0=R0,
            q_kw=dict(var=0.01, n_dims=3),  # TODO! as options
        )

        smooth_mean_path = kf.fit(
            arr,
            dts,
            method="stepupdate",
            use_filterpy=None,
        )

        return smooth_mean_path, kf, dts

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
        frame: T.Optional[BaseCoordinateFrame] = stream._system_frame
        frame_fit: T.Optional[FitResult] = self._cache.get("frame_fit", None)

        # 2) Fit (& cache), if still None.
        #    This can be turned off using `fit_frame_if_needed`, but is
        #    probably more important than the following step of the SOM.
        if frame is None and fit_frame_if_needed:
            kw: dict = rotated_frame_fit_kw or {}
            frame, frame_fit = self._fit_rotated_frame(stream, **kw)

        # 3) if it's still None, give up
        if frame is None:
            frame: BaseCoordinateFrame = stream.data_coord.frame.replicate_without_data()
            frame_fit = None

        # Cache the fit frame on the stream. This is used for transforming the
        # coordinates into the system frame (if that wasn't provided to the
        # stream on initialization).
        self._cache["frame"] = frame  # SkyOffsetICRS
        self._cache["frame_fit"] = frame_fit
        stream._cache["frame"] = frame

        # get arms, in frame
        # do this after caching b/c coords can use the cache
        arm1: coord.SkyCoord = stream.arm1.coords
        arm2: coord.SkyCoord = stream.arm2.coords

        # -------------------
        # Self-Organizing Map
        # Unlike the previous step, we must do this for both arms.

        # -----
        # Arm 1
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
        visit_order = np.array(visit_order, dtype=int)
        # the visit order can be backward so need to detect proximity to origin
        # TODO! more careful if closest point not end point. & adjust SOM!
        arm1ep = arm1[visit_order[[0, -1]]]  # end points
        if np.argmin(arm1ep.separation_3d(stream.origin)) == 1:
            visit_order = visit_order[::-1]
        arm1 = arm1[visit_order]

        # cache (even if None)
        self._cache["arm1_visit_order"] = visit_order
        self._cache["arm1_SOM"] = som

        # -----
        # Arm 2 (if not None)
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
            visit_order = np.array(visit_order, dtype=int)
            # now rearrange the data
            # the visit order can be backward so need to detect proximity to origin
            # TODO! more careful if closest point not end point. & adjust SOM!
            arm2ep = arm2[visit_order[[0, -1]]]  # end points
            if np.argmin(arm2ep.separation_3d(stream.origin)) == 1:
                visit_order = visit_order[::-1]
            arm2 = arm2[visit_order]

        # cache (even if None)
        self._cache["arm2_visit_order"] = visit_order
        self._cache["arm2_SOM"] = som

        # -------------------
        # Kalman Filter
        # both arms start at 0 displacement wrt themselves, but not each other.
        # e.g. the progenitor is cut out. To address this the start of affine
        # is offset by epsilon = min(1e-10, 1e-10 * dp2p[0])

        # Arm 1  (never None)
        # -----
        kalman_fit_kw = kalman_fit_kw or {}
        mean1, kf1, dts1 = self._fit_kalman_filter(arm1, **kalman_fit_kw)
        # cache
        self._cache["arm1_mean_path"] = mean1
        self._cache["arm1_kalman"] = kf1

        # TODO! make sure get the frame and units right
        r1 = coord.CartesianRepresentation(mean1.Xs[:, ::2].T, unit=u.kpc)
        c1 = frame.realize_frame(r1)  # (not interpolated)
        sp2p1 = c1[:-1].separation(c1[1:])  # point-2-point sep
        affine1 = np.concatenate(([min(1e-10 * sp2p1.unit, 1e-10 * sp2p1[0])], sp2p1.cumsum()))

        # covariance matrix. select only the phase-space positions
        # everything is Gaussian so there are no off-diagonal elements,
        # so the 1-sigma error is quite easy.
        cov = mean1.Ps[:, ::2, ::2]
        var = np.diagonal(cov, axis1=1, axis2=2)
        sigma1 = np.sqrt(np.sum(np.square(var), axis=-1)) * u.kpc

        # Arm 2
        # -----
        if arm2 is None:
            mean2 = kf2 = dts2 = None
        else:
            mean2, kf2, dts2 = self._fit_kalman_filter(arm2, **kalman_fit_kw)

            # TODO! make sure get the frame and units right
            r2 = coord.CartesianRepresentation(mean2.Xs[:, ::2].T, unit=u.kpc)
            c2 = frame.realize_frame(r2)  # (not interpolated)
            sp2p2 = c2[:-1].separation(c2[1:])  # point-2-point sep
            affine2 = np.concatenate(([min(1e-10 * sp2p2.unit, 1e-10 * sp2p2[0])], sp2p2.cumsum()))

            cov = mean2.Ps[:, ::2, ::2]
            var = np.diagonal(cov, axis1=1, axis2=2)
            sigma2 = np.sqrt(np.sum(np.square(var), axis=-1)) * u.kpc

        # cache (even if None)
        self._cache["arm2_mean_path"] = mean2
        self._cache["arm2_kalman"] = kf2

        # -------------------
        # Combine together into a single path
        # Need to reverse order of one arm to be indexed toward origin, not away

        if arm2 is None:
            affine, c, sigma = affine1, c1, sigma1
        else:
            affine = np.concatenate((-affine2[::-1], affine1))
            c = coord.concatenate((c2[::-1], c1))
            sigma = np.concatenate((sigma2[::-1], sigma1))

        path = Path(
            path=c,
            width=sigma,
            affine=affine,
            frame=frame,
        )

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


# /class


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
        stream_data: T.Union[Table, TH.CoordinateType, None],
        origin: TH.CoordinateType,
        # frame: T.Optional[TH.FrameLikeType] = None,
        **meta,
    ):
        # validation of types
        if not isinstance(path, Path):
            raise TypeError("`path` must be <Path>.")
        elif not isinstance(origin, (coord.SkyCoord, coord.BaseCoordinateFrame)):
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
        self, affine: T.Optional[u.Quantity] = None, angular: bool = False
    ) -> path_moments:
        """Get discrete points along interpolated stream track.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like or None, optional
            The affine interpolation parameter. If None (default), return
            path moments evaluated at all "tick" interpolation points.

        Returns
        -------
        `trackstream.utils.path.path_moments`
            Realized from the ``.path`` attribute.
        """
        return self.path(affine=affine, angular=angular)

    # def

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


# /class


##############################################################################
# END
