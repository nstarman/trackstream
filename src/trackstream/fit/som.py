"""Self-Organizing Maps.

References
----------
.. [MiniSom] Giuseppe Vettigli. MiniSom: minimalistic and NumPy-based
    implementation of the Self Organizing Map.
.. [frankenz] Josh Speagle. Frankenz: a photometric redshift monstrosity.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import warnings
from abc import abstractmethod
from copy import deepcopy
from math import pi
from typing import TYPE_CHECKING, Any, TypeVar, Union, cast

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import (
    Angle,
    BaseCoordinateFrame,
    BaseRepresentation,
    CartesianRepresentation,
    SkyCoord,
    UnitSphericalRepresentation,
)
from astropy.coordinates.angle_utilities import angular_separation
from astropy.units import Quantity, StructuredUnit
from attrs import cmp_using  # type: ignore
from attrs import NOTHING, Attribute, define, field
from numpy import arccos, cos, diff, exp, mean, ndarray, nonzero, power, subtract
from numpy.lib.recfunctions import structured_to_unstructured
from numpy.linalg import norm
from numpy.random import Generator, default_rng
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial import distance_matrix
from scipy.stats import binned_statistic

# LOCAL
from trackstream._typing import CoordinateType
from trackstream.base import FramedBase
from trackstream.utils.coord_utils import offset_by, position_angle
from trackstream.utils.pbar import get_progress_bar

if TYPE_CHECKING:
    # LOCAL
    from trackstream.stream import StreamArm


__all__ = ["CartesianSelfOrganizingMap1D", "UnitSphereSelfOrganizingMap1D"]

__credits__ = "MiniSom"

##############################################################################
# PARAMETERS

warnings.filterwarnings("ignore", message="sigma is too high for the dimension of the map")

DataType = Union[CoordinateType, BaseRepresentation]

NDT = TypeVar("NDT", bound=ndarray)


##############################################################################
# CODE
##############################################################################


def _respace_bins_from_left(bins: NDT, maxsep: ndarray, onsky: bool, eps: float | np.floating) -> NDT:
    """Respace bins to have a maximum separation.

    Bins are respaced from the left-most bin up to the penultimate bin. The
    respaced bins are iteratively processed until all bins (except the last) are
    no more than ``maxsep`` separated.

    Parameters
    ----------
    bins : (N,) ndarray
        The bins to make sure are correctly spaced. Must be sorted.
    maxsep : scalar ndarray
        Maximum separation between bins.
    onsky : bool
        Whether to respace with Cartesian or angular spacing.
    eps : float
        The small adjustment subtracted from maxsep when respacing bins so that
        bins are actually a little closer together. This prevents computer
        precision issues from making bins too far apart.

    Returns
    -------
    (N,) ndarray
        Better-spaced bins. Bins are modified in-place.
    """
    # Initial state
    diffs = arccos(cos(diff(bins[:-1]))) if onsky else diff(bins[:-1])
    (seps,) = nonzero(diffs > maxsep)

    i = 0  # cap at 10k iterations
    while any(seps) and i < 10_000:

        # Move the bins by the separation
        bins[seps + 1] = bins[seps] + maxsep * (1 - eps)

        diffs = arccos(cos(diff(bins[:-1]))) if onsky else diff(bins[:-1])
        (seps,) = nonzero(diffs > maxsep)

        i += 1

    return bins


def _respace_bins(bins: NDT, maxsep: ndarray, onsky: bool, eps: float | np.floating) -> NDT:
    """Respace bins to have a maximum separation.

    Parameters
    ----------
    bins : (N,) ndarray
        The bins, which will be respaced.
    maxsep : scalar ndarray
        Maximum separation between bins
    onsky : bool
        Whether to respace with Cartesian or angular spacing.
    eps : float
        The small adjustment subtracted from maxsep when respacing bins so that
        bins are actually a little closer together. This prevents computer
        precision issues from making bins too far apart.

    Returns
    -------
    bins : (N,) ndarray
        Better-spaced bins. Bins are modified in-place.
    """
    # Check the separations
    diffs = arccos(cos(diff(bins))) if onsky else diff(bins)
    (seps,) = nonzero(diffs > maxsep)

    i = 0  # cap at 10k iterations
    while any(seps) and i < 50:

        # Adjust from the left, then adjust from the right
        bins = _respace_bins_from_left(bins, maxsep=maxsep, onsky=onsky, eps=eps)
        bins[::-1] = -_respace_bins_from_left(-bins[::-1], maxsep=maxsep, onsky=onsky, eps=eps)

        # Check the separations
        diffs = arccos(cos(diff(bins))) if onsky else diff(bins)
        (seps,) = nonzero(diffs > maxsep)

        i += 1

    return bins


# ===================================================================


def _decay_function(learning_rate: float, iteration: int, max_iter: float) -> float:
    """Decay function of the learning process.

    Parameters
    ----------
    learning_rate : float
        current learning rate.
    iteration : int
        current iteration.
    max_iter : int
        maximum number of iterations for the training.

    Returns
    -------
    float
    """
    return learning_rate / (1.0 + (2.0 * iteration / max_iter))


##############################################################################


@define(frozen=True, kw_only=True)
class SelfOrganizingMap1DBase(FramedBase):
    """Initializes a Self-Organizing Map.

    Inspired by the design of [MiniSom]_

    Parameters
    ----------
    nlattice : int
        Number of lattice points (prototypes) in the 1D SOM.

    sigma : float, optional (default=1.0)
        Spread of the neighborhood function, needs to be adequate to the
        dimensions of the map. (at the iteration t we have sigma(t) = sigma / (1
        + t/T) where T is #num_iteration/2)
    learning_rate : initial learning rate
        (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T)
        where T is #num_iteration/2)
    rng : int, optional keyword-only (default=None)
        Random seed to use.

    prototypes : ndarray, optional
        The prototype vectors

    units : `~astropy.units.StructuredUnit`, keyword-only
        The units.

    frame : `astropy.coordinates.BaseCoordinateFrame`
        The frame in which to build the SOM. Data is transformed into this frame
        before the SOM is fit.
    frame_representation_type : |BaseRepresentation| or None, optional keyword-only
        The representation type in which to return coordinates.
        The representation type in which the SOM is fit is either
        `astropy.coordinates.CartesianRepresentation` if `onsky` is `False`
        or `astropy.coordinates.UnitSphericalRepresentation` if `onsky` is `True`.

    Notes
    -----
    neighborhood_function : 'gaussian'
        Function that weights the neighborhood of a position in the map.

    activation_distance : 'euclidean'
        Distance used to activate the map.

    References
    ----------
    .. [MiniSom] Giuseppe Vettigli. MiniSom: minimalistic and NumPy-based
        implementation of the Self Organizing Map.
    .. [frankenz] Josh Speagle. Frankenz: a photometric redshift monstrosity.
    """

    nlattice: int = field()
    sigma: float = field(default=0.1, converter=float)
    learning_rate: float = field(default=0.3, converter=float)
    rng: Generator = field(default=None, converter=default_rng)

    units: u.StructuredUnit = field(kw_only=True)
    _yy: ndarray = field(init=False, repr=False, eq=False)
    prototypes: ndarray = field(eq=cmp_using(eq=np.array_equal))
    _init_prototypes = field(init=False, repr=False, eq=cmp_using(eq=np.array_equal))

    @prototypes.default  # type: ignore
    def _prototypes_factory(self) -> ndarray:
        prototypes = 2 * self.rng.random((self.nlattice, self.nfeature)) - 1
        return prototypes

    @_init_prototypes.default  # type: ignore
    def _init_prototypes_factory(self) -> ndarray:
        ps = deepcopy(self.prototypes)
        ps.flags.writeable = False
        return ps

    @_yy.default  # type: ignore
    def _yy_factory(self) -> ndarray:
        return np.arange(self.nlattice, dtype=float)

    @sigma.validator  # type: ignore
    def _sigma_validator(self, _: Attribute, value: float) -> None:
        if value >= 1 or value >= self.nlattice:
            warnings.warn("sigma is too high for the dimension of the map")

    @units.validator  # type: ignore
    def _units_validator(self, _: Attribute, value: u.StructuredUnit) -> None:
        required = tuple(self.internal_representation_type.attr_classes.keys())  # type: ignore
        if not value.keys() == required:
            raise ValueError

    @classmethod
    def from_stream(
        cls,
        arm: StreamArm,
        /,
        *,
        nlattice: int | None = None,
        sigma: float = 0.1,
        learning_rate: float = 0.3,
        rng: Generator | int | None = None,
    ):
        data = arm.coords  # unordered, in system_frame

        frame = data.frame.replicate_without_data()
        representation_type = data.frame.representation_type

        # Determine number of lattice points, within [5, 100]
        if nlattice is None:
            nlattice = min(max(len(data) // 50, 5), 20)

        rep = data.transform_to(frame).represent_as(cls.internal_representation_type)
        units: dict[str, u.UnitBase] = rep._units
        su = StructuredUnit(tuple(units.values()), names=tuple(units.keys()))

        # Make SOM
        som = cls(
            nlattice=nlattice,
            frame=frame,
            frame_representation_type=representation_type,
            frame_differential_type=None,
            sigma=sigma,
            learning_rate=learning_rate,
            rng=rng,  # type: ignore
            prototypes=NOTHING,  # type: ignore
            units=su,
        )

        return som

    # ========================================================

    @classmethod
    @property
    @abstractmethod
    def onsky(cls) -> bool:
        """Whether to fit on-sky or 3d."""
        raise NotImplementedError

    @classmethod
    @property
    @abstractmethod
    def internal_representation_type(cls) -> type[BaseRepresentation]:
        """Representation type."""

    # -------------------------------------------

    @property
    @abstractmethod
    def nfeature(self) -> int:
        """Number of features."""

    @property
    def prototypes_crd(self) -> BaseCoordinateFrame:
        "Read-only view of prototypes vectors."
        p = self.prototypes.view()
        p.flags.writeable = False

        return self._v_to_crd(p)

    @property
    def init_prototypes_crd(self) -> BaseCoordinateFrame:
        "Read-only view of prototypes vectors."
        p = self._init_prototypes.view()
        p.flags.writeable = False

        return self._v_to_crd(p)

    # ===============================================================

    def _crd_to_q(self, crd: CoordinateType) -> Quantity:
        """Coordinate to structured Quantity.

        Parameters
        ----------
        crd : |Frame| or |SkyCoord|, positional-only

        Returns
        -------
        Quantity
        """
        rep = crd.transform_to(self.frame).represent_as(self.internal_representation_type)
        units: dict[str, u.UnitBase] = rep._units

        # structured quantity
        su = StructuredUnit(tuple(units.values()), names=tuple(units.keys()))
        q: Quantity = rep._values << su

        return q.to(self.units)

    def _crd_to_v(self, crd: CoordinateType | ndarray) -> ndarray:
        """Coordinate to unstructured array."""
        if isinstance(crd, ndarray):
            return crd

        q = self._crd_to_q(crd)
        v: ndarray = structured_to_unstructured(q.to_value(self.units))
        return v

    def _v_to_crd(self, arr: ndarray, /) -> BaseCoordinateFrame:
        data = {n: (arr[:, i] << unit) for i, (n, unit) in enumerate(self.units.items())}
        rep = self.internal_representation_type(**data)
        crd = self.frame.realize_frame(rep)
        crd.representation_type = self.frame_representation_type
        return crd

    # ===============================================================

    # ---------------------------------------------------------------
    # prototypes

    @abstractmethod
    def make_prototypes_binned(
        self, data: SkyCoord, byphi: bool = False, maxsep: Quantity | None = None, **_: Any
    ) -> None:
        pass

    # ---------------------------------------------------------------
    # fitting

    @abstractmethod
    def _activation_distance(self, x: ndarray, w: ndarray) -> ndarray:
        pass

    @abstractmethod
    def _update(self, x: ndarray, t: int, max_iteration: int) -> None:
        pass

    def fit(
        self,
        data: SkyCoord,
        num_iteration: int = int(1e5),
        random_order: bool = False,
        progress: bool = False,
    ) -> None:
        """Trains the SOM.

        Parameters
        ----------
        data : SkyCoord

        num_iteration : int
            Maximum number of iterations (one iteration per sample).
            Must be greater than the length of the data.
        random_order : bool (default=False)
            If True, samples are picked in random order.
            Otherwise the samples are picked sequentially.

        progress : bool (default=False)
            If True, show a progress bar
        """
        # Number of cycles through the data
        iterations = np.arange(num_iteration) % len(data)
        # Optionally randomize the cycles
        if random_order:
            self.rng.shuffle(iterations)

        # Get the in internal unitless representation
        X = self._crd_to_v(data)  # (D, 2 or 3)

        # Fit the data by sequential update
        with get_progress_bar(progress, len(iterations)) as pbar:
            for t, iteration in enumerate(iterations):
                pbar.update(1)

                self._update(X[iteration], t, num_iteration)

    def _neighborhood(self, c: int, sigma: float) -> ndarray:
        """Returns a Gaussian centered in c.

        This is in the lattice space, so Cartesian vs UnitSpherical does not
        matter.
        """
        d = 2 * pi * sigma**2
        ay: ndarray = exp(-power(self._yy - self._yy.T[c], 2) / d).T
        return ay  # the external product gives a matrix

    def _best_matching_unit_index(self, x: ndarray, /) -> int:
        """Computes the coordinates of the best prototype for the sample.

        Parameters
        ----------
        x : ndarray, positional-only

        Returns
        -------
        int
            The index of the best-matching prototype.
        """
        activation_map = self._activation_distance(x, self.prototypes)
        ibmu = int(activation_map.argmin())
        return ibmu

    # ---------------------------------------------------------------
    # Predicting structure
    # steps:
    #

    def _get_info_for_projection(
        self,
        data: ndarray,
    ) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        """Get orthogonal distance matrix for each point to each segment & node.

        Parameters
        ----------
        data : (N, D) ndarray
            The data. Rows are points, columns are features.

        Returns
        -------
        (N, PSN) ndarray
            Where P is the number of prototypes and connecting segments.
        """
        data_len, nfeature = data.shape

        # vector from one point to next  (nlattice-1, nfeature)
        lattice_points = self.prototypes
        p1 = lattice_points[:-1, :]
        p2 = lattice_points[1:, :]
        # vector from one point to next  (nlattice-1, nfeature)
        viip1 = np.subtract(p2, p1)
        # square distance from one point to next  (nlattice-1, nfeature)
        liip1 = np.sum(np.square(viip1), axis=1)

        # data - point_i  (D, nlattice-1, nfeature)
        # for each slice in D,
        dmi = np.subtract(data[:, None, :], p1[None, :, :])  # d-p1

        # The line extending the segment is parameterized as p1 + t (p2 - p1).
        # The projection falls where t = [(p3-p1) . (p2-p1)] / |p2-p1|^2
        # tM is the matrix of "t"'s.
        tM = np.sum((dmi * viip1[None, :, :]), axis=-1) / liip1

        projected_points = p1[None, :, :] + tM[:, :, None] * viip1[None, :, :]

        # add in the nodes and find all the distances
        # the correct "place" to put the data point is within a
        # projection, unless it outside (by an endpoint)
        # or inside, but on the convex side of a segment junction
        all_points = np.empty((data_len, 2 * self.nlattice - 1, nfeature), dtype=float)
        all_points[:, 1::2, :] = projected_points
        all_points[:, 0::2, :] = lattice_points
        distances = norm(np.subtract(data[:, None, :], all_points), axis=-1)

        # detect whether it is in the segment
        # nodes are considered in the segment
        not_in_projection = np.zeros(all_points.shape[:-1], dtype=bool)
        not_in_projection[:, 1::2] = np.logical_or(tM <= 0, 1 <= tM)

        # make distances for not-in-segment infinity
        distances[not_in_projection] = np.inf

        return viip1, tM, all_points, distances

    def _get_projected_point(self, data: ndarray, all_points: ndarray, distances: ndarray, /) -> SkyCoord:
        """Project data onto SOM.

        Parameters
        ----------
        data : (N, D) ndarray
            The data to project

        Returns
        -------
        SkyCoord
        """
        ind_best_distance = np.argmin(distances, axis=1)
        projpnts = all_points[(np.arange(len(distances))), ind_best_distance, :]

        return SkyCoord(self._v_to_crd(projpnts), copy=False)

    def _order_along_projection(self, data: ndarray, /, viip1: ndarray, tM: ndarray, distances: ndarray) -> ndarray:

        # find the index of the best distance (including nodes)
        ind_best_distance = np.argmin(distances, axis=1)

        ordering = np.zeros(len(data), dtype=int) - 1

        counter = 0  # count through edge/node groups
        for i in np.unique(ind_best_distance):
            # for i in (1, ):
            # get the data rows for which the best distance is the i'th node/segment
            rowi = np.where(ind_best_distance == i)[0]
            numrows = len(rowi)

            # move edge points to corresponding segment
            if i == 0:
                i = 1
            elif i == 2 * (self.nlattice - 1):
                i = self.nlattice - 1

            # odds (remainder 1) are segments
            if bool(i % 2):
                ts = tM[rowi, i // 2]
                rowsorter = np.argsort(ts)

            # evens are by nodes
            else:  # TODO! find and fix the potential ordering mistake
                phi1 = np.arctan2(*viip1[i // 2 - 1, :2])
                phim2 = np.arctan2(*-viip1[i // 2, :2])
                phi = np.arctan2(*data[rowi, :2].T)

                # detect if in branch cut territory
                if (np.pi / 2 <= phi1) & (-np.pi <= phim2) & (phim2 <= -np.pi / 2):
                    phi1 -= 2 * np.pi
                    phi -= 2 * np.pi

                rowsorter = np.argsort(phi) if phim2 < phi1 else np.argsort(-phi)

            slc = slice(counter, counter + numrows)
            ordering[slc] = rowi[rowsorter]
            counter += numrows

        ordering = np.array(ordering, dtype=int)

        return ordering

    @abstractmethod
    def predict(self, crd: SkyCoord, origin: SkyCoord | None = None) -> tuple[SkyCoord, ndarray]:
        pass

    def fit_predict(
        self,
        data: SkyCoord,
        num_iteration: int = int(1e5),
        random_order: bool = False,
        progress: bool = False,
        origin: SkyCoord | None = None,
    ) -> tuple[SkyCoord, ndarray]:
        """Fit then predict.

        Returns
        -------
        projdata : SkyCoord
            Ordered.
        order : ndarray
            Array to order ``data``.

        See Also
        --------
        trackstream.SelfOrganizingMap1D.fit
        trackstream.SelfOrganizingMap1D.predict
        """
        self.fit(data, num_iteration=num_iteration, random_order=random_order, progress=progress)
        projdata, order = self.predict(data, origin=origin)
        return projdata, order


# ----------------------------------------------------------------------------


@define(frozen=True, kw_only=True)
class CartesianSelfOrganizingMap1D(SelfOrganizingMap1DBase):
    """Initializes a Self-Organizing Map.

    Inspired by the design of [MiniSom]_

    Parameters
    ----------
    nlattice : int
        Number of lattice points (prototypes) in the 1D SOM.

    frame : `astropy.coordinates.BaseCoordinateFrame`
        The frame in which to build the SOM. Data is transformed into this frame
        before the SOM is fit.

    sigma : float, optional (default=1.0)
        Spread of the neighborhood function, needs to be adequate to the
        dimensions of the map. (at the iteration t we have sigma(t) = sigma / (1
        + t/T) where T is #num_iteration/2)
    learning_rate : initial learning rate
        (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T)
        where T is #num_iteration/2)
    rng : int, optional keyword-only (default=None)
        Random seed to use.

    representation_type : `astropy.coordinates.BaseRepresentation` or None, optional keyword-only
        The representation type in which to return coordinates.
        The representation type in which the SOM is fit is either
        `astropy.coordinates.CartesianRepresentation` if `onsky` is `False`
        or `astropy.coordinates.UnitSphericalRepresentation` if `onsky` is `True`.

    Notes
    -----
    neighborhood_function : 'gaussian'
        Function that weights the neighborhood of a position in the map.

    activation_distance : 'euclidean'
        Distance used to activate the map.

    References
    ----------
    .. [MiniSom] Giuseppe Vettigli. MiniSom: minimalistic and NumPy-based
        implementation of the Self Organizing Map.
    .. [frankenz] Josh Speagle. Frankenz: a photometric redshift monstrosity.
    """

    # -------------------------------------------

    @property
    def onsky(self) -> bool:
        """Whether to fit on-sky or 3d."""
        return False

    @classmethod
    @property
    def internal_representation_type(cls) -> type[CartesianRepresentation]:
        irt: type[CartesianRepresentation] = CartesianRepresentation
        return irt

    @property
    def nfeature(self) -> int:
        """Number of features."""
        return 3

    # ===============================================================

    def _activation_distance(self, x: ndarray, w: ndarray) -> ndarray:
        distance: ndarray
        distance = norm(subtract(x, w), axis=-1)
        return distance
        # TODO! change to ChiSquare
        # REDUCES TO GAUSSIAN LIKELIHOOD

    # ---------------------------------------------------------------
    # initialization

    def make_prototypes_binned(
        self, data: SkyCoord, byphi: bool = False, maxsep: Quantity | None = None, **_: Any
    ) -> None:
        r"""Initialize prototype vectors from binned data.

        Parameters
        ----------
        data : SkyCoord
        byphi : bool, optional
            Whether to bin by the |Longitude|, or by :math:`\phi=atan(lat/lon)`
        maxsep : Quantity or None, optional keyword-only
            Maximum separation (in data space) between prototypes.
        """
        # Get the data as a structured Quantity to set the units parameter
        q = self._crd_to_q(data)

        # Get coordinate to bin
        # This is most easily done as a NON-structured array
        xq: Quantity = cast(Quantity, q[q.dtype.names[0]])
        x = xq.value
        XT = np.array([cast(Quantity, q[n]).value for n in q.dtype.names])

        # Determine the binning coordinate
        if byphi:
            xq = np.arctan2(q[q.dtype.names[1]], q[q.dtype.names[0]])
            x = xq.value

        # Create equi-frequency bins
        bins: Quantity = cast(
            Quantity,
            np.interp(x=np.linspace(0, len(xq), self.nlattice + 1), xp=np.arange(len(xq)), fp=np.sort(xq)),
        )

        # Optionally respace the bins to have a maximum separation
        if maxsep is not None:
            # Check respacing is even possible
            if (abs(max(xq) - min(xq)) / self.nlattice) > maxsep:
                raise ValueError(
                    f"{self.nlattice} bins is not enough to cover [{min(xq)}, {max(xq)}] "
                    f"with a maximum bin separation of {maxsep}"
                )
            # Respace bins
            unit = u.rad if self.onsky else bins.unit
            bins = (
                _respace_bins(  # TODO! speed up
                    deepcopy(bins.to_value(unit)),
                    maxsep.to_value(unit),
                    onsky=self.onsky,
                    eps=2 * np.finfo(maxsep.dtype).eps,
                )
                * unit
            )

        # self._XT = XT
        # self._bins = deepcopy(bins)
        # self._xq = xq

        res = binned_statistic(x, XT, bins=bins, statistic="median")  # type: ignore
        prototypes: ndarray = res.statistic.T

        # When there is no data in a bin, it is set to NaN.
        # This is replaced with the interplation from nearby points.
        for j, d in enumerate(prototypes.T):
            mask = np.isnan(d)
            prototypes[mask, j] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), d[~mask])

        object.__setattr__(self, "prototypes", prototypes)
        object.__setattr__(self, "_init_prototypes", deepcopy(prototypes))

    # ---------------------------------------------------------------
    # fitting

    def _update(self, x: ndarray, t: int, max_iteration: int) -> None:
        """Update the locations of the prototypes.

        Parameters
        ----------
        x : ndarray
            Current point to learn.
        t : int
            Iteration index
        max_iteration : int
            Maximum number of training itarations.
        """
        eta = _decay_function(self.learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = _decay_function(self.sigma, t, max_iteration)
        # improves the performances
        ibmu = self._best_matching_unit_index(x)
        g = self._neighborhood(ibmu, sig) * eta
        # w_new = eta * neighborhood_function * (x-w)
        self.prototypes[:] += g[:, None] * (x - self.prototypes)  # type: ignore

    # ---------------------------------------------------------------
    # predicting structure

    def predict(self, crd: SkyCoord, origin: SkyCoord | None = None) -> tuple[SkyCoord, ndarray]:
        """Order data from SOM in 2+N Dimensions.

        Parameters
        ----------
        crd : SkyCoord
            This will generally be the same data used to train the SOM.
        origin : SkyCoord or None

        Returns
        -------
        ndarray

        Notes
        -----
        The SOM creates a 1D lattice of connected nodes ($q$'s, gray) ordered by
        proximity to the designated origin, then along the lattice.
        Data points ($p$'s, green) are assigned an order from the SOM-lattice.
        The distance from the data to each node is computed. Likewise the projection
        is found of each data point on the edges connecting each SOM node.
        All projections lying outside the edges (shaded regions) are eliminated,
        also eliminated are all but the closest nodes. Remaining edges and node
        connections in dotted block, with projections labeled $e$.
        Data points are sorted into the closest node regions (blue)
        and edge regions (shaded).
        Data points in end-cap node regions are sorted by extended
        projection on nearest edge.
        Data points in edge regions are ordered by projection along the edge.
        Data points in intermediate node regions are ordered by the angle between
        the edge regions.
        """
        # length of data, number of features: (x, y, z) or (ra, dec), etc.
        data = self._crd_to_v(crd)

        # Prediction of ordering. Might need to correct for the origin and
        # phase wraps.
        viip1, tM, all_points, distances = self._get_info_for_projection(data)
        ordering = self._order_along_projection(data, viip1, tM, distances)
        projdata = self._get_projected_point(data, all_points, distances)

        # Correct for a phase wrap

        if origin is not None:

            # the visit order can be backward so need to detect proximity to origin
            # TODO! more careful if closest point not end point. & adjust SOM!
            armep = cast(SkyCoord, crd[ordering[[0, -1]]])  # end points

            sep: Quantity
            sep = armep.separation(origin) if self.onsky else armep.separation_3d(origin)

            if np.argmin(sep) == 1:  # the end point is closer than the start
                ordering = ordering[::-1]

        projdata = cast(SkyCoord, projdata[ordering])
        return projdata, ordering


# ----------------------------------------------------------------------------


@define(frozen=True, kw_only=True)
class UnitSphereSelfOrganizingMap1D(SelfOrganizingMap1DBase):
    """Initializes a Self-Organizing Map.

    Inspired by the design of [MiniSom]_

    Parameters
    ----------
    nlattice : int
        Number of lattice points (prototypes) in the 1D SOM.
    onsky : bool or None
        Whether to fit on-sky or 3d.

    frame : `astropy.coordinates.BaseCoordinateFrame`
        The frame in which to build the SOM. Data is transformed into this frame
        before the SOM is fit.

    sigma : float, optional (default=1.0)
        Spread of the neighborhood function, needs to be adequate to the
        dimensions of the map. (at the iteration t we have sigma(t) = sigma / (1
        + t/T) where T is #num_iteration/2)
    learning_rate : initial learning rate
        (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T)
        where T is #num_iteration/2)
    rng : int, optional keyword-only (default=None)
        Random seed to use.

    representation_type : `astropy.coordinates.BaseRepresentation` or None, optional keyword-only
        The representation type in which to return coordinates.
        The representation type in which the SOM is fit is either
        `astropy.coordinates.CartesianRepresentation` if `onsky` is `False`
        or `astropy.coordinates.UnitSphericalRepresentation` if `onsky` is `True`.

    Notes
    -----
    neighborhood_function : 'gaussian'
        Function that weights the neighborhood of a position in the map.

    activation_distance : 'euclidean'
        Distance used to activate the map.

    References
    ----------
    .. [MiniSom] Giuseppe Vettigli. MiniSom: minimalistic and NumPy-based
        implementation of the Self Organizing Map.
    .. [frankenz] Josh Speagle. Frankenz: a photometric redshift monstrosity.
    """

    @property
    def onsky(self) -> bool:
        """Whether to fit on-sky or 3d."""
        return True

    @classmethod
    @property
    def internal_representation_type(cls) -> type[UnitSphericalRepresentation]:
        return UnitSphericalRepresentation

    @property
    def nfeature(self) -> int:
        """Number of features."""
        return 2

    # ===============================================================

    # ---------------------------------------------------------------
    # initialization

    def make_prototypes_binned(
        self, data: SkyCoord, byphi: bool = False, maxsep: Quantity | None = None, **_: Any
    ) -> None:
        r"""Initialize prototype vectors from binned data.

        Parameters
        ----------
        data : SkyCoord
        byphi : bool, optional
            Whether to bin by the |Longitude|, or by :math:`\phi=atan(lat/lon)`
        maxsep : Quantity or None, optional keyword-only
            Maximum separation (in data space) between prototypes.
        """
        # Get the data as a structured Quantity to set the units parameter
        q = self._crd_to_q(data)

        # Get coordinate to bin
        # This is most easily done as a NON-structured array
        xq: Quantity = cast(Quantity, q[q.dtype.names[0]])
        x: ndarray = xq.value
        XT = np.array([cast(Quantity, q[n]).value for n in q.dtype.names])

        # Determine the binning coordinate
        if byphi:
            xq = np.arctan2(q[q.dtype.names[1]], q[q.dtype.names[0]])
            x = xq.value
        else:
            xq = Quantity(Angle(xq, copy=False).wrap_at("180d"), u.rad, copy=False)
            x = cast(ndarray, xq.to_value(u.rad))

            # Unwrap, as best we can
            # Start by separating the populations so we can grab all wrapped points
            d = distance_matrix(x[:, None], x[:, None])
            d[d == 0] = np.nan
            t = mean(np.nanmean(d, axis=0))  # typical separation, upweighted by sep groups
            label = fclusterdata(x[:, None], t=t, criterion="distance") - 1
            # TODO! this is only for 2 pops, what if 3+?
            x0, x1, lesser = x[label == 0], x[label == 1], 0
            # determine if there's more than one group. There might be only 1.
            groups = True if (len(x0) >= 1 and len(x1) >= 1) else False
            if groups and min(x1) < min(x0):  # rearrange to correct order
                lesser = 1
                x0, x1 = x1, x0
            if groups and angular_separation(min(x0), 0, max(x1), 0) < t:
                idx = label == lesser

                x[idx] = x[idx] + 2 * np.pi
                # xq[idx] = xq[idx] + 2 * np.pi * u.rad  # x is a view
                XT[0, idx] = XT[0, idx] + 2 * np.pi

        # Create equi-frequency bins
        # https://www.statology.org/equal-frequency-binning-python/
        # endpoint=False is used to prevent a x>xp endpoint repetition
        bins: Quantity = cast(
            Quantity,
            np.interp(
                x=np.linspace(0, len(xq), self.nlattice + 1, endpoint=False),
                xp=np.arange(len(xq)),
                fp=np.sort(xq),
            ),
        )

        # Optionally respace the bins to have a maximum separation
        if maxsep is not None:
            # Check respacing is even possible
            if (abs(max(xq) - min(xq)) / self.nlattice) > maxsep:
                raise ValueError(
                    f"{self.nlattice} bins is not enough to cover [{min(xq)}, {max(xq)}] "
                    f"with a maximum bin separation of {maxsep}"
                )

            # Respace bins
            unit = u.rad if self.onsky else bins.unit
            bins = Quantity(
                _respace_bins(
                    deepcopy(bins.to_value(unit)),
                    maxsep.to_value(unit),
                    onsky=self.onsky,
                    eps=2 * np.finfo(maxsep.dtype).eps,
                ),
                unit,
            )

        res = binned_statistic(x, XT, bins=bins, statistic="median")  # type: ignore
        prototypes: ndarray = res.statistic.T

        # When there is no data in a bin, it is set to NaN.
        # This is replaced with the interplation from nearby points.
        for j, d in enumerate(prototypes.T):
            mask = np.isnan(d)
            prototypes[mask, j] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), d[~mask])

        object.__setattr__(self, "prototypes", prototypes)
        object.__setattr__(self, "_init_prototypes", deepcopy(prototypes))

    # ---------------------------------------------------------------
    # fitting

    def _activation_distance(self, x: ndarray, w: ndarray) -> ndarray:
        distance: ndarray
        distance = angular_separation(*x, *w.T)
        return distance

    def _update(self, x: ndarray, t: int, max_iteration: int) -> None:
        """Update the locations of the prototypes.

        Parameters
        ----------
        x : (D,) ndarray['radian']
            Current point to learn.
        t : int
            Iteration index
        max_iteration : int
            Maximum number of training itarations.
        """
        eta = _decay_function(self.learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = _decay_function(self.sigma, t, max_iteration)
        # improves the performances
        ibmu = self._best_matching_unit_index(x)
        g = self._neighborhood(ibmu, sig) * eta
        # w_new = eta * neighborhood_function * (x-w)
        # The first two dimensions are angular. The distance should be on the sphere.
        ps = self.prototypes
        separation = angular_separation(ps[:, 0], ps[:, 1], x[0], x[1])
        posang = position_angle(ps[:, 0], ps[:, 1], x[0], x[1])
        nlon, nlat = offset_by(ps[:, 0], ps[:, 1], posang=posang, distance=g * separation)
        nps = np.c_[nlon, nlat]  # same shape
        # Note: need to do non-angular components, if ever add

        object.__setattr__(self, "prototypes", nps)

    # ---------------------------------------------------------------
    # predicting structure

    def predict(self, crd: SkyCoord, origin: SkyCoord | None = None) -> tuple[SkyCoord, ndarray]:
        """Order data from SOM in 2+N Dimensions.

        Parameters
        ----------
        crd : SkyCoord
            This will generally be the same data used to train the SOM.
        origin : SkyCoord or None

        Returns
        -------
        ndarray

        Notes
        -----
        The SOM creates a 1D lattice of connected nodes ($q$'s, gray) ordered by
        proximity to the designated origin, then along the lattice.
        Data points ($p$'s, green) are assigned an order from the SOM-lattice.
        The distance from the data to each node is computed. Likewise the projection
        is found of each data point on the edges connecting each SOM node.
        All projections lying outside the edges (shaded regions) are eliminated,
        also eliminated are all but the closest nodes. Remaining edges and node
        connections in dotted block, with projections labeled $e$.
        Data points are sorted into the closest node regions (blue)
        and edge regions (shaded).
        Data points in end-cap node regions are sorted by extended
        projection on nearest edge.
        Data points in edge regions are ordered by projection along the edge.
        Data points in intermediate node regions are ordered by the angle between
        the edge regions.
        """
        # length of data, number of features: (x, y, z) or (ra, dec), etc.
        data = self._crd_to_v(crd)

        # Prediction of ordering. Might need to correct for the origin and
        # phase wraps.
        viip1, tM, all_points, distances = self._get_info_for_projection(data)
        ordering = self._order_along_projection(data, viip1, tM, distances)
        projdata = self._get_projected_point(data, all_points, distances)

        # Correct for a phase wrap
        qs = self._crd_to_q(crd)
        oq = cast(Quantity, qs[qs.dtype.names[0]][ordering])

        # def unwrap(q, /, visit_order=None, discont=pi/2*u.rad, period=2*pi*u.rad):
        discont = np.pi / 2 * u.rad
        # period = 2 * np.pi * u.rad

        # TODO! more general correction for arbitrary number of phase wraps
        jumps = np.where(np.diff(oq) >= discont)[0]
        if len(jumps) == 1:
            i = jumps[0] + 1
            ordering = np.concatenate((ordering[i:], ordering[:i]))

        # ----------------------------------------

        if origin is not None:

            # the visit order can be backward so need to detect proximity to origin
            # TODO! more careful if closest point not end point. & adjust SOM!
            armep = cast(SkyCoord, crd[ordering[[0, -1]]])  # end points

            sep: Quantity
            sep = armep.separation(origin)

            if np.argmin(sep) == 1:  # the end point is closer than the start
                ordering = ordering[::-1]

        projdata = cast(SkyCoord, projdata[ordering])
        return projdata, ordering
