# -*- coding: utf-8 -*-

"""Self-Organizing Maps.

References
----------
.. [MiniSom] Giuseppe Vettigli. MiniSom: minimalistic and NumPy-based
    implementation of the Self Organizing Map.
.. [frankenz] Josh Speagle. Frankenz: a photometric redshift monstrosity.

"""
__credits__ = "MiniSom"


##############################################################################
# IMPORTS

# STDLIB
import warnings
from abc import abstractmethod
from copy import deepcopy
from typing import Any, Dict, Optional, Type, Union, cast

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import Angle, BaseCoordinateFrame, BaseRepresentation
from astropy.coordinates import CartesianRepresentation, SkyCoord, UnitSphericalRepresentation
from astropy.coordinates.angle_utilities import angular_separation
from astropy.units import Quantity, StructuredUnit
from numpy import arange, arccos, arctan2, array, cos, diff, dtype, exp, flatnonzero, interp, isnan
from numpy import linalg, linspace, mean, nan, nanmean, ndarray, nonzero, pi, power, random, sort
from numpy import square, subtract, sum
from numpy.lib.recfunctions import structured_to_unstructured, unstructured_to_structured
from scipy.cluster.hierarchy import fclusterdata
from scipy.spatial import distance_matrix
from scipy.stats import binned_statistic

# LOCAL
from trackstream._type_hints import CoordinateType
from trackstream.base import CommonBase
from trackstream.utils.coord_utils import offset_by, position_angle
from trackstream.utils.pbar import get_progress_bar

__all__ = ["CartesianSelfOrganizingMap1D", "UnitSphereSelfOrganizingMap1D"]

##############################################################################
# PARAMETERS

warnings.filterwarnings(
    "ignore",
    message="sigma is too high for the dimension of the map",
)

DataType = Union[CoordinateType, BaseRepresentation]


##############################################################################
# CODE
##############################################################################


def _respace_bins_from_left(bins: ndarray, maxsep: ndarray, onsky: bool, eps: float) -> ndarray:
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


def _respace_bins(bins: ndarray, maxsep: ndarray, onsky: bool, eps: float) -> ndarray:
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
    diffs = np.arccos(np.cos(np.diff(bins))) if onsky else diff(bins)
    (seps,) = np.nonzero(diffs > maxsep)

    i = 0  # cap at 10k iterations
    while any(seps) and i < 50:

        # Adjust from the left, then adjust from the right
        bins = _respace_bins_from_left(bins, maxsep=maxsep, onsky=onsky, eps=eps)
        bins[::-1] = -_respace_bins_from_left(-bins[::-1], maxsep=maxsep, onsky=onsky, eps=eps)

        # Check the separations
        diffs = np.arccos(np.cos(np.diff(bins))) if onsky else diff(bins)
        (seps,) = np.nonzero(diffs > maxsep)

        i += 1

    return bins


##############################################################################


class SelfOrganizingMap1DBase(CommonBase):
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
    random_seed : int, optional keyword-only (default=None)
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

    _units: Optional[u.StructuredUnit]

    _nlattice: int
    _sigma: float
    _prototypes: ndarray
    _yy: ndarray

    _rng: random.Generator

    def __init__(
        self,
        nlattice: int,
        frame: BaseCoordinateFrame,
        *,
        sigma: float = 0.1,
        learning_rate: float = 0.3,
        random_seed: Optional[int] = None,
        representation_type: Optional[Type[BaseRepresentation]] = None,
    ) -> None:
        if sigma >= 1 or sigma >= nlattice:
            warnings.warn("sigma is too high for the dimension of the map")

        # for interpreting input data
        super().__init__(frame=frame, representation_type=representation_type)
        self._units = None

        # normal SOM stuff
        self._nlattice = nlattice
        self._sigma = sigma
        self._learning_rate = learning_rate

        self._rng = random.default_rng(random_seed)

        # used to evaluate the neighborhood function
        self._yy = arange(nlattice, dtype=float)

        # random initialization, if onsky is known
        self.make_prototypes_random()

    # -------------------------------------------

    @property
    @abstractmethod
    def onsky(self) -> bool:
        """Whether to fit on-sky or 3d."""

    @property
    def data_units(self) -> StructuredUnit:
        if self._units is None:
            raise ValueError("SOM must be fit for data_units")
        return self._units  # type: ignore

    @property
    @abstractmethod
    def internal_representation_type(self) -> Type[BaseRepresentation]:
        """Representation type."""

    # -------------------------------------------

    @property
    def nlattice(self) -> int:
        """Number of lattice points."""
        return self._nlattice

    @property
    @abstractmethod
    def nfeature(self) -> int:
        """Number of features."""

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def learning_rate(self) -> float:
        """Learning Rate."""
        return self._learning_rate

    @staticmethod
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

    @property
    def prototypes(self) -> BaseCoordinateFrame:
        "Read-only view of prototypes vectors."
        p = self._prototypes.view()
        p.flags.writeable = False

        return self._v_to_crd(p)

    @property
    def init_prototypes(self) -> BaseCoordinateFrame:
        "Read-only view of prototypes vectors."
        p = self._init_prototypes.view()
        p.flags.writeable = False

        return self._v_to_crd(p)

    # ===============================================================

    def _crd_to_q(self, crd: CoordinateType) -> Quantity:
        """Coordinate to structured Quantity."""
        crd = crd.transform_to(self.frame)
        rep = crd.represent_as(self.internal_representation_type)
        units: Dict[str, u.UnitBase] = rep._units

        # structured quantity
        su = StructuredUnit(tuple(units.values()), names=tuple(units.keys()))
        q: Quantity = rep._values << su
        return q

    def _crd_to_v(self, crd: Union[CoordinateType, ndarray]) -> ndarray:
        """Coordinate to unstructured array."""
        if isinstance(crd, ndarray):
            return crd

        q = self._crd_to_q(crd)
        v: ndarray = structured_to_unstructured(q.value)
        return v

    def _v_to_q(self, v: ndarray, /) -> Quantity:
        dt = dtype([(n, v.dtype) for n in self.data_units.field_names])
        q = Quantity(unstructured_to_structured(v, dt), unit=self.data_units)
        return q

    def _v_to_crd(self, arr: ndarray, /) -> BaseCoordinateFrame:
        data = {n: (arr[:, i] << unit) for i, (n, unit) in enumerate(self.data_units.items())}
        rep = self.internal_representation_type(**data)
        crd = self.frame.realize_frame(rep)
        crd.representation_type = self.representation_type
        return crd

    # ===============================================================

    # ---------------------------------------------------------------
    # prototypes

    def make_prototypes_random(self) -> None:
        """Randomly initialize prototype vectors."""
        prototypes = 2 * self._rng.random((self.nlattice, self.nfeature)) - 1
        self._prototypes = prototypes
        self._init_prototypes = deepcopy(prototypes)

    @abstractmethod
    def make_prototypes_binned(
        self, data: SkyCoord, byphi: bool = False, maxsep: Optional[Quantity] = None, **_: Any
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
        num_iteration: int,
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
        iterations = arange(num_iteration) % len(data)
        # Optionally randomize the cycles
        if random_order:
            self._rng.shuffle(iterations)

        # Get the in internal unitless representation
        q = self._crd_to_q(data)
        if self._units is None:  # if not init weighted
            self._units = cast(u.StructuredUnit, q.unit)
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
        d = 2 * pi * sigma ** 2
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
        activation_map = self._activation_distance(x, self._prototypes)
        ibmu = int(activation_map.argmin())
        return ibmu

    # ---------------------------------------------------------------
    # predicting structure

    def _order_along_projection(self, data: ndarray) -> ndarray:
        data_len, nfeature = data.shape
        nlattice = self.nlattice

        # vector from one point to next  (nlattice-1, nfeature)
        lattice_points = self._prototypes
        p1 = lattice_points[:-1, :]
        p2 = lattice_points[1:, :]
        # vector from one point to next  (nlattice-1, nfeature)
        viip1 = p2 - p1
        # square distance from one point to next  (nlattice-1, nfeature)
        liip1 = sum(square(viip1), axis=1)

        # data - point_i  (D, nlattice-1, nfeature)
        # for each slice in D,
        dmi = data[:, None, :] - p1[None, :, :]  # d-p1

        # The line extending the segment is parameterized as p1 + t (p2 - p1).
        # The projection falls where t = [(p3-p1) . (p2-p1)] / |p2-p1|^2
        # tM is the matrix of "t"'s.
        tM = sum((dmi * viip1[None, :, :]), axis=-1) / liip1

        projected_points = p1[None, :, :] + tM[:, :, None] * viip1[None, :, :]

        # add in the nodes and find all the distances
        # the correct "place" to put the data point is within a
        # projection, unless it outside (by and endpoint)
        # or inside, but on the convex side of a segment junction
        all_points = np.empty((data_len, 2 * nlattice - 1, nfeature), dtype=float)
        all_points[:, 1::2, :] = projected_points
        all_points[:, 0::2, :] = lattice_points
        distances = np.linalg.norm(data[:, None, :] - all_points, axis=-1)

        # detect whether it is in the segment
        # nodes are considered in the segment
        not_in_projection = np.zeros(all_points.shape[:-1], dtype=bool)
        not_in_projection[:, 1::2] = np.logical_or(tM <= 0, 1 <= tM)

        # make distances for not-in-segment infinity
        distances[not_in_projection] = np.inf

        # find the best distance (including nodes)
        best_distance = np.argmin(distances, axis=1)

        ordering = np.zeros(len(data), dtype=int) - 1

        counter = 0  # count through edge/node groups
        for i in np.unique(best_distance):
            # for i in (1, ):
            # get the data rows for which the best distance is the i'th node/segment
            rowi = np.where((best_distance == i))[0]
            numrows = len(rowi)

            # move edge points to corresponding segment
            if i == 0:
                i = 1
            elif i == 2 * (nlattice - 1):
                i = nlattice - 1

            # odds (remainder 1) are segments
            if bool(i % 2):
                ts = tM[rowi, i // 2]
                rowsorter = np.argsort(ts)

            # evens are by nodes
            else:  # TODO! find and fix the potential ordering mistake
                phi1 = arctan2(*viip1[i // 2 - 1, :2])
                phim2 = arctan2(*-viip1[i // 2, :2])
                phi = arctan2(*data[rowi, :2].T)

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
    def predict(self, crd: SkyCoord, origin: Optional[SkyCoord] = None) -> ndarray:
        pass

    def fit_predict(
        self,
        data: SkyCoord,
        num_iteration: int,
        random_order: bool = False,
        progress: bool = False,
        origin: Optional[SkyCoord] = None,
    ) -> ndarray:
        """Fit then predict.

        Returns
        -------
        ndarray

        See Also
        --------
        trackstream.SelfOrganizingMap1D.fit
        trackstream.SelfOrganizingMap1D.predict
        """
        self.fit(data, num_iteration=num_iteration, random_order=random_order, progress=progress)
        order = self.predict(data, origin=origin)
        return order


# ----------------------------------------------------------------------------


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
    random_seed : int, optional keyword-only (default=None)
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

    _units: Optional[u.StructuredUnit]

    _nlattice: int
    _sigma: float
    _prototypes: ndarray
    _yy: ndarray

    _rng: random.Generator

    # -------------------------------------------

    @property
    def onsky(self) -> bool:
        """Whether to fit on-sky or 3d."""
        return False

    @property
    def internal_representation_type(self) -> Type[CartesianRepresentation]:
        irt: Type[CartesianRepresentation] = CartesianRepresentation
        return irt

    @property
    def nfeature(self) -> int:
        """Number of features."""
        return 3

    # ===============================================================

    def _activation_distance(self, x: ndarray, w: ndarray) -> ndarray:
        distance: ndarray
        distance = linalg.norm(subtract(x, w), axis=-1)
        return distance
        # TODO! change to ChiSquare
        # REDUCES TO GAUSSIAN LIKELIHOOD

    # ---------------------------------------------------------------
    # initialization

    def make_prototypes_binned(
        self, data: SkyCoord, byphi: bool = False, maxsep: Optional[Quantity] = None, **_: Any
    ) -> None:
        r"""Initialize prototype vectors from binned data.

        Parameters
        ----------
        data : SkyCoord
        byphi : bool, optional
            Whether to bin by the longitude, or by :math:`\phi=atan(lat/lon)`
        maxsep : Quantity or None, optional keyword-only
            Maximum separation (in data space) between prototypes.
        """
        # Get the data as a structured Quantity to set the units parameter
        q = self._crd_to_q(data)
        self._units = cast(u.StructuredUnit, q.unit)

        # Get coordinate to bin
        # This is most easily done as a NON-structured array
        xq: Quantity = cast(Quantity, q[q.dtype.names[0]])
        x = xq.value
        XT = array([cast(Quantity, q[n]).value for n in q.dtype.names])

        # Determine the binning coordinate
        if byphi:
            xq = arctan2(q[q.dtype.names[1]], q[q.dtype.names[0]])
            x = xq.value

        # Create equi-frequency bins
        bins: Quantity = interp(
            x=linspace(0, len(xq), self.nlattice + 1), xp=arange(len(xq)), fp=sort(xq)
        )

        # Optionally respace the bins to have a maximum separation
        if maxsep is not None:
            # Check respacing is even possible
            if (abs(max(xq) - min(xq)) / self.nlattice) > maxsep:
                raise ValueError(
                    f"{self.nlattice} bins is not enough to cover [{min(xq)}, {max(xq)}] "
                    f"with a maximum bin separation of {maxsep}",
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
            mask = isnan(d)
            prototypes[mask, j] = interp(flatnonzero(mask), flatnonzero(~mask), d[~mask])

        self._prototypes = prototypes
        self._init_prototypes = deepcopy(prototypes)

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
        eta = self._decay_function(self._learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = self._decay_function(self._sigma, t, max_iteration)
        # improves the performances
        ibmu = self._best_matching_unit_index(x)
        g = self._neighborhood(ibmu, sig) * eta
        # w_new = eta * neighborhood_function * (x-w)
        self._prototypes += g[:, None] * (x - self._prototypes)  # type: ignore

    # ---------------------------------------------------------------
    # predicting structure

    def predict(self, crd: SkyCoord, origin: Optional[SkyCoord] = None) -> ndarray:
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
        ordering = self._order_along_projection(data)

        # Correct for a phase wrap

        if origin is not None:

            # the visit order can be backward so need to detect proximity to origin
            # TODO! more careful if closest point not end point. & adjust SOM!
            armep = cast(SkyCoord, crd[ordering[[0, -1]]])  # end points

            sep: Quantity
            sep = armep.separation(origin) if self.onsky else armep.separation_3d(origin)

            if np.argmin(sep) == 1:  # the end point is closer than the start
                ordering = ordering[::-1]

        return ordering


# ----------------------------------------------------------------------------


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
    random_seed : int, optional keyword-only (default=None)
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

    _units: Optional[u.StructuredUnit]

    _nlattice: int
    _sigma: float
    _prototypes: ndarray
    _yy: ndarray

    _rng: random.Generator

    @property
    def onsky(self) -> bool:
        """Whether to fit on-sky or 3d."""
        return True

    @property
    def internal_representation_type(self) -> Type[UnitSphericalRepresentation]:
        irt: Type[UnitSphericalRepresentation] = UnitSphericalRepresentation
        return irt

    @property
    def nfeature(self) -> int:
        """Number of features."""
        return 2

    # ===============================================================

    # ---------------------------------------------------------------
    # initialization

    def make_prototypes_binned(
        self, data: SkyCoord, byphi: bool = False, maxsep: Optional[Quantity] = None, **_: Any
    ) -> None:
        r"""Initialize prototype vectors from binned data.

        Parameters
        ----------
        data : SkyCoord
        byphi : bool, optional
            Whether to bin by the longitude, or by :math:`\phi=atan(lat/lon)`
        maxsep : Quantity or None, optional keyword-only
            Maximum separation (in data space) between prototypes.
        """
        # Get the data as a structured Quantity to set the units parameter
        q = self._crd_to_q(data)
        self._units = cast(u.StructuredUnit, q.unit)

        # Get coordinate to bin
        # This is most easily done as a NON-structured array
        xq: Quantity = cast(Quantity, q[q.dtype.names[0]])
        x: ndarray = xq.value
        XT = array([cast(Quantity, q[n]).value for n in q.dtype.names])

        # Determine the binning coordinate
        if byphi:
            xq = arctan2(q[q.dtype.names[1]], q[q.dtype.names[0]])
            x = xq.value
        else:
            xq = Quantity(Angle(xq, copy=False).wrap_at("180d"), u.rad, copy=False)
            x = cast(ndarray, xq.to_value(u.rad))

            # Unwrap, as best we can
            # Start by separating the populations so we can grab all wrapped points
            d = distance_matrix(x[:, None], x[:, None])
            d[d == 0] = nan
            t = mean(nanmean(d, axis=0))  # typical separation, upweighted by sep groups
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
        bins: Quantity = interp(
            x=linspace(0, len(xq), self.nlattice + 1, endpoint=False),
            xp=arange(len(xq)),
            fp=sort(xq),
        )

        # Optionally respace the bins to have a maximum separation
        if maxsep is not None:
            # Check respacing is even possible
            if (abs(max(xq) - min(xq)) / self.nlattice) > maxsep:
                raise ValueError(
                    f"{self.nlattice} bins is not enough to cover [{min(xq)}, {max(xq)}] "
                    f"with a maximum bin separation of {maxsep}",
                )

            # Respace bins
            unit = u.rad if self.onsky else bins.unit
            bins = (
                _respace_bins(
                    deepcopy(bins.to_value(unit)),
                    maxsep.to_value(unit),
                    onsky=self.onsky,
                    eps=2 * np.finfo(maxsep.dtype).eps,
                )
                * unit
            )

        res = binned_statistic(x, XT, bins=bins, statistic="median")  # type: ignore
        prototypes: ndarray = res.statistic.T

        # When there is no data in a bin, it is set to NaN.
        # This is replaced with the interplation from nearby points.
        for j, d in enumerate(prototypes.T):
            mask = isnan(d)
            prototypes[mask, j] = interp(flatnonzero(mask), flatnonzero(~mask), d[~mask])

        self._prototypes = prototypes
        self._init_prototypes = deepcopy(prototypes)

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
        eta = self._decay_function(self._learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = self._decay_function(self._sigma, t, max_iteration)
        # improves the performances
        ibmu = self._best_matching_unit_index(x)
        g = self._neighborhood(ibmu, sig) * eta
        # w_new = eta * neighborhood_function * (x-w)
        # The first two dimensions are angular. The distance should be on the sphere.
        ps = self._prototypes
        separation = angular_separation(ps[:, 0], ps[:, 1], x[0], x[1])
        posang = position_angle(ps[:, 0], ps[:, 1], x[0], x[1])
        nlon, nlat = offset_by(ps[:, 0], ps[:, 1], posang=posang, distance=g * separation)
        nps = np.c_[nlon, nlat]  # same shape
        # Note: need to do non-angular components, if ever add

        self._prototypes = nps

    # ---------------------------------------------------------------
    # predicting structure

    def predict(self, crd: SkyCoord, origin: Optional[SkyCoord] = None) -> ndarray:
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
        ordering = self._order_along_projection(data)

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

        return ordering
