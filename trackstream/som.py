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
from typing import Any, Callable, Optional, Union

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import BaseCoordinateFrame, BaseRepresentation, SkyCoord
from astropy.units import Quantity, StructuredUnit, Unit
from numpy import linalg, ndarray, pi, random
from numpy.lib.recfunctions import structured_to_unstructured
from scipy.stats import binned_statistic

# LOCAL
from trackstream._type_hints import CoordinateType
from trackstream.base import CommonBase
from trackstream.utils.pbar import get_progress_bar

__all__ = ["SelfOrganizingMap1D"]

##############################################################################
# PARAMETERS

warnings.filterwarnings(
    "ignore",
    message="Warning: sigma is too high for the dimension of the map.",
)

DataType = Union[CoordinateType, BaseRepresentation]


##############################################################################
# CODE
##############################################################################


class SelfOrganizingMap1D(CommonBase):
    """Initializes a Self-Organizing Map, (modified from [MiniSom]_).

    Parameters
    ----------
    nlattice : int
        Number of lattice points (prototypes) in the 1D SOM.
    nfeature : int
        Number of dimensions in the input.

    sigma : float, optional (default=1.0)
        Spread of the neighborhood function, needs to be adequate
        to the dimensions of the map.
        (at the iteration t we have sigma(t) = sigma / (1 + t/T)
        where T is #num_iteration/2)
    learning_rate : initial learning rate
        (at the iteration t we have
        learning_rate(t) = learning_rate / (1 + t/T)
        where T is #num_iteration/2)

    random_seed : int, optional keyword-only (default=None)
        Random seed to use.

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

    def __init__(
        self,
        nlattice: int,
        nfeature: int,
        frame: BaseCoordinateFrame,
        *,
        sigma: float = 0.1,
        learning_rate: float = 0.3,
        random_seed: Optional[int] = None,
        representation_type: Optional[BaseRepresentation] = None,
        **kwargs: Any,
    ) -> None:
        if sigma >= 1 or sigma >= nlattice:
            warnings.warn("sigma is too high for the dimension of the map")

        # for interpreting input data
        super().__init__(frame=frame, representation_type=representation_type)
        self._units = None

        # normal SOM stuff
        self._nlattice = nlattice
        self._nfeature = nfeature
        self._sigma = sigma
        self._learning_rate = learning_rate
        self._decay_function = asymptotic_decay

        self._rng = random.default_rng(random_seed)

        # used to evaluate the neighborhood function
        self._yy = np.arange(nlattice).astype(float)

        # random initialization
        self._prototypes = 2 * self._rng.random((nlattice, nfeature)) - 1

    @property
    def data_units(self) -> StructuredUnit:
        if self._units is None:
            raise ValueError("SOM must be fit for data_units")
        return self._units  # type: ignore

    @property
    def nlattice(self) -> int:
        """Number of lattice points."""
        return self._nlattice

    @property
    def nfeature(self) -> int:
        """Number of features."""
        return self._nfeature

    @property
    def sigma(self) -> float:
        return self._sigma

    @property
    def learning_rate(self) -> float:
        """Learning Rate."""
        return self._learning_rate

    @property
    def decay_function(self) -> Callable:
        """Decay function."""
        return self._decay_function

    @property
    def prototypes(self) -> BaseCoordinateFrame:
        "Read-only view of prototypes vectors."
        p = self._prototypes.view()
        p.flags.writeable = False

        return self._v_to_crd(p)

    # ===============================================================

    def _crd_to_q(self, crd: DataType, /) -> Quantity:
        """Coordinate to structured Quantity."""

        if isinstance(crd, (BaseCoordinateFrame, SkyCoord)):
            crd = crd.transform_to(self.frame)

        rep = crd.represent_as(self.representation_type, in_frame_units=True)

        data = rep._values << Unit(tuple(rep._units.values()))  # structured quantity
        return data

    def _crd_to_v(self, crd: Union[DataType, ndarray], /) -> ndarray:
        """Coordinate to unstructured array."""
        if isinstance(crd, ndarray):  # TODO! more careful check
            return crd

        v: ndarray = structured_to_unstructured(self._crd_to_q(crd).value)
        return v

    def _v_to_crd(self, arr: ndarray, /) -> BaseCoordinateFrame:
        data = {n: (arr[:, i] << unit) for i, (n, unit) in enumerate(self.data_units.items())}
        rep = self.representation_type(**data)
        crd = self.frame.realize_frame(rep)
        return crd

    # ===============================================================

    def _activation_distance(self, x: ndarray, w: ndarray) -> ndarray:
        distance: ndarray = linalg.norm(np.subtract(x, w), axis=-1)
        return distance
        # TODO! change to ChiSquare
        # REDUCES TO GAUSSIAN LIKELIHOOD

    def _distance_from_weights(self, data: ndarray) -> ndarray:
        """Euclidean distance matrix.

        Parameters
        ----------
        data : ndarray

        Returns
        -------
        ndarray
            A matrix D where D[i,j] is the euclidean distance between
            data[i] and the j-th weight.
        """
        weights_flat = self._prototypes
        input_data_sq = np.power(data, 2).sum(axis=1, keepdims=True)
        weights_flat_sq = np.power(weights_flat, 2).sum(axis=1, keepdims=True)
        cross_term = np.dot(data, weights_flat.T)
        distance: ndarray = np.sqrt(input_data_sq + weights_flat_sq.T - (2 * cross_term))
        return distance

    # ---------------------------------------------------------------
    # initialization

    def binned_weights_init(self, data: ndarray, byphi: bool = False, **kw: Any) -> None:
        r"""Initialize prototype vectors from binned data.

        Parameters
        ----------
        data : Frame or Representation
        byphi : bool, optional
            Whether to bin by the longitude, or by :math:`\phi=atan(lat/lon)`
        """
        q = self._crd_to_q(data)
        self._units = q.unit

        X = self._crd_to_v(data)

        if byphi:
            x1 = np.arctan2(X[:, 1], X[:, 0])
        else:
            x1 = X[:, 0]
        xlen = len(x1)

        # create equi-frequency bins
        bins = np.interp(np.linspace(0, xlen, self.nlattice + 1), np.arange(xlen), np.sort(x1))

        # compute the mean positions
        res = binned_statistic(x1, X.T, bins=bins, statistic="median")

        # TODO! set minimumseparation

        self._prototypes = res.statistic.T

    # ---------------------------------------------------------------
    # fitting

    #     def quantization(self, data: DataType) -> ndarray:  # TODO!
    #         """Assigns a code book (weights vector of the winning neuron)
    #         to each sample in data.
    #
    #         Parameters
    #         ----------
    #         """
    #         data = self._crd_to_v(data)
    #         winners_coords = np.argmin(self._distance_from_weights(data), axis=1)
    #         ps = self._prototypes[winners_coords]
    #         return self._v_to_crd(ps)

    # def quantization_error(self, data):  # TODO!
    #     """
    #     Returns the quantization error computed as the average
    #     distance between each input sample and its best matching unit.
    #     """
    #     return linalg.norm(data - self.quantization(data), axis=1).mean()

    def fit(
        self, data: DataType, num_iteration: int, random_order: bool = False, progress: bool = False
    ) -> None:
        """Trains the SOM.

        Parameters
        ----------
        data : Frame or Representation

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
            self._rng.shuffle(iterations)

        # Get the in internal unitless representation
        if self._units is None:  # if not init weighted
            q = self._crd_to_q(data)
            self._units = q.unit
        X = self._crd_to_v(data)

        # Fit the data by sequential update
        with get_progress_bar(progress, len(iterations)) as pbar:
            for t, iteration in enumerate(iterations):
                pbar.update(1)

                self._update(
                    X[iteration],
                    self._best_matching_unit_index(X[iteration]),
                    t,
                    num_iteration,
                )

    def _neighborhood(self, c: int, sigma: float) -> ndarray:
        """Returns a Gaussian centered in c."""
        d = 2 * pi * sigma ** 2
        ay: ndarray = np.exp(-np.power(self._yy - self._yy.T[c], 2) / d).T
        return ay  # the external product gives a matrix

    def _update(self, x: ndarray, ibmu: int, t: int, max_iteration: int) -> None:
        """Update the locations of the prototypes.

        Parameters
        ----------
        x : np.array
            Current point to learn.
        ibmu : int
            Position of the best-matching prototype for x.
        t : int
            Iteration index
        max_iteration : int
            Maximum number of training itarations.
        """
        eta = self._decay_function(self._learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = self._decay_function(self._sigma, t, max_iteration)
        # improves the performances
        g = self._neighborhood(ibmu, sig) * eta
        # w_new = eta * neighborhood_function * (x-w)
        self._prototypes += np.einsum("i, ij->ij", g, x - self._prototypes)

    def _best_matching_unit_index(self, x: ndarray) -> int:
        """Computes the coordinates of the best prototype for the sample.

        Parameters
        ----------
        x : array

        Returns
        -------
        int
            The index of the best-matching prototype.
        """
        activation_map = self._activation_distance(x, self._prototypes)
        bmu = int(activation_map.argmin())
        return bmu

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
        liip1 = np.sum(np.square(viip1), axis=1)

        # data - point_i  (D, nlattice-1, nfeature)
        # for each slice in D,
        dmi = data[:, None, :] - p1[None, :, :]  # d-p1

        # The line extending the segment is parameterized as p1 + t (p2 - p1).
        # The projection falls where t = [(p3-p1) . (p2-p1)] / |p2-p1|^2
        # tM is the matrix of "t"'s.
        tM = np.sum((dmi * viip1[None, :, :]), axis=-1) / liip1

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

    def predict(self, crd: DataType, origin: Optional[SkyCoord] = None) -> ndarray:
        """Order data from SOM in 2+N Dimensions.

        Parameters
        ----------
        crd : |SkyCoord| or |Frame| or |Representation|
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
        # TODO! more general correction for arbitrary number of phase wraps
        qs = self._crd_to_q(crd)
        oq = qs[qs.dtype.names[0]][ordering]

        if oq.unit.physical_type == "angle":
            # def unwrap(q, /, visit_order=None, discont=pi/2*u.rad, period=2*pi*u.rad):
            discont = np.pi / 2 * u.rad
            # period = 2 * np.pi * u.rad

            jumps = np.where(np.diff(oq) >= discont)[0]
            if jumps:
                i = jumps[0] + 1
                ordering = np.concatenate((ordering[i:], ordering[:i]))

        # ----------------------------------------

        if origin is not None:

            # the visit order can be backward so need to detect proximity to origin
            # TODO! more careful if closest point not end point. & adjust SOM!
            armep = crd[ordering[[0, -1]]]  # end points

            # FIXME! be careful about 2d versus 3d
            try:
                sep = armep.separation_3d(origin)
            except ValueError:
                sep = armep.separation(origin)

            if np.argmin(sep) == 1:  # the end point is closer than the start
                ordering = ordering[::-1]

        return ordering

    def fit_predict(
        self,
        data: DataType,
        num_iteration: int,
        random_order: bool = False,
        progress: bool = False,
        origin: Optional[SkyCoord] = None,
    ) -> ndarray:
        """

        Returns
        -------
        ndarray
        """
        self.fit(data, num_iteration=num_iteration, random_order=random_order, progress=progress)
        order = self.predict(data, origin=origin)
        return order


# -------------------------------------------------------------------


def asymptotic_decay(learning_rate: float, iteration: int, max_iter: float) -> float:
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


# def reorder_visits(
#     data: CoordinateType,
#     visit_order: ndarray,
#     start_ind: int,
# ):
#     """Reorder the points from the SOM.
#
#     The SOM does not always keep the starting point at the beginning
#     nor even "direction" of the indices. This function can flip the
#     index ordering and rotate the indices such that the starting point
#     stays at the beginning.
#
#     The rotation is done by snipping and restitching the array at
#     `start_ind`. The "direction" is determined by the distance between
#     data points at visit_order[start_ind-1] and visit_order[start_ind+1].
#
#     Parameters
#     ----------
#     data : CoordinateType
#         The data.
#     visit_order: ndarray[int]
#         Index array ordering `data`. Will be flipped and / or rotated
#         such that `start_ind` is the 0th element and the next element
#         is the closest.
#     start_ind : int
#         The starting index
#
#     Returns
#     -------
#     new_visit_order : Sequence
#         reordering of `visit_order`
#
#     """
#     # index of start_ind in visit_order
#     # this needs to be made the first index.
#     i = list(visit_order).index(start_ind)
#
#     # snipping and restitching
#     # first taking care of the edge cases
#     if i == 0:  # AOK
#         pass
#     elif i == len(visit_order) - 1:  # just reverse.
#         i = 0
#         visit_order = visit_order[::-1]
#     else:  # need to figure out direction before restitching
#         back = (data[visit_order[i]] - data[visit_order[i - 1]]).norm()
#         forw = (data[visit_order[i + 1]] - data[visit_order[i]]).norm()
#
#         if back < forw:  # things got reversed...
#             visit_order = visit_order[::-1]  # flip visit_order
#             i = list(visit_order).index(start_ind)  # re-find index
#
#     # do the stitch
#     new_visit_order = np.concatenate((visit_order[i:], visit_order[:i]))
#
#     return new_visit_order
