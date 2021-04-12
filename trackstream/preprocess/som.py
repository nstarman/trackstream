# -*- coding: utf-8 -*-

"""Self-Organizing Maps.

TODO find slowest lines
https://marcobonzanini.com/2015/01/05/my-python-code-is-slow-tips-for-profiling/

References
----------
.. [MiniSom] Giuseppe Vettigli. MiniSom: minimalistic and NumPy-based
    implementation of the Self Organizing Map.
.. [frankenz] Josh Speagle. Frankenz: a photometric redshift monstrosity.

"""

__all__ = [
    "SelfOrganizingMap",
    # functions
    "apply_SOM",
    "apply_SOM_repeat",
    "make_transition_matrix",
    "draw_ordering",
    "draw_most_probable_ordering",
]

__credits__ = "MiniSom"


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T
import warnings
from collections import namedtuple

# THIRD PARTY
import astropy.coordinates as coord
import numpy as np
import typing_extensions as TE
from numpy import linalg, pi, random
from scipy import sparse
from tqdm import tqdm

# PROJECT-SPECIFIC
from .utils import DataType  # , find_closest_point, set_starting_point
from trackstream.config import conf
from trackstream.setup_package import HAS_MINISOM
from trackstream.type_hints import CoordinateType
from trackstream.utils.pbar import get_progress_bar

if conf.use_minisom:
    if not HAS_MINISOM:
        warnings.warn("Can't find MiniSOM, falling back to built-in SOM.")


##############################################################################
# PARAMETERS

warnings.filterwarnings(
    "ignore",
    message="Warning: sigma is too high for the dimension of the map.",
)


preprocessed = namedtuple(
    "preprocessed",
    ("data", "trmat", "visit_orders", "start_point"),
)


##############################################################################
# CODE
##############################################################################


class SelfOrganizingMap(object):
    """Initializes a Self-Organizing Map.

    Altered from [MiniSom]_.
    Log-Likelihood weighted "distance" from [frankenz]

    A rule of thumb to set the size of the grid for a dimensionality
    reduction task is that it should contain 5*sqrt(N) neurons
    where N is the number of samples in the dataset to analyze.

    E.g. if your dataset has 150 samples, 5*sqrt(150) = 61.23
    hence a map 8-by-8 should perform well.

    Parameters
    ----------
    x, y : int
        x, y dimension of the SOM.

    input_len : int
        Number of the elements of the vectors in input.

    sigma : float, optional (default=1.0)
        Spread of the neighborhood function, needs to be adequate
        to the dimensions of the map.
        (at the iteration t we have sigma(t) = sigma / (1 + t/T)
        where T is #num_iteration/2)
    learning_rate : initial learning rate
        (at the iteration t we have
        learning_rate(t) = learning_rate / (1 + t/T)
        where T is #num_iteration/2)

    decay_function : function (default=None)
        Function that reduces learning_rate and sigma at each iteration
        the default function is:
                    learning_rate / (1+t/(max_iterarations/2))

        A custom decay function will need to to take in input
        three parameters in the following order:

        1. learning rate
        2. current iteration
        3. maximum number of iterations allowed


        Note that if a lambda function is used to define the decay
        MiniSom will not be pickable anymore.

    random_seed : int, optional (default=None)
        Random seed to use.

    Notes
    -----
    neighborhood_function : 'gaussian'
        Function that weights the neighborhood of a position in the map.

    topology : 'rectangular'
        Topology of the map.

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
        x: int,
        y: int,
        input_len: int,
        sigma: float = 1.0,
        learning_rate: float = 0.5,
        decay_function: T.Union[
            T.Callable,
            TE.Literal["asymptotic"],
        ] = "asymptotic",
        random_seed: T.Optional[int] = None,
        **kwargs,
    ):
        """Self-Organizing Maps."""
        if sigma >= x or sigma >= y:
            warnings.warn(
                "Warning: sigma is too high for the dimension of the map.",
            )

        self._input_len = input_len
        self._sigma = sigma
        self._learning_rate = learning_rate

        if decay_function == "asymptotic":
            decay_function = asymptotic_decay
        self._decay_function = decay_function

        self._rng = random.RandomState(random_seed)

        self._activation_map = np.zeros((x, y))
        # used to evaluate the neighborhood function
        self._neigx = np.arange(x)
        self._neigy = np.arange(y)

        self._xx, self._yy = np.meshgrid(self._neigx, self._neigy)
        self._xx = self._xx.astype(float)
        self._yy = self._yy.astype(float)

        # random initialization
        self._weights = 2 * self._rng.rand(x, y, input_len) - 1
        self._weights /= linalg.norm(self._weights, axis=-1, keepdims=True)

    # /def

    def neighborhood(self, c, sigma) -> np.ndarray:
        """Returns a Gaussian centered in c."""
        d = 2 * pi * sigma ** 2
        ax = np.exp(-np.power(self._xx - self._xx.T[c], 2) / d)
        ay = np.exp(-np.power(self._yy - self._yy.T[c], 2) / d)
        return (ax * ay).T  # the external product gives a matrix

    # /def

    def _activate(self, x):
        """Updates matrix activation_map, in this matrix
        the element i,j is the response of the neuron i,j to x.

        """
        self._activation_map = self._activation_distance(x, self._weights)

    # /def

    def _activation_distance(self, x, w):
        return linalg.norm(np.subtract(x, w), axis=-1)

    # Change to ChiSquare
    # REDUCES TO GAUSSIAN LIKELIHOOD

    # /def

    def _distance_from_weights(self, data):
        """Returns a matrix d where d[i,j] is the euclidean distance between
        data[i] and the j-th weight.
        """
        input_data = np.array(data)
        weights_flat = self._weights.reshape(-1, self._weights.shape[2])
        input_data_sq = np.power(input_data, 2).sum(axis=1, keepdims=True)
        weights_flat_sq = np.power(weights_flat, 2).sum(axis=1, keepdims=True)
        cross_term = np.dot(input_data, weights_flat.T)
        return np.sqrt(input_data_sq + weights_flat_sq.T - (2 * cross_term))

    # /def

    def pca_weights_init(self, data):
        """Initializes the weights to span the first two principal components.

        This initialization doesn't depend on random processes and
        makes the training process converge faster.

        It is strongly recommended to normalize the data before initializing
        the weights and use the same normalization for the training data.

        """
        pc_length, pc = linalg.eig(np.cov(np.transpose(data)))
        pc_order = np.argsort(-pc_length)

        pc0 = pc[pc_order[0]]
        pc1 = pc[pc_order[1]]

        for i, c1 in enumerate(np.linspace(-1, 1, len(self._neigx))):
            for j, c2 in enumerate(np.linspace(-1, 1, len(self._neigy))):
                self._weights[i, j] = c1 * pc0 + c2 * pc1

    # /def

    def quantization(self, data):
        """Assigns a code book (weights vector of the winning neuron)
        to each sample in data.

        """
        winners_coords = np.argmin(self._distance_from_weights(data), axis=1)
        return self._weights[
            np.unravel_index(winners_coords, self._weights.shape[:2])
        ]

    # /def

    def quantization_error(self, data):
        """Returns the quantization error computed as the average
        distance between each input sample and its best matching unit."""
        return linalg.norm(data - self.quantization(data), axis=1).mean()

    # /def

    def train(
        self,
        data,
        num_iteration: int,
        random_order=False,
        progress: bool = False,
        **kw,
    ) -> None:
        """Trains the SOM.

        Parameters
        ----------
        data : `~numpy.ndarray` or list
            Data matrix.

        num_iteration : int
            Maximum number of iterations (one iteration per sample).
        random_order : bool (default=False)
            If True, samples are picked in random order.
            Otherwise the samples are picked sequentially.

        verbose : bool (default=False)
            If True the status of the training
            will be printed at each iteration.

        """
        iterations = np.arange(num_iteration) % len(data)

        if random_order:
            self._rng.shuffle(iterations)

        with get_progress_bar(progress, len(iterations)) as pbar:
            for t, iteration in enumerate(iterations):
                pbar.update(1)

                self.update(
                    data[iteration],
                    self.winner(data[iteration]),
                    t,
                    num_iteration,
                )

    # /def

    def update(self, x, win, t, max_iteration):
        """Updates the weights of the neurons.

        Parameters
        ----------
        x : np.array
            Current pattern to learn.
        win : tuple
            Position of the winning neuron for x (array or tuple).
        t : int
            Iteration index
        max_iteration : int
            Maximum number of training itarations.

        """
        eta = self._decay_function(self._learning_rate, t, max_iteration)
        # sigma and learning rate decrease with the same rule
        sig = self._decay_function(self._sigma, t, max_iteration)
        # improves the performances
        g = self.neighborhood(win, sig) * eta
        # w_new = eta * neighborhood_function * (x-w)
        self._weights += np.einsum("ij, ijk->ijk", g, x - self._weights)

    # /def

    def winner(self, x):
        """Computes the coordinates of the winning neuron for the sample x."""
        self._activate(x)
        return np.unravel_index(
            self._activation_map.argmin(),
            self._activation_map.shape,
        )

    # /def


# /class


###########################################################


def reorder_visits(
    data: CoordinateType,
    visit_order: T.Sequence,
    start_ind: int,
):
    """Reorder the points from the SOM.

    The SOM does not always keep the starting point at the beginning
    nor even "direction" of the indices. This function can flip the
    index ordering and rotate the indices such that the starting point
    stays at the beginning.

    The rotation is done by snipping and restitching the array at
    `start_ind`. The "direction" is determined by the distance between
    data points at visit_order[start_ind-1] and visit_order[start_ind+1].

    Parameters
    ----------
    data : CoordinateType
        The data.
    visit_order: Sequence
        Index array ordering `data`. Will be flipped and / or rotated
        such that `start_ind` is the 0th element and the next element
        is the closest.
    start_ind : int
        The starting index

    Returns
    -------
    new_visit_order : Sequence
        reordering of `visit_order`

    """
    # index of start_ind in visit_order
    # this needs to be made the first index.
    i = list(visit_order).index(start_ind)

    # snipping and restitching
    # first taking care of the edge cases
    if i == 0:  # AOK
        pass
    elif i == len(visit_order) - 1:  # just reverse.
        i = 0
        visit_order = visit_order[::-1]
    else:  # need to figure out direction before restitching
        back = (data[visit_order[i]] - data[visit_order[i - 1]]).norm()
        forw = (data[visit_order[i + 1]] - data[visit_order[i]]).norm()

        if back < forw:  # things got reversed...
            visit_order = visit_order[::-1]  # flip visit_order
            i = list(visit_order).index(start_ind)  # re-find index

    # do the stitch
    new_visit_order = np.concatenate((visit_order[i:], visit_order[:i]))

    return new_visit_order


# /def

# -------------------------------------------------------------------


def asymptotic_decay(learning_rate: float, iteration: int, max_iter: float):
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


# /def


##############################################################################


def apply_SOM(
    data: CoordinateType,
    *,
    learning_rate: float = 2.0,
    sigma: float = 20.0,
    iterations: int = 10000,
    random_seed: T.Optional[int] = None,
    progress: bool = False,
    reorder: T.Optional[int] = None,
) -> T.Sequence[int]:
    """Apply Self Ordered Mapping to the data.

    Currently only implemented for Spherical.

    Parameters
    ----------
    data : |Representation| or |Frame| or |SkyCoord|
        The data. (lon, lat, distance)
    learning_rate : float (optional, keyword-only)
        (at the iteration t we have
        learning_rate(t) = learning_rate / (1 + t/T)
        where T is #num_iteration/2)
    sigma : float (optional, keyword-only)
        Spread of the neighborhood function, needs to be adequate
        to the dimensions of the map.
        (at the iteration t we have sigma(t) = sigma / (1 + t/T)
        where T is #num_iteration/2)
    iterations : int (optional, keyword-only)
        number of times the SOM is trained.
    random_seed: int (optional, keyword-only)
        Random seed to use. (default=None).
    reorder : int or None (optional, keyword-only)
        If not None (default), the starting index.

    Returns
    -------
    order : sequence

    Other Parameters
    ----------------
    plot : bool (optional, keyword-only)
        Whether to plot the results

    Raises
    ------
    TypeError
        If `data` is not a |Representation| or |Frame| or |SkyCoord|
        and `reorder` is not None.

    """
    # ----------
    # Conversion

    try:
        rep = data.represent_as(coord.SphericalRepresentation)
    except Exception:
        # only need to enforce data-type if reorder not None
        if reorder is not None:
            raise TypeError
    else:
        data = rep._values.view("f8").reshape(-1, len(rep.components))

    # -------
    # SOM

    if conf.use_minisom and HAS_MINISOM:
        # THIRD PARTY
        from minisom import MiniSom as SOM

        USING_MINISOM = True
    else:
        SOM = SelfOrganizingMap
        USING_MINISOM = False

    stream_dims = 1  # streams are 1D

    # length of data, number of features: (x, y, z) or (ra, dec), etc.
    data_len, nfeature = data.shape
    nlattice = data_len // 10

    som = SOM(
        stream_dims,
        nlattice,
        nfeature,
        sigma=sigma,
        learning_rate=learning_rate,
        # decay_function=None,
        neighborhood_function="gaussian",
        topology="rectangular",
        activation_distance="euclidean",
        random_seed=random_seed,
    )
    som.pca_weights_init(data)

    # train the data, preserving order
    if USING_MINISOM:
        som.train(data, iterations, verbose=False, random_order=False)
    else:
        som.train(
            data,
            iterations,
            verbose=False,
            random_order=False,
            progress=progress,
        )

    # get the ordering by "vote" of the Prototypes
    visit_order = order_data_from_SOM(som, data)

    # Reorder
    if reorder is not None:
        order = reorder_visits(rep, visit_order, start_ind=reorder)
    else:  # don't reorder
        order = visit_order

    return order, som


# /def

# -------------------------------------------------------------------


def order_data_from_SOM(som, data):
    """Order data from SOM in 2D.

    Parameters
    ----------
    som : `~minisom.MiniSom` or `~trackstream.preprocess.som.SOM`
    data : (N, M) array_like

    Returns
    -------
    `~numpy.ndarray`

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
    data_len, nfeature = data.shape
    nlattice = som._neigy[-1] + 1

    # vector from one point to next  (nlattice-1, nfeature)
    lattice_points = som._weights
    p1 = lattice_points[0, :-1, :]
    p2 = lattice_points[0, 1:, :]
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
    # or inide, but on the convex side of a segment junction
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
        else:  # TODO! find and fix the ordering mistake
            phi1 = np.arctan2(*viip1[i // 2 - 1, :2])
            phim2 = np.arctan2(*-viip1[i // 2, :2])
            phi = np.arctan2(*data[rowi, :2].T)

            # detect if in branch cut territory
            if (np.pi / 2 <= phi1) & (-np.pi <= phim2) & (phim2 <= -np.pi / 2):
                phi1 -= 2 * np.pi
                phi -= 2 * np.pi

            rowsorter = np.argsort(phi) if phim2 < phi1 else np.argsort(-phi)

        ordering[counter : counter + numrows] = rowi[rowsorter]
        counter += numrows

    return ordering


# /def

# -------------------------------------------------------------------


def apply_SOM_repeat(
    data: DataType,
    random_seeds: T.Optional[T.Sequence[int]],
    *,
    dims: int = 1,
    learning_rate: float = 0.8,
    sigma: float = 4.0,
    iterations: int = 10000,
    reorder: T.Optional[T.Sequence] = True,
    plot: bool = False,
    _tqdm: bool = True,
) -> T.Sequence[int]:
    """SOM Preprocess.

    Parameters
    ----------
    data : BaseCoordinateFrame instance
    random_seeds: Sequence
        Random seeds to use.

    dims : int (optional, keyword-only)
    learning_rate : float (optional, keyword-only)
    sigma : float (optional, keyword-only)
        Spread of the neighborhood function, needs to be adequate
        to the dimensions of the map.
        (at the iteration t we have sigma(t) = sigma / (1 + t/T)
        where T is #num_iteration/2)
    neighborhood_function : str (optional, keyword-only)
    iterations : int (optional, keyword-only)

    reorder : Sequence (optional, keyword-only)
        If not None, the starting index.
    plot : bool (optional, keyword-only)

    Returns
    -------
    orders : Sequence
        Shape (len(`random_seeds`), len(`data`))

    """
    if isinstance(
        data,
        (coord.SkyCoord, coord.BaseCoordinateFrame, coord.BaseRepresentation),
    ):
        rep = data.represent_as(coord.CartesianRepresentation)
        data = rep._values.view("f8").reshape(-1, len(rep.components))
    else:
        raise TypeError

    # -------

    nrows = len(random_seeds)
    orders = np.empty((nrows, len(data)), dtype=int)

    iterator = tqdm(random_seeds) if _tqdm else random_seeds
    for i, seed in enumerate(iterator):
        visit_order = apply_SOM(
            data,
            learning_rate=learning_rate,
            sigma=sigma,
            iterations=iterations,
            random_seed=seed,
            reorder=None,
            plot=False,
        )

        if reorder is not None:
            order = reorder_visits(rep, visit_order, start_ind=reorder)

        else:
            order = visit_order

        orders[i] = order

    # /for

    # ---------------
    # if plot:
    #     fig, ax = plt.subplots(figsize=(10, 9))
    #     ax.scatter(
    #         data[:, 0],
    #         data[:, 1],
    #         c="k",
    #         vmax=len(data),
    #         cmap="plasma",
    #         label="data",
    #     )
    #     for order in orders:
    #         ax.plot(data[order][:, 0], data[order][:, 1], c="gray", alpha=0.5)
    #         ax.scatter(data[order][0, 0], data[order][-1, 1], c="g")
    #     fig.legend(loc="upper left")
    #     fig.tight_layout()
    #     plt.show()
    # # /if

    # ---------------

    return orders


# /def


# -------------------------------------------------------------------


def make_transition_matrix(orders: T.Sequence[T.Sequence]):
    """Make Transition Matrix from SOM-derived orders.

    The SOM-derived orders can vary with the random seed. To account
    for the non-determinism we construct a transition matrix that
    gives the probability of going from index *i* to *j*. For this,
    we need many SOM-derived orders.

    Parameters
    ----------
    orders: list of lists
        Shape (N, len(data))

    Returns
    -------
    trmat : `~scipy.sparse.lil_matrix`
        The sparse transition matrix. All rows and columns sum to 1.
        Row index is for *i*, column index for *j*.
        Shape (len(data), len(data)).

    See Also
    --------
    `~draw_ordering`

    """
    nelt = orders.shape[1]  # number of elements in matrix

    # empty and sparse transition matrix
    trmat = sparse.lil_matrix((nelt, nelt), dtype=float)

    # fill in transition pairs, counting number of occurrences.
    for i in range(nelt):  # TODO vectorize
        visited, counts = np.unique(orders[:, i], return_counts=True)
        trmat[i, visited] += counts

    # convert from count to probability by dividing by number of orderings.
    trmat /= orders.shape[0]

    return trmat


# /def


def draw_ordering(
    trmat,
    num: int = 1,
    rng: T.Optional[np.random.Generator] = None,
):
    """Draw ordering(s) from transition matrix.

    Parameters
    ----------
    trmat : ndarray
        Transition matrix. square.
    num : int, optional
        number of orderings to draw
    rng : Generator instance, optional
        Random number generator.

    Returns
    -------
    orders : `~numpy.ndarray`
        Shape (`num`, len(`trmat`)).

    Notes
    -----
    The orderings are drawn by iterating through each row *i* of `trmat`,
    which encodes the index in the data. The columns *j* of each row
    encode the next index in the ordering, with some probability
    :math:`p_{i,j}`. The probabilities are accumulated (summing to 1)
    and a random number is uniformly generated to select the *j* index.

    """
    rng = rng if rng is not None else np.random.default_rng()
    size = trmat.shape[0]
    shape = (size, num)

    # The selection function for which index *j* to select in each row *i*
    # flipped to get half-open bound (0, 1]
    alpha = rng.uniform(low=1.0, high=0.0, size=shape)
    orders = np.full(shape, -1, dtype=int)

    # iterating through rows of trmat, selecting the transition pair.
    for i, (inds, probs) in enumerate(zip(trmat.rows, trmat.data)):
        if probs[0] == 1.0:  # easy exit
            orders[i] = inds[0]
            continue
        # select among steps by probs
        sel = np.cumsum(probs) >= alpha[i][:, None]  # shape st rows=num
        # get first index for which sel is True
        orders[i] = [inds[i] for i in np.argmax(sel, axis=1)]  # for rows

    # TODO check no -1s left from the np.full
    if np.any(orders == -1):
        raise Exception("Something went wrong in `draw_ordering`")

    return orders.T


# /def


# -------------------------------------------------------------------


def draw_most_probable_ordering(trmat):
    """Draw most probably ordering from transition matrix.

    Parameters
    ----------
    trmat : ndarray
        Transition matrix. square.
    num : int, optional
        number of orderings to draw
    rng : Generator instance, optional
        Random number generator.

    Returns
    -------
    best_order : `~numpy.ndarray`
        Same length as `trmat`.

    Notes
    -----
    The orderings are drawn by iterating through each row *i* of `trmat`,
    which encodes the index in the data. The columns *j* of each row
    encode the next index in the ordering, with some probability
    :math:`p_{i,j}`. The probabilities are accumulated (summing to 1)
    and a random number is uniformly generated to select the *j* index.

    """
    best_order = np.full(trmat.shape[0], -1, dtype=int)

    # TODO vectorize this for loop
    # iterating through rows of trmat, selecting the transition pair.
    for i, (inds, probs) in enumerate(zip(trmat.rows, trmat.data)):
        # select among steps by most probable
        best_order[i] = inds[np.argmax(probs)]

    return best_order


# /def


# -------------------------------------------------------------------


# def preprocess(
#     data: CoordinateType,
#     start_point: T.Sequence,
#     N_repeats: T.Union[int, T.Sequence] = 10,
#     *,
#     iterations: int = int(3e3),
#     learning_rate: float = 1.5,
#     sigma: float = 15,
#     plot: bool = True,
#     _tqdm: bool = True,
# ):
#     """Preprocess.
#
#     Parameters
#     ----------
#     data : CoordinateType
#     start_point : Sequence
#         The coordinates of the starting point.
#     N_repeats : int or Sequence, optional
#         The number of times to do the SOM, setting the random seeds.
#         If sequence, interpreted as the random seeds, where number of
#         repeats is length of sequence.
#
#     iterations : int (optional, keyword-only)
#         Number of iterations for each SOM.
#     learning_rate : float (optional, keyword-only)
#         SOM learning rate.
#     sigma : float (optional, keyword-only)
#         SOM sigma.
#     plot : bool (optional, keyword-only)
#         Whether to plot the preprocessing results.
#
#     Returns
#     -------
#     visit_orders
#
#     Other Parameters
#     ----------------
#     _tqdm : bool (optional, keyword-only)
#         Whether to use tqdm progress bar
#
#     """
#     # ---------------
#     # starting point
#     start_point = np.asanyarray(start_point)  # lets units through
#
#     start_ind: int
#     start_point, start_ind = find_closest_point(data, start_point)
#
#     data = set_starting_point(data, start_ind)
#     start_ind: int = 0
#
#     # ---------------
#     # SOM
#
#     if isinstance(N_repeats, int):
#         random_seeds = np.arange(0, N_repeats, 1)
#     else:
#         random_seeds = N_repeats
#
#     visit_orders = apply_SOM_repeat(
#         data,
#         random_seeds=random_seeds,
#         iterations=iterations,
#         learning_rate=learning_rate,
#         sigma=sigma,
#         plot=plot,
#         reorder=start_ind,
#         _tqdm=_tqdm,
#     )
#
#     # ---------------
#     # Transition Matrix
#
#     trmat = make_transition_matrix(visit_orders)
#
#     # ---------------
#
#     return preprocessed(
#         data=data,
#         trmat=trmat,
#         visit_orders=visit_orders,
#         start_point=start_point,
#     )
#
#
# # /def


##############################################################################
# END
