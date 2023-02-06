"""SOM base class."""

from __future__ import annotations

# STDLIB
import copy
import warnings
from abc import abstractmethod
from dataclasses import dataclass
from functools import singledispatchmethod
from math import pi
from typing import Any, ClassVar, final

# THIRD PARTY
import astropy.units as u
import numpy as np
from numpy import exp, ndarray, power
from numpy.lib.recfunctions import structured_to_unstructured
from numpy.random import Generator, default_rng

# LOCAL
from trackstream.stream.core import StreamArm
from trackstream.stream.plural import StreamArmsBase
from trackstream.track.fit.exceptions import EXCEPT_NO_KINEMATICS
from trackstream.track.fit.som.utils import project_data_on_som
from trackstream.track.fit.utils import FrameInfo
from trackstream.utils.coord_utils import f2q
from trackstream.utils.pbar import get_progress_bar

__all__: list[str] = []

__credits__ = "MiniSom"

##############################################################################
# PARAMETERS

warnings.filterwarnings("ignore", message="sigma is too high for the dimension of the map")


@final
@dataclass(frozen=True)
class SOMInfo(FrameInfo):
    """SOM info class."""

    REGISTRY: ClassVar[dict[type, SOMInfo]] = {}


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class SOM1DBase:
    """Initializes a Self-Organizing Map.

    Inspired by the design of [MiniSom]_.

    Parameters
    ----------
    prototypes : (N, F) ndarray
        The N prototype vectors of F features.

    sigma : float, optional (default=1.0)
        Spread of the neighborhood function, needs to be adequate to the
        dimensions of the map. (at the iteration t we have sigma(t) = sigma / (1
        + t/T) where T is #num_iteration/2)
    learning_rate : initial learning rate
        (at the iteration t we have learning_rate(t) = learning_rate / (1 + t/T)
        where T is #num_iteration/2)
    rng : int, optional keyword-only (default=None)
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

    prototypes: ndarray
    rng: Generator = default_rng()
    sigma: float = 0.3
    learning_rate: float = 0.3

    info: ClassVar[SOMInfo]  # for typing

    def __post_init__(self) -> None:
        # prototypes cannot be a Quantity
        if isinstance(self.prototypes, u.Quantity):
            prototypes = self.prototypes.value
            object.__setattr__(self, "prototypes", prototypes)

        # initial prototypes
        self._init_prototypes: ndarray
        _init_prototypes = self.prototypes.copy()
        _init_prototypes.flags.writeable = False
        object.__setattr__(self, "_init_prototypes", _init_prototypes)

        self._lattice: ndarray
        object.__setattr__(self, "_lattice", np.arange(self.nlattice, dtype=float))

        if self.sigma >= 1 or self.sigma >= self.nlattice:
            warnings.warn("sigma is too high for the dimension of the map")

    @property
    def nlattice(self) -> int:
        """Number of lattice points."""
        return self.prototypes.shape[0]

    @property
    def nfeature(self) -> int:
        """Number of features."""
        return self.prototypes.shape[1]

    @staticmethod
    @abstractmethod
    def _make_prototypes_from_binned_data(
        data: ndarray,
        /,
        nlattice: int,
        *,
        byphi: bool = False,
        maxsep: ndarray | None = None,
        **_: Any,
    ) -> ndarray:
        raise NotImplementedError

    # =======================================================

    @singledispatchmethod
    @classmethod
    def from_format(
        cls,
        arm: object,
        /,
        kinematics: bool | None = None,
        *,
        nlattice: int | None = None,
        sigma: float = 0.1,
        learning_rate: float = 0.3,
        rng: Generator | int | None = None,
        prototype_kw: dict[str, Any] | None = None,
    ) -> Any:  # https://github.com/python/mypy/issues/11727
        """Initialize a SOM from an object.

        Parameters
        ----------
        arm : object, positional-only
            The object to initialize from.
        kinematics : bool, optional
            Whether to use kinematics. If `None`, will use kinematics if available.
        nlattice : int | None, optional
            Number of lattice points.
        sigma : float | None, optional
            Spread of the neighborhood function, needs to be adequate to the dimensions of the map.
        learning_rate : float | None, optional
            Initial learning rate.
        rng : int, optional
            Random seed to use.
        prototype_kw : dict | None, optional
            Keyword arguments to pass to the prototype initialization function.

        Returns
        -------
        SOM1DBase
        """
        raise NotImplementedError("not dispatched")  # noqa: EM101

    @from_format.register(StreamArm)
    @classmethod
    def _from_format_streamarm(
        cls,
        arm: StreamArm,
        /,
        kinematics: bool | None = None,
        *,
        nlattice: int | None = None,
        sigma: float = 0.1,
        learning_rate: float = 0.3,
        rng: Generator | int | None = None,
        prototype_kw: dict[str, Any] | None = None,
    ) -> SOM1DBase:
        # flags
        if kinematics is None:
            kinematics = arm.has_kinematics
        elif kinematics is True and not arm.has_kinematics:
            raise EXCEPT_NO_KINEMATICS
        D = len(cls.info.components(kinematics=kinematics))  # index for slicing

        prototype_kw = copy.copy(prototype_kw) if prototype_kw is not None else {}
        if prototype_kw.setdefault("maxsep") is not None:
            prototype_kw["maxsep"] = prototype_kw["maxsep"].to_value(cls.info.units[0][0])
            # it's just spatial

        # Determine number of lattice points, within [5, 100]
        if nlattice is None:
            nlattice = min(max(len(arm.coords) // 50, 5), 20)

        # make prototypes
        crds = arm.coords
        crds.representation_type = cls.info.representation_type
        crds.differential_type = cls.info.differential_type
        vs = structured_to_unstructured(f2q(crds).to_value(cls.info.units))[:, :D]

        prototypes = cls._make_prototypes_from_binned_data(vs, nlattice, **prototype_kw)

        # Make SOM
        return cls(
            prototypes=prototypes,
            sigma=sigma,
            learning_rate=learning_rate,
            rng=default_rng(rng),
        )

    @from_format.register(StreamArmsBase)
    @classmethod
    def _from_format_streamarmsbase(
        cls,
        arms: StreamArmsBase,
        /,
        kinematics: bool | None = None,
        *,
        nlattice: int | None = None,
        sigma: float = 0.1,
        learning_rate: float = 0.3,
        rng: Generator | int | None = None,
        prototype_kw: dict[str, Any] | None = None,
    ) -> dict[str, SOM1DBase]:
        out: dict[str, SOM1DBase] = {}
        for k, arm in arms.items():
            out[k] = cls.from_format(
                arm,
                kinematics=kinematics,
                nlattice=nlattice,
                sigma=sigma,
                learning_rate=learning_rate,
                rng=rng,
                prototype_kw=prototype_kw,
            )
        return out

    # ===============================================================

    @property
    def init_prototypes(self) -> ndarray:
        """Initial prototypes."""
        return self._init_prototypes

    # ---------------------------------------------------------------
    # fitting

    @abstractmethod
    def _activation_distance(self, x: ndarray, w: ndarray) -> ndarray:
        pass

    @abstractmethod
    def _update(self, x: ndarray, t: int, max_iteration: int) -> None:
        pass

    @final
    def fit(
        self,
        data: ndarray,
        num_iteration: int = int(1e5),
        *,
        random_order: bool = False,
        progress: bool = False,
    ) -> None:
        """Trains the SOM.

        Parameters
        ----------
        data : (N, F) ndarray
            N data points with F features.

        num_iteration : int
            Maximum number of iterations (one iteration per sample).
            Must be greater than the length of the data.
        random_order : bool (default=False)
            If True, samples are picked in random order.
            Otherwise the samples are picked sequentially.

        progress : bool (default=False)
            If True, show a progress bar

        Returns
        -------
        None
        """
        # Number of cycles through the data
        iterations = np.arange(num_iteration) % len(data)
        # Optionally randomize the cycles
        if random_order:
            self.rng.shuffle(iterations)

        # Fit the data by sequential update
        with get_progress_bar(display=progress, total=len(iterations)) as pbar:
            for t, i in enumerate(iterations):
                pbar.update(1)

                self._update(data[i], t, num_iteration)

    def _neighborhood(self, c: int, sigma: float) -> ndarray:
        """Returns a Gaussian centered in c.

        This is in the lattice space, so Cartesian vs UnitSpherical does not
        matter.
        """
        d = 2 * pi * sigma**2
        ay: ndarray = exp(-power(self._lattice - self._lattice.T[c], 2) / d).T
        return ay  # the external product gives a matrix

    def _best_matching_unit_index(self, x: ndarray, /) -> int:
        """Computes the coordinates of the best prototype for the sample.

        Parameters
        ----------
        x : (D,) ndarray, positional-only

        Returns
        -------
        int
            The index of the best-matching prototype.
        """
        activation_map = self._activation_distance(x, self.prototypes)
        return int(activation_map.argmin())

    # ---------------------------------------------------------------
    # Predicting structure

    def predict(self, data: ndarray, /) -> tuple[ndarray, ndarray]:
        """Order data from SOM in 2+N Dimensions.

        Parameters
        ----------
        data : ndarray, positional-only
            This will generally be the same data used to train the SOM.

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
        # Prediction of ordering.
        projdata, ordering = project_data_on_som(self.prototypes, data)
        projdata = projdata[ordering]
        return projdata, ordering

    @final
    def fit_predict(
        self,
        data: ndarray,
        /,
        num_iteration: int = int(1e5),
        *,
        random_order: bool = False,
        progress: bool = False,
    ) -> tuple[ndarray, ndarray]:
        """Fit then predict.

        Returns
        -------
        projdata : ndarray
            Ordered.
        order : ndarray
            Array to order ``data``.

        See Also
        --------
        trackstream.SelfOrganizingMap1D.fit
        trackstream.SelfOrganizingMap1D.predict
        """
        self.fit(data, num_iteration=num_iteration, random_order=random_order, progress=progress)
        projdata, order = self.predict(data)
        return projdata, order
