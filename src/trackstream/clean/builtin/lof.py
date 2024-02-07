"""Local Outlier Factor (LOF) method of detecting stream outliears."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import numpy as np
from scipy.spatial import KDTree

from trackstream.clean.base import OutlierDetectorBase

__all__: list[str] = []

if TYPE_CHECKING:
    from numpy.typing import NDArray

    from trackstream._typing import N1, NDFloat


KDT = TypeVar("KDT")


class KDTreeLOFBase(OutlierDetectorBase, Generic[KDT], register=False):
    """Abstract Base Class for Kernel Density Tree Local Outlier Factor."""

    kdtree_kw: dict[str, Any]
    tree: KDT

    def __init__(self, **kdtree_kw: Any) -> None:
        object.__setattr__(self, "kdtree_kw", kdtree_kw)

    @abstractmethod
    def fit(self, data: NDFloat[N1], /) -> None:
        """Fit the outlier detection method to the training data.

        Parameters
        ----------
        data : (N, D) ndarray[float], positional-only
            The training data. Rows are distinct objects, e.g stars, columns are
            features, e.g. ``D``-dimensional coordinates of the stars.

        Returns
        -------
        None

        """
        super().fit(data)

    @abstractmethod
    def predict(self, data: NDFloat[N1], /, threshold: float = 2, *, k: int = 10, **query_kw: Any) -> NDArray[np.bool_]:
        """Predict stream outliers given this fit model.

        Parameters
        ----------
        self : KDTreeLOFBase
            The cleaning object.
        data : (N, D) ndarray, positional-only
            The data. Rows are distinct objects, e.g stars, columns are
            features, e.g. ``D``-dimensional coordinates of the stars.
        threshold : float, optional
            The threshold for the LOF to consider a point an outlier.
        k : int, optional keyword-only
            The number of nearest neighbors to use.
        **query_kw : Any
            Keyword arguments.

        Returns
        -------
        (N,) ndarray[bool]
            The predicted labels for each row index in ``X`` whether the data
            point is an outlier (`True`) or not (`False`).

        Raises
        ------
        NotFittedError

        """
        return super().predict(data, threshold=threshold, k=k, **query_kw)


class ScipyKDTreeLOF(KDTreeLOFBase["KDTree"]):
    """LOF using a KDTree from `scipy.spatial`."""

    def fit(self, data: NDFloat[N1], /) -> None:
        """Fit the outlier detection method to the training data.

        Parameters
        ----------
        data : (N, D) ndarray[float], positional-only
            The training data. Rows are distinct objects, e.g stars, columns are
            features, e.g. ``D``-dimensional coordinates of the stars.

        Returns
        -------
        None

        """
        super().fit(data)
        object.__setattr__(self, "tree", KDTree(data, **self.kdtree_kw))

    def predict(self, data: NDFloat[N1], /, threshold: float = 2, *, k: int = 10, **query_kw: Any) -> NDArray[np.bool_]:
        """Predict stream outliers given this fit model.

        Parameters
        ----------
        self : KDTreeLOFBase
            The cleaning object.
        data : (N, D) ndarray, positional-only
            The data. Rows are distinct objects, e.g stars, columns are
            features, e.g. ``D``-dimensional coordinates of the stars.
        threshold : float, optional
            The threshold for the LOF to consider a point an outlier.
        k : int, optional keyword-only
            The number of nearest neighbors to use.
        **query_kw : Any
            Keyword arguments.

        Returns
        -------
        (N,) ndarray[bool]
            The predicted labels for each row index in ``X`` whether the data
            point is an outlier (`True`) or not (`False`).

        Raises
        ------
        ValueError
            If `k` <= 1.

        """
        if k == 1:
            msg = "k must be > 1"
            raise ValueError(msg)

        # Query for k nearest
        dx, idx_knn = self.tree.query(data, k=k, **query_kw)

        # Get the distance of the most-distant neighbor
        radius = np.array(dx)[:, -1]
        # Calculate the local reachability density
        lrd = np.mean(np.maximum(dx, radius[idx_knn]), axis=1)

        # Calculate the outlier score
        invdens = 1.0 / lrd  # inverse of density
        outlier_score = lrd * np.sum(invdens[idx_knn], axis=1) / k

        return cast("np.ndarray[Any, np.dtype[np.bool_]]", outlier_score > threshold)
