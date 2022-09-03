# see LICENSE.rst

"""Detect Outliers from Stream Data."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

# THIRD PARTY
import numpy as np
from scipy.spatial import KDTree

# LOCAL
from trackstream.clean.base import OutlierDetectorBase

if TYPE_CHECKING:
    # THIRD PARTY
    from numpy.typing import NDArray

    # LOCAL
    from trackstream._typing import N1, NDFloat

__all__: list[str] = []


##############################################################################
# PARAMETERS

KDT = TypeVar("KDT")


##############################################################################
# CODE
##############################################################################


class KDTreeLOFBase(OutlierDetectorBase, Generic[KDT], register=False):
    """Abstract Base Class for Kernel Density Tree Local Outlier Factor."""

    kdtree_kw: dict[str, Any]
    tree: KDT

    def __init__(self, **kdtree_kw: Any) -> None:
        object.__setattr__(self, "kdtree_kw", kdtree_kw)

    @abstractmethod
    def fit(self, data: NDFloat[N1], /) -> None:
        pass

    @abstractmethod
    def predict(self, X: NDFloat[N1], /, threshold: float = 2, *, k: int = 10, **query_kw: Any) -> NDArray[np.bool_]:
        # if k <= 1:
        #     raise ValueError
        pass


class scipyKDTreeLOF(KDTreeLOFBase["KDTree"]):
    def fit(self, data: NDFloat[N1], /) -> None:
        super().fit(data)
        object.__setattr__(self, "tree", KDTree(data, **self.kdtree_kw))

    def predict(self, X: NDFloat[N1], /, threshold: float = 2, *, k: int = 10, **query_kw: Any) -> NDArray[np.bool_]:
        if k == 1:
            raise ValueError("k must be > 1")

        # Query for k nearest
        dx, idx_knn = self.tree.query(X, k=k, **query_kw)

        # Get the distance of the most-distant neighbor
        radius = np.array(dx)[:, -1]
        # Calculate the local reachability density
        lrd = np.mean(np.maximum(dx, radius[idx_knn]), axis=1)

        # Calculate the outlier score
        invdens = 1.0 / lrd  # inverse of density
        outlier_score = lrd * np.sum(invdens[idx_knn], axis=1) / k

        return cast(np.ndarray[Any, np.dtype[np.bool_]], outlier_score > threshold)
