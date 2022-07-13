# see LICENSE.rst

"""Detect Outliers from Stream Data."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import inspect
from abc import ABCMeta, abstractmethod
from typing import Any, cast

# THIRD PARTY
import numpy as np
from numpy import ndarray

##############################################################################
# PARAMETERS

OUTLIER_DETECTOR_CLASSES: dict[str, type[OutlierDetectorBase]] = {}


##############################################################################
# CODE
##############################################################################


class OutlierDetectorBase(metaclass=ABCMeta):
    """Abstract Base Class for Outlier Detection."""

    def __init_subclass__(cls, register: bool = True) -> None:
        # Register class
        if register:
            qn = cls.__qualname__
            if qn in OUTLIER_DETECTOR_CLASSES:
                raise TypeError(
                    f"{qn} is already registered;" " to redefine, first remove from 'OUTLIER_DETECTOR_CLASSES'."
                )
            OUTLIER_DETECTOR_CLASSES[qn] = cls

        cls.fit.__signature__ = inspect.signature(cls.fit)
        cls.predict.__signature__ = inspect.signature(cls.predict)

    def __init__(self, **_: Any) -> None:
        pass

    @abstractmethod
    def fit(self, data: ndarray, /, **kwargs: Any) -> None:
        pass

    @abstractmethod
    def predict(self, X: ndarray, /, **kwargs: Any) -> ndarray:
        pass

    def fit_predict(self, data: ndarray, X: ndarray | None = None, /, **kwargs: Any) -> ndarray:
        """Predict if is point in data is an outier.

        Parameters
        ----------
        data : ndarray, positional-only
            The data to fit.
        X : ndarray or None, positional-only
            The data to predict if it's an outlier.
            If `None` (default), ``data`` is used.

        Returns
        -------
        ndarray[bool]
            `True` if an outlier, `False` if not.
        """
        # Fit
        self.fit(data)
        # Predict
        X = data if X is None else X
        pba = self.predict.__signature__.bind(None, X, **kwargs)  # None -> self b/c unbound method
        # call predict method, skipping `self` since instance method.
        return self.predict(*pba.args[1:], **pba.kwargs)


class KDTreeLOFBase(OutlierDetectorBase, register=False):
    """Abstract Base Class for Kernel Density Tree Local Outlier Factor."""

    def __init__(self, **kdtree_kw: Any) -> None:
        self.kdtree_kw = kdtree_kw

    @abstractmethod
    def fit(self, data: ndarray, /) -> None:
        pass

    @abstractmethod
    def predict(self, X: ndarray, /, threshold: float = 2, *, k: int = 10, **query_kw: Any) -> ndarray:
        if k <= 1:
            raise ValueError


class scipyKDTreeLOF(KDTreeLOFBase):
    def fit(self, data: ndarray, /) -> None:
        # THIRD PARTY
        from scipy.spatial import KDTree

        super().fit(data)
        self.tree = KDTree(data, **self.kdtree_kw)

    def predict(self, X: ndarray, /, threshold: float = 2, *, k: int = 10, **query_kw: Any) -> ndarray:
        # Query for k nearest
        dx, idx_knn = self.tree.query(X, k=k, **query_kw)
        dx = cast(ndarray, dx)

        # Get the distance of the most-distant neighbor
        radius = dx[:, -1]
        # Calculate the local reachability density
        lrd = np.mean(np.maximum(dx, radius[idx_knn]), axis=1)

        # Calculate the outlier score
        invdens = 1.0 / lrd  # inverse of density
        outlier_score = lrd * np.sum(invdens[idx_knn], axis=1) / k

        return outlier_score > threshold
