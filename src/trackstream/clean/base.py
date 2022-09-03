# see LICENSE.rst

"""Detect Outliers from Stream Data."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import inspect
from abc import ABCMeta, abstractmethod
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    # THIRD PARTY
    from numpy import bool_
    from numpy.typing import NDArray

    # LOCAL
    from trackstream._typing import N1, N2, NDFloat

__all__: list[str] = []


##############################################################################
# PARAMETERS


OUTLIER_DETECTOR_CLASSES: dict[str, type[OutlierDetectorBase]] = {}


##############################################################################
# CODE
##############################################################################


class OutlierDetectorBase(metaclass=ABCMeta):
    """Abstract Base Class for Outlier Detection."""

    _predict__signature__: ClassVar[inspect.Signature]

    def __init_subclass__(cls, register: bool = True) -> None:
        super().__init_subclass__()

        # Register class
        if register:
            qn = cls.__qualname__
            if qn in OUTLIER_DETECTOR_CLASSES:
                raise TypeError(
                    f"{qn} is already registered; to redefine, first remove from 'OUTLIER_DETECTOR_CLASSES'."
                )
            OUTLIER_DETECTOR_CLASSES[qn] = cls

        cls._predict__signature__ = inspect.signature(cls.predict)

    def __init__(self, **_: Any) -> None:
        pass

    @abstractmethod
    def fit(self, data: NDFloat[N1], /) -> None:
        pass

    @abstractmethod
    def predict(self, X: NDFloat[N1], /, **kwargs: Any) -> NDArray[bool_]:
        pass

    def fit_predict(self, data: NDFloat[N1], X: NDFloat[N2] | None = None, /, **kwargs: Any) -> NDArray[bool_]:
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
        x = data if X is None else X
        pba = self._predict__signature__.bind(None, x, **kwargs)  # None -> self b/c unbound method
        # call predict method, skipping `self` since instance method.
        return self.predict(*pba.args[1:], **pba.kwargs)
