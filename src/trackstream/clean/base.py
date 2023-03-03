"""Detect outliers from stream data."""


from __future__ import annotations

from abc import ABCMeta, abstractmethod
import inspect
from typing import TYPE_CHECKING, Any, ClassVar

from trackstream.utils.exceptions import NotFittedError

if TYPE_CHECKING:
    from numpy import bool_
    from numpy.typing import NDArray

    from trackstream._typing import N1, N2, NDFloat

__all__: list[str] = []


##############################################################################
# PARAMETERS


OUTLIER_DETECTOR_CLASSES: dict[str, type[OutlierDetectorBase]] = {}
# Registry of outlier-detection classes. See ``OutlierDetectorBase`` for the
# registration details.


##############################################################################
# CODE
##############################################################################


class OutlierDetectorBase(metaclass=ABCMeta):
    """Abstract Base Class for Outlier Detection.

    When subclasssing, this class accepts the optional keyword argument
    'register' for registering the subclass in
    `~trackstream.clean.base.OUTLIER_DETECTOR_CLASSES`. If registered,
    :meth:`StreamArm.mask_outliers` and similar functions can specify the
    outlier detection method by the class' qualitative name, rather than
    requiring an instance of the class.
    """

    _predict__signature__: ClassVar[inspect.Signature]
    # cached signature of the ``.predict()`` method for faster signature binding
    # in ``fit_predict``.

    def __init_subclass__(cls, *, register: bool = True) -> None:
        """Initialize subclass.

        Parameters
        ----------
        register : bool, optional
            Whether to register this subclass in
            `~trackstream.clean.base.OUTLIER_DETECTOR_CLASSES`, by default
            `True`. If registered, :meth:`StreamArm.mask_outliers` and similar
            functions can specify the outlier detection method by the class'
            qualitative name, rather than requiring an instance of the class.

        Raises
        ------
        TypeError
            If a class with the same ``__qualname__`` is already registered.
        """
        super().__init_subclass__()

        # Optionally register class
        if register:
            qn = cls.__qualname__
            if qn in OUTLIER_DETECTOR_CLASSES:
                msg = f"{qn} is already registered; to redefine, first remove from 'OUTLIER_DETECTOR_CLASSES'."
                raise TypeError(msg)
            OUTLIER_DETECTOR_CLASSES[qn] = cls

        # cache the signature of ``.predict()`` for use in ``fit_predict``.
        cls._predict__signature__ = inspect.signature(cls.predict)

    def __init__(self, **_: Any) -> None:
        # Flag if the outlier detector has been fit.
        self._isfit: bool = False

    @abstractmethod
    def fit(self, data: NDFloat[N1], /) -> None:
        """Fit the outlier detection method to the training data.

        Parameters
        ----------
        data : (N, D) ndarray[float], positional-only
            The training data. Rows are distinct objects, e.g stars, columns are
            features, e.g. ``D``-dimensional coordinates of the stars.
        """
        self._isfit = True

    @abstractmethod
    def predict(self, data: NDFloat[N1], /, **kwargs: Any) -> NDArray[bool_]:
        """Predict stream outliers given this fit model.

        Parameters
        ----------
        data : (N, D) ndarray, positional-only
            The data. Rows are distinct objects, e.g stars, columns are
            features, e.g. ``D``-dimensional coordinates of the stars.
        **kwargs : Any
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
        if not self._isfit:
            raise NotFittedError

    def fit_predict(
        self,
        fit_data: NDFloat[N1],
        predict_data: NDFloat[N2] | None = None,
        /,
        **kwargs: Any,
    ) -> NDArray[bool_]:
        """Predict if is point in data is an outlier.

        Parameters
        ----------
        fit_data : (N, D) ndarray, positional-only
            The training data.
        predict_data : (N, D) ndarray or None, positional-only
            The data . Rows are distinct objects, e.g stars, columns are
            features, e.g. ``D``-dimensional coordinates of the stars.
            If `None` (default), ``fit_data`` is used.
        **kwargs : Any
            Keyword arguments.

        Returns
        -------
        ndarray[bool]
            The predicted labels for each row index in ``X`` whether the data
            point is an outlier (`True`) or not (`False`).
        """
        # Fit
        self.fit(fit_data)
        # Predict
        x = fit_data if predict_data is None else predict_data
        pba = self._predict__signature__.bind(None, x, **kwargs)  # None -> self b/c unbound method
        # call predict method, skipping `self` since instance method.
        return self.predict(*pba.args[1:], **pba.kwargs)
