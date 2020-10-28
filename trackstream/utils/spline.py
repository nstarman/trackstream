# -*- coding: utf-8 -*-

"""Splines, with :mod:`~astropy.units`."""

__all__ = [
    "InterpolatedUnivariateSplinewithUnits",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import astropy.units as u
import numpy as np
from scipy.interpolate import InterpolatedUnivariateSpline

##############################################################################
# PARAMETERS

UnitType = T.TypeVar("UnitType", bound=u.UnitBase)
QuantityType = T.TypeVar("QuantityType", bound=u.Quantity)

_BBoxType = T.List[T.Optional[QuantityType]]

##############################################################################
# CODE
##############################################################################


class InterpolatedUnivariateSplinewithUnits(InterpolatedUnivariateSpline):
    """1-D interpolating spline for a given set of data points, with units.

    Fits a spline y = spl(x) of degree `k` to the provided `x`, `y` data.
    Spline function passes through all provided points. Equivalent to
    `UnivariateSpline` with s=0.

    Parameters
    ----------
    x : (N,) |Quantity| array_like
        Input dimension of data points -- must be strictly increasing
    y : (N,) |Quantity| array_like
        input dimension of data points
    w : (N,) |Quantity| array_like, optional
        Weights for spline fitting.  Must be positive.  If None (default),
        weights are all equal.
    bbox : (2,) |Quantity| array_like, optional
        2-sequence specifying the boundary of the approximation interval. If
        None (default), ``bbox=[x[0], x[-1]]``.
    k : int, optional
        Degree of the smoothing spline.  Must be 1 <= `k` <= 5.
    ext : int or str, optional
        Controls the extrapolation mode for elements
        not in the interval defined by the knot sequence.

        * if ext=0 or 'extrapolate', return the extrapolated value.
        * if ext=1 or 'zeros', return 0
        * if ext=2 or 'raise', raise a ValueError
        * if ext=3 of 'const', return the boundary value.

        The default value is 0.

    check_finite : bool, optional
        Whether to check that the input arrays contain only finite numbers.
        Disabling may give a performance gain, but may result in problems
        (crashes, non-termination or non-sensical results) if the inputs
        do contain infinities or NaNs.
        Default is False.

    x_unit, y_unit : `~astropy.units.UnitBase`, optionl, keyword only
        The unit for x and y, respectively. If None (default), gets
        the units from x and y.

    See Also
    --------
    UnivariateSpline : Superclass -- allows knots to be selected by a
        smoothing condition
    LSQUnivariateSpline : spline for which knots are user-selected
    splrep : An older, non object-oriented wrapping of FITPACK
    splev, sproot, splint, spalde
    BivariateSpline : A similar class for two-dimensional spline interpolation

    Notes
    -----
    The number of data points must be larger than the spline degree `k`.

    Examples
    --------
    >>> from scipy.interpolate import InterpolatedUnivariateSpline
    >>> x = np.linspace(-3, 3, 50)
    >>> y = np.exp(-x**2) + 0.1 * np.random.randn(50)
    >>> spl = InterpolatedUnivariateSpline(x, y)

    .. code-block:: python

        import matplotlib.pyplot as plt
        plt.plot(x, y, 'ro', ms=5)
        xs = np.linspace(-3, 3, 1000)
        plt.plot(xs, spl(xs), 'g', lw=3, alpha=0.7)
        plt.show()

    Notice that the ``spl(x)`` interpolates `y`:

    >>> spl.get_residual()
    0.0

    """

    def __init__(
        self,
        x: QuantityType,
        y: QuantityType,
        w: T.Optional[np.ndarray] = None,
        bbox: _BBoxType = [None, None],
        k: int = 3,
        ext: int = 0,
        check_finite: bool = False,
        *,
        x_unit: T.Optional[UnitType] = None,
        y_unit: T.Optional[UnitType] = None,
    ):
        # The unit for x and y, respectively. If None (default), gets
        # the units from x and y.
        self._xunit = x_unit or x.unit
        self._yunit = y_unit or y.unit

        # Make x, y to value, so can create IUS as normal
        x = x.to_value(x_unit)
        y = y.to_value(y_unit)

        if bbox[0] is not None:
            bbox[0] = bbox[0].to(self._xunit).value
        if bbox[1] is not None:
            bbox[1] = bbox[1].to(self._xunit).value

        # Make spline
        super().__init__(
            x, y, w=w, bbox=bbox, k=k, ext=ext, check_finite=check_finite
        )

    # /def

    def __call__(self, x, nu=0, ext=None):
        """Evaluate spline (or its nu-th derivative) at positions x.

        Parameters
        ----------
        x : |Quantity| array_like
            A 1-D array of points at which to return the value of the smoothed
            spline or its derivatives. Note: `x` can be unordered but the
            evaluation is more efficient if `x` is (partially) ordered.
        nu  : int, optional
            The order of derivative of the spline to compute.
        ext : int, optional
            Controls the value returned for elements of `x` not in the
            interval defined by the knot sequence.

            * if ext=0 or 'extrapolate', return the extrapolated value.
            * if ext=1 or 'zeros', return 0
            * if ext=2 or 'raise', raise a ValueError
            * if ext=3 or 'const', return the boundary value.

            The default value is 0, passed from the initialization of
            UnivariateSpline.

        Returns
        -------
        y : |Quantity| array_like
            Evaluated spline with units ``._yunit``. Same shape as `x`.

        """
        y = super().__call__(x.to_value(self._xunit), nu=nu, ext=ext)
        return y * self._yunit

    # /def


# /class

##############################################################################
# END
