# -*- coding: utf-8 -*-

"""Splines classes with :mod:`~astropy.units` support.

`scipy` splines do not support quantities with units. The standard workaround
solution is to strip the quantities of their units, apply the interpolation,
then add the units back.

As an example:

    >>> from scipy.interpolate import InterpolatedUnivariateSpline
    >>> x = np.linspace(-3, 3, 50) * u.s
    >>> y = 8 * u.m / (x.value**2 + 4)
    >>> xs = np.linspace(-2, 2, 10) * u.s  # for evaluating spline

    >>> spl = InterpolatedUnivariateSpline(x.to_value(u.s), y.to_value(u.m))
    >>> y_ntrp = spl(xs.to_value(u.s)) * u.m  # evaluate, adding back units
    >>> y_ntrp
    <Quantity [1.00000009, 1.24615404, 1.52830261, 1.79999996, 1.97560874,
               1.97560874, 1.79999996, 1.52830261, 1.24615404, 1.00000009] m>


This is fine, but a bit of a hassle. Instead, we can wrap the unit stripping /
adding process into a unit-aware version of the spline interpolation classes.

The same example as above, but with the new class:

    >>> from trackstream.utils import InterpolatedUnivariateSplinewithUnits
    >>> spl = InterpolatedUnivariateSplinewithUnits(x, y)
    >>> spl(xs)
    <Quantity [1.00000009, 1.24615404, 1.52830261, 1.79999996, 1.97560874,
               1.97560874, 1.79999996, 1.52830261, 1.24615404, 1.00000009] m>

.. plot::
   :context: close-figs
   :alt: example spline plot.

    from trackstream.utils import InterpolatedUnivariateSplinewithUnits
    import astropy.units as u
    from astropy.visualization import quantity_support; quantity_support()

    x = np.linspace(-3, 3, num=50) * u.s
    y = 8 * u.m / (x.value**2 + 4)
    spl = InterpolatedUnivariateSplinewithUnits(x, y)
    spl(np.linspace(-2, 2, num=10) * u.s)  # Evaluate spline

    xs = np.linspace(-3, 3, num=1000) * u.s  # for sampling

    fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(111)
    ax.plot(xs, spl(xs), c="gray", alpha=0.7, lw=3, label="evaluated spline")
    ax.scatter(x, y, c="r", s=25, label="points")

    ax.set_title("Witch of Agnesi (a=1)")
    ax.set_xlabel(f"x [{ax.get_xlabel()}]")
    ax.set_ylabel(f"y [{ax.get_ylabel()}]")
    plt.legend()
    plt.show();


All method except ``derivative`` and ``antiderivative`` have been implemented.


"""

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
    >>> from trackstream.utils import InterpolatedUnivariateSplinewithUnits
    >>> x = np.linspace(-3, 3, 50) * u.s
    >>> y = 8 * u.m / (x.value**2 + 4)
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

    def get_knots(self):
        """Return positions of interior knots of the spline.

        Internally, the knot vector contains ``2*k`` additional boundary knots.
        Has units of `x` position

        """
        return super().get_knots() * self._xunit

    # /def

    def get_residual(self):
        """Return weighted sum of squared residuals of spline approximation.

        This is equivalent to::
            sum((w[i] * (y[i]-spl(x[i])))**2, axis=0)

        """
        return super().get_residual() * self._yunit

    # /def

    def integral(self, a, b):
        r"""Return definite integral of the spline between two given points.

        Parameters
        ----------
        a : float
            Lower limit of integration.
        b : float
            Upper limit of integration.

        Returns
        -------
        integral : float
            The value of the definite integral of the spline between limits.

        Examples
        --------
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, 3, 11)
        >>> y = x**2
        >>> spl = UnivariateSpline(x, y)
        >>> spl.integral(0, 3)
        9.0

        which agrees with :math:`\\int x^2 dx = x^3 / 3` between the limits
        of 0 and 3.

        A caveat is that this routine assumes the spline to be zero outside of
        the data limits:

        >>> spl.integral(-1, 4)
        9.0

        >>> spl.integral(-1, 0)
        0.0

        """
        a_val = a.to_value(self._xunit)
        b_val = b.to_value(self._xunit)
        return super().integral(a_val, b_val) * self._yunit

    # /def

    def derivatives(self, x):
        """Return all derivatives of the spline at the point x.

        Parameters
        ----------
        x : float
            The point to evaluate the derivatives at.

        Returns
        -------
        der : ndarray, shape(k+1,)
            Derivatives of the orders 0 to k.

        Examples
        --------
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, 3, 11)
        >>> y = x**2
        >>> spl = UnivariateSpline(x, y)
        >>> spl.derivatives(1.5)  # doctest: +FLOAT_CMP
        array([2.25, 3.  , 2.  , 0.  ])

        """
        x_val = x.to_value(self._xunit)
        d_vals = super().derivatives(x_val)
        return np.array(
            [d * self._yunit / self._xunit ** i for i, d in enumerate(d_vals)],
            dtype=u.Quantity,
        )

    def roots(self):
        """Return the zeros of the spline.

        Restriction: only cubic splines are supported by fitpack.

        """
        return super().roots() * self._xunit

    # /def

    def derivative(self, n=1):
        r"""Construct a new spline representing the derivative of this spline.

        Parameters
        ----------
        n : int, optional
            Order of derivative to evaluate. Default: 1

        Returns
        -------
        spline : UnivariateSpline
            Spline of order k2=k-n representing the derivative of this
            spline.

        See Also
        --------
        splder, antiderivative

        Examples
        --------
        This can be used for finding maxima of a curve:

        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, 10, 70)
        >>> y = np.sin(x)
        >>> spl = UnivariateSpline(x, y, k=4, s=0)

        Now, differentiate the spline and find the zeros of the
        derivative. (NB: `sproot` only works for order 3 splines, so we
        fit an order 4 spline):

        >>> spl.derivative().roots() / np.pi  # doctest: +FLOAT_CMP
        array([ 0.50000001,  1.5       ,  2.49999998])

        This agrees well with roots :math:`\\pi/2 + n\\pi` of
        :math:`\\cos(x) = \\sin'(x)`.

        """
        raise NotImplementedError("TODO")

    # /def

    def antiderivative(self, n=1):
        r"""Construct a new spline representing the antiderivative of this spline.

        Parameters
        ----------
        n : int, optional
            Order of antiderivative to evaluate. Default: 1

        Returns
        -------
        spline : UnivariateSpline
            Spline of order k2=k+n representing the antiderivative of this
            spline.

        See Also
        --------
        splantider, derivative

        Examples
        --------
        >>> from scipy.interpolate import UnivariateSpline
        >>> x = np.linspace(0, np.pi/2, 70)
        >>> y = 1 / np.sqrt(1 - 0.8*np.sin(x)**2)
        >>> spl = UnivariateSpline(x, y, s=0)

        The derivative is the inverse operation of the antiderivative,
        although some floating point error accumulates:

        >>> spl(1.7) - spl.antiderivative().derivative()(1.7) != 0
        True

        Antiderivative can be used to evaluate definite integrals:

        >>> ispl = spl.antiderivative()
        >>> ispl(np.pi/2) - ispl(0)
        2.2572053588768486

        This is indeed an approximation to the complete elliptic integral
        :math:`K(m) = \\int_0^{\\pi/2} [1 - m\\sin^2 x]^{-1/2} dx`:

        >>> from scipy.special import ellipk
        >>> ellipk(0.8)  # doctest: +FLOAT_CMP
        2.2572053268208538

        """
        raise NotImplementedError("TODO")

    # /def


# /class

##############################################################################
# END
