# -*- coding: utf-8 -*-

"""Interpolated Coordinates, Representations, and SkyCoords.

Astropy coordinate objects are a collection of points.
This module provides wrappers to interpolate each dimension with an affine
parameter.

For all the following examples we assume the following imports:

    >>> import astropy.units as u
    >>> import astropy.coordinates as coord
    >>> import numpy as np
    >>> from trackstream.utils import interpolated_coordinates as icoord

We will start with interpolated representations.

    >>> num = 40
    >>> affine = np.linspace(0, 10, num=num) * u.Myr
    >>> rep = coord.CartesianRepresentation(
    ...     x=np.linspace(0, 1, num=num) * u.kpc,
    ...     y=np.linspace(1, 2, num=num) * u.kpc,
    ...     z=np.linspace(2, 3, num=num) * u.kpc,
    ...     differentials=coord.CartesianDifferential(
    ...         d_x=np.linspace(3, 4, num=num) * (u.km / u.s),
    ...         d_y=np.linspace(4, 5, num=num) * (u.km / u.s),
    ...         d_z=np.linspace(5, 6, num=num) * (u.km / u.s)))
    >>> irep = icoord.InterpolatedRepresentation(rep, affine)
    >>> irep[:4]
    <InterpolatedCartesianRepresentation (affine| x, y, z) in Myr| kpc
        [(0.        , 0.        , 1.        , 2.        ),
         (0.25641026, 0.02564103, 1.02564103, 2.02564103),
         (0.51282051, 0.05128205, 1.05128205, 2.05128205),
         (0.76923077, 0.07692308, 1.07692308, 2.07692308)]
     (has differentials w.r.t.: 's')>

Interpolation means we can get the coordinate (representation) at any point
supported by the affine parameter. For example, the Cartesian coordinate
at some arbitrary value, say ``affine=4.873 * u.Myr``, is

    >>> irep(4.873 * u.Myr)
    <CartesianRepresentation (x, y, z) in kpc
        (0.4873, 1.4873, 2.4873)
     (has differentials w.r.t.: 's')>

The interpolation can be evaluated on a scalar or any shaped |Quantity|
array, returning a Representation with the same shape.

This interpolation machinery is built on top of Astropy's Representation
class and supports all the expected operations, like changing representations,
while maintaining the interpolation.

    >>> irep.represent_as(coord.SphericalRepresentation)[:4]
    <InterpolatedSphericalRepresentation (affine| lon, lat, distance) in ...
        [(0.        , 1.57079633, 1.10714872, 2.23606798),
         (0.25641026, 1.54580153, 1.10197234, 2.27064276),
         (0.51282051, 1.52205448, 1.09671629, 2.30555457),
         (0.76923077, 1.49948886, 1.09140331, 2.34078832)]>

Also supported are some of :mod:`~scipy` interpolation methods. In particular,
we can differentiate the interpolated coordinates with respect to the affine
parameter.

    >>> irep.derivative()[:4]
    <InterpolatedCartesianDifferential (affine| d_x, d_y, d_z) in ...
        [(0.        , 0.1, 0.1, 0.1), (0.25641026, 0.1, 0.1, 0.1),
         (0.51282051, 0.1, 0.1, 0.1), (0.76923077, 0.1, 0.1, 0.1)]>

Note that the result is an interpolated Differential class. Higher-order
derivatives can also be constructed, but they do not have a corresponding
class in Astropy, so a "Generic" class is constructed.

    >>> irep.derivative(n=2)[:4]
    <InterpolatedGenericCartesian2ndDifferential (affine| d_x, d_y, d_z) in ...
        [(0.        , -5.41233725e-16,  3.35564909e-15, -9.45535317e-14),
         (0.25641026,  1.80411242e-17, -2.88657986e-16, -1.91326122e-14),
         (0.51282051,  5.77315973e-16, -3.93296506e-15,  5.62883073e-14),
         (0.76923077, -8.65973959e-16,  5.89944760e-15, -5.06594766e-14)]>

Care should be taken not to change representations for these higher-order
derivatives. The Astropy machinery allows them to be transformed, but
the transformation is often incorrect.


Representations are all well and good, but what about coordinate frames?
The interpolated representations can be used the same as Astropy's, including
in a |Frame|.

    >>> frame = coord.ICRS(irep)
    >>> frame[:1]
    <ICRS Coordinate: (ra, dec, distance) in (deg, deg, kpc)
        [(90., 63.43494882, 2.23606798)]
     (pm_ra_cosdec, pm_dec, radial_velocity) in (mas / yr, mas / yr, km / s)
        [(-0.28301849, -0.12656972, 6.26099034)]>

The underlying representation is still interpolated, and the interpolation
is even kept when transforming frames.

    >>> frame = frame.transform_to(coord.Galactic())
    >>> frame.data[:4]
    <InterpolatedCartesianRepresentation (affine| x, y, z) in Myr| kpc
        [(0.        , -1.8411072 , 1.04913465, 0.71389129),
         (0.25641026, -1.87731612, 1.06955162, 0.69825645),
         (0.51282051, -1.91352503, 1.08996859, 0.68262162),
         (0.76923077, -1.94973395, 1.11038556, 0.66698678)]
     (has differentials w.r.t.: 's')>

For deeper integration and access to interpolated methods the
``InterpolatedCoordinateFrame`` can wrap any ``CoordinateFame``, whether
or not it contains an interpolated representation.

    >>> iframe = icoord.InterpolatedCoordinateFrame(frame)
    >>> iframe[:4]
    <InterpolatedGalactic Coordinate: (affine| l, b, distance) in ...
        [(0.        , 150.32382371, 18.61829304, 2.23606798),
         (0.25641026, 150.32880684, 17.90952972, 2.27064276),
         (0.51282051, 150.33360184, 17.22212858, 2.30555457),
         (0.76923077, 150.33821918, 16.55532737, 2.34078832)]
     (affine| pm_l, pm_b, radial_velocity) in (Myr| mas / yr, mas / yr, km / s)
        [(0.        , 0.00218867, -0.31002428, 6.26099034),
         (0.25641026, 0.00210526, -0.30065482, 6.33590983),
         (0.51282051, 0.00202654, -0.29161849, 6.40935614),
         (0.76923077, 0.00195215, -0.28290567, 6.48140523)]>

When wrapping an un-interpolated coordinate, the affine parameter is required.

    >>> frame = coord.ICRS(rep)  # no interp
    >>> iframe = icoord.InterpolatedCoordinateFrame(frame, affine=affine)
    >>> iframe[:4]
    <InterpolatedICRS Coordinate: (affine| ra, dec, distance) in ...
        [(0.        , 90.        , 63.43494882, 2.23606798),
         (0.25641026, 88.56790382, 63.13836438, 2.27064276),
         (0.51282051, 87.20729763, 62.83721465, 2.30555457),
         (0.76923077, 85.91438322, 62.53280357, 2.34078832)]
     (affine| pm_ra, pm_dec, radial_velocity) in ...
        [(0.        , -0.63284858, -0.12656972, 6.26099034),
         (0.25641026, -0.60122591, -0.12884151, 6.33590983),
         (0.51282051, -0.57125382, -0.13051534, 6.40935614),
         (0.76923077, -0.54290056, -0.13166259, 6.48140523)]>

Just as for interpolated representations, interpolated frames can be evaluated,
differentiated, etc.

    >>> iframe(4.873 * u.Myr)
    <ICRS Coordinate: (ra, dec, distance) in (deg, deg, kpc)
        (71.8590987, 57.82047953, 2.93873848)
     (pm_ra_cosdec, pm_dec, radial_velocity) in (mas / yr, mas / yr, km / s)
        (-0.13759357, -0.1152677, 7.49365212)>

There are also interpolated |SkyCoord|. This is actually a direct subclass
of SkyCoord, not a proxy class like the interpolated representations and
coordinate frame. As such, ``InterpolatedSkyCoord`` can be instantiated in
all the normal ways, except that it requires the kwarg ``affine``. The only
exception is if SkyCoord is wrapping an interpolated CoordinateFrame.

    >>> isc = icoord.InterpolatedSkyCoord(
    ...         [1, 2, 3, 4], [-30, 45, 8, 16],
    ...         frame="icrs", unit="deg",
    ...         affine=affine[:4])
    >>> isc
    <InterpolatedSkyCoord (ICRS): (affine| ra, dec) in Myr| deg
        [(0.        , 1., -30.), (0.25641026, 2.,  45.),
         (0.51282051, 3.,   8.), (0.76923077, 4.,  16.)]>

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import abc
import copy
import inspect
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import astropy.coordinates.representation as r
import astropy.units as u
import numpy as np
import numpy.lib.recfunctions as rfn
from astropy.coordinates import (
    BaseDifferential,
    BaseRepresentation,
    BaseRepresentationOrDifferential,
    CartesianDifferential,
    CartesianRepresentation,
    SkyCoord,
)
from astropy.coordinates.representation import _array2string
from astropy.utils.decorators import format_doc
from numpy import array_equal

# LOCAL
import trackstream._type_hints as TH
from .generic_coordinates import GenericDifferential
from .interpolate import InterpolatedUnivariateSplinewithUnits as IUSU

__all__ = [
    "InterpolatedBaseRepresentationOrDifferential",
    "InterpolatedRepresentation",
    "InterpolatedDifferential",
    "InterpolatedCoordinateFrame",
    "InterpolatedSkyCoord",
]

##############################################################################
# PARAMETERS

_UNIT_DIF_TYPES = (  # the unit-differentials
    r.UnitSphericalDifferential,
    r.UnitSphericalCosLatDifferential,
    r.RadialDifferential,
)

IRoDType = T.TypeVar("IRoDType", bound="InterpolatedBaseRepresentationOrDifferential")
IRType = T.TypeVar("IRType", bound="InterpolatedRepresentation")
ICRType = T.TypeVar("ICRType", bound="InterpolatedCartesianRepresentation")
IDType = T.TypeVar("IDType", bound="InterpolatedDifferential")
DType = T.TypeVar("DType", bound=BaseDifferential)


##############################################################################
# CODE
##############################################################################


def _find_first_best_compatible_differential(
    rep: BaseRepresentation,
    n: int = 1,
) -> T.Union[BaseDifferential, GenericDifferential]:
    """Find a compatible differential.

    There can be more than one, so we select the first one.

    """
    # get names of derivatives wrt the affine parameter
    pkeys = {"d_" + k for k in rep.components}

    # then get compatible differential classes (by matching keys)
    dif_comps = [
        cls
        for cls in rep._compatible_differentials  # the options
        if pkeys == set(cls.attr_classes.keys())  # key match
    ]

    if dif_comps:  # not empty. Can't tell them apart, so the first will do
        derivative_type = dif_comps[0]

    # TODO uncomment when encounter (then can also write test)
    # else:  # nothing matches, so we make a differential
    #     derivative_type = _make_generic_differential_for_representation(
    #         rep.__class__,
    #         n=n,
    #     )

    if n != 1:
        derivative_type = GenericDifferential._make_generic_cls(derivative_type, n=n)

    return derivative_type


def _infer_derivative_type(
    rep: BaseRepresentationOrDifferential,
    dif_unit: TH.UnitLikeType,
    n: int = 1,
) -> T.Union[BaseDifferential, GenericDifferential]:
    """Infer the Differential class used in a derivative wrt time.

    If it can't infer the correct differential class, defaults
    to `~trackstream.utils.generic_coordinates.GenericDifferential`.

    Checks compatible differentials for classes with matching
    names.


    Parameters
    ----------
    rep : `~astropy.coordinates.BaseRepresentationOrDifferential` instance
        The representation object
    dif_unit : unit-like
        The differential unit
    n : int
        The order of the derivative

    Returns
    -------
    `astropy.coordinates.BaseDifferential` or `trackstream.utils.generic_coordinates.GenericDifferential`  # noqa: E501
    """
    unit = u.Unit(dif_unit)
    rep_cls = rep.__class__  # (store rep class for line length)

    # Now check we can even do this: if can't make a better Generic
    # 1) can't for `Differentials` and stuff without compatible diffs
    if isinstance(rep, BaseDifferential):
        derivative_type = GenericDifferential._make_generic_cls(rep_cls, n=n + 1)
    # 2) can't for non-time derivatives
    elif unit.physical_type != "time":
        derivative_type = GenericDifferential._make_generic_cls_for_representation(
            rep_cls,
            n=n,
        )

    else:  # Differentiating a Representation wrt time
        derivative_type = _find_first_best_compatible_differential(rep, n=n)

    return derivative_type


##############################################################################


class InterpolatedBaseRepresentationOrDifferential:
    """Wrapper for Representations, adding affine interpolations.

    .. todo::

        override all the methods, mapping to underlying Representation

        figure out how to do ``from_cartesian`` as a class method

        get ``X_interp`` as properties. Need to do __dict__ manipulation,
        like BaseCoordinateFrame

        pass through derivative_type in all methods!

    Parameters
    ----------
    rep : `~astropy.coordinates.BaseRepresentation` instance
    affine : `~astropy.units.Quantity` array-like
        The affine interpolation parameter.

    interps : Mapping or None (optional, keyword-only)
        Has same structure as a Representation

        ```
            dict(
                component name: interpolation,
                ...
                "differentials": dict(
                    "s" : dict(
                        component name: interpolation,
                        ...
                    ),
                    ...
                )
            )
        ```
    **interp_kwargs
        Only used if `interps` is None.
        keyword arguments into interpolation class

    Other Parameters
    ----------------
    interp_cls : Callable (optional, keyword-only)
        option for 'interp_kwargs'.
        If not specified, default is `IUSU`.

    derivative_type : Callable (optional, keyword-only)
        The class to use when differentiating wrt to the affine parameter.
        If not provided, will use `_infer_derivative_type` to infer.
        Defaults to `GenericDifferential` if all else fails.

    Raises
    ------
    ValueError
        If `rep` is a BaseRepresentationOrDifferential class, not instance
        If affine shape is not 1-D.
        If affine is not same length as `rep`
    TypeError
        If `rep` not not type BaseRepresentationOrDifferential
    """

    def __new__(cls: T.Type[IRoDType], *args: T.Any, **kwargs: T.Any) -> IRoDType:
        if cls is InterpolatedBaseRepresentationOrDifferential:
            raise TypeError(f"Cannot instantiate a {cls}.")

        inst: IRoDType = super().__new__(cls)
        return inst

    def __init__(
        self,
        rep: BaseRepresentationOrDifferential,
        affine: u.Quantity,
        *,
        interps: T.Optional[T.Dict] = None,
        derivative_type: T.Optional[BaseDifferential] = None,
        **interp_kwargs: T.Any,
    ) -> None:
        # Check it's instantiated and right class
        if inspect.isclass(rep) and issubclass(
            rep,
            BaseRepresentationOrDifferential,
        ):
            raise ValueError("Must instantiate `rep`.")
        elif not isinstance(rep, BaseRepresentationOrDifferential):
            raise TypeError("`rep` must be a `BaseRepresentationOrDifferential`.")

        # Affine parameter
        affine = u.Quantity(affine, copy=False)  # ensure Quantity
        if not affine.ndim == 1:
            raise ValueError("`affine` must be 1-D.")
        elif len(affine) != len(rep):
            raise ValueError("`affine` must be same length as `rep`")

        # store representation and affine parameter
        self.data = rep
        self._affine = affine = u.Quantity(affine, copy=True)  # TODO copy?

        # The class to use when differentiating wrt to the affine parameter.
        if derivative_type is None:
            derivative_type = _infer_derivative_type(rep, affine.unit)
        # TODO better detection if derivative_type doesn't work!
        self._derivative_type = derivative_type
        self._derivatives: T.Dict[str, T.Any] = dict()

        # -----------------------
        # Construct interpolation

        self._interp_kwargs = interp_kwargs.copy()  # TODO need copy?

        if interps is not None:
            self._interps = interps
        else:
            # determine interpolation type
            interp_cls = interp_kwargs.pop("interp_cls", IUSU)

            self._interps = dict()
            # positional information
            for comp in rep.components:
                self._interps[comp] = interp_cls(affine, getattr(rep, comp), **interp_kwargs)

            # differentials information
            # these are stored in a dictionary with keys wrt time
            # ex : rep.differentials["s"] is a Velocity
            if hasattr(rep, "differentials"):
                for k, differential in rep.differentials.items():
                    d_derivative_type: T.Optional[type]

                    # Is this already an InterpolatedDifferential?
                    # then need to pop back to the Differential
                    if isinstance(differential, InterpolatedDifferential):
                        d_derivative_type = differential.derivative_type
                        differential = differential.data

                    else:
                        d_derivative_type = None

                    # interpolate differential
                    dif = InterpolatedDifferential(
                        differential,
                        affine,
                        interp_cls=interp_cls,
                        derivative_type=d_derivative_type,
                        **interp_kwargs,
                    )

                    # store in place of original
                    self.data.differentials[k] = dif

    @property
    def affine(self) -> u.Quantity:  # read-only
        return self._affine

    @property
    def _class_(self: IRoDType) -> T.Type[IRoDType]:
        """Get this object's true class, not the un-interpolated class."""
        cls: T.Type[IRoDType] = object.__class__(self)
        return cls

    def _realize_class(self: IRoDType, rep: BaseRepresentation, affine: u.Quantity) -> IRoDType:
        inst: IRoDType = self._class_(
            rep, affine, derivative_type=self.derivative_type, **self._interp_kwargs
        )
        return inst

    #################################################################
    # Interpolation Methods

    @abc.abstractmethod
    def __call__(self, affine: T.Optional[u.Quantity] = None) -> IRoDType:
        """Evaluate interpolated representation.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like
            The affine interpolation parameter.
            If None, returns representation points.
        """

    @property
    def derivative_type(self) -> type:
        """The class used when taking a derivative."""
        return self._derivative_type

    @derivative_type.setter
    def derivative_type(self, value: type) -> None:
        """The class used when taking a derivative."""
        self._derivative_type = value
        self.clear_derivatives()

    def clear_derivatives(self: IRoDType) -> IRoDType:
        """Return self, clearing cached derivatives."""
        if hasattr(self, "_derivatives"):
            keys = tuple(self._derivatives.keys())
            for key in keys:
                if key.startswith("affine "):
                    self._derivatives.pop(key)

        return self

    def derivative(self, n: int = 1) -> InterpolatedDifferential:
        r"""Construct a new spline representing the derivative of this spline.

        .. todo::

            Keep the derivatives of the differentials

        Parameters
        ----------
        n : int, optional
            Order of derivative to evaluate. Default: 1
        """
        # evaluate the spline on each argument of the position
        params = {
            (k if k.startswith("d_") else "d_" + k): interp.derivative(n=n)(self.affine)
            for k, interp in self._interps.items()
        }

        if n == 1:
            deriv_cls = self.derivative_type
        else:
            deriv_cls = _infer_derivative_type(self.data, self.affine.unit, n=n)

        # make Differential
        deriv = deriv_cls(**params)

        # interpolate
        ideriv = InterpolatedDifferential(
            deriv,
            self.affine,
            **self._interp_kwargs,
        )
        # TODO! rare case when differentiating an integral of a Representation
        # then want to return an interpolated Representation!

        return ideriv

    def antiderivative(self, n: int = 1) -> T.Any:
        r"""Construct a new spline representing the integral of this spline.

        .. todo::

            Allow for attaching the differentials?

            a differential should become a position!

        Parameters
        ----------
        n : int, optional
            Order of derivative to evaluate. Default: 1
        """
        # evaluate the spline on each argument of the position
        params = {
            k.lstrip("d_"): interp.antiderivative(n=n)(self.affine)
            for k, interp in self._interps.items()
        }
        # deriv = GenericDifferential(*params)
        # return self._class_(deriv, self.affine, **self._interp_kwargs)
        return params

    # def integral(self, a, b):
    #     """Return definite integral between two given points."""
    #     raise NotImplementedError("What does this even mean?")

    # # /def

    #################################################################
    # Mapping to Underlying Representation

    # ---------------------------------------------------------------
    # hidden methods

    @property
    def __class__(self) -> T.Type[BaseRepresentationOrDifferential]:
        """Make class appear the same as the underlying Representation."""
        cls: T.Type[BaseRepresentationOrDifferential] = self.data.__class__
        return cls

    @__class__.setter
    def __class__(self, value: T.Any) -> None:  # needed for mypy  # noqa: F811
        raise TypeError("cannot set attribute ``__class__``.")

    def __getattr__(self, key: str) -> T.Any:
        """Route everything to underlying Representation."""
        got: T.Any = getattr(self.data, key)
        return got

    def __getitem__(self: IRoDType, key: T.Union[str, slice, np.ndarray]) -> IRoDType:
        """Getitem on Representation, re-interpolating."""
        rep: BaseRepresentation = self.data[key]
        afn: u.Quantity = self.affine[key]
        inst: IRoDType = self._realize_class(rep, afn)
        return inst

    def __len__(self) -> int:
        return len(self.data)

    def __repr__(self) -> str:
        """String Representation, adding interpolation information."""
        prefixstr = "    "
        values = rfn.merge_arrays(
            (self.affine.value, self.data._values),
            flatten=True,
        )
        arrstr = _array2string(values, prefix=prefixstr)

        diffstr = ""
        if getattr(self, "differentials", None):
            diffstr = "\n (has differentials w.r.t.: {})".format(
                ", ".join([repr(key) for key in self.differentials.keys()]),
            )

        aurep = str(self.affine.unit) or "[dimensionless]"

        _unitstr = self.data._unitstr
        if _unitstr:
            if _unitstr[0] == "(":
                unitstr = "in " + "(" + aurep + "| " + _unitstr[1:]
            else:
                unitstr = "in " + aurep + "| " + _unitstr
        else:
            unitstr = f"{aurep}| [dimensionless]"

        s: str = "<Interpolated{} (affine| {}) {:s}\n{}{}{}>".format(
            self.__class__.__name__,
            ", ".join(self.data.components),
            unitstr,
            prefixstr,
            arrstr,
            diffstr,
        )
        return s

    def _scale_operation(
        self: IRoDType, op: T.Callable, *args: T.Any, scaled_base: bool = False
    ) -> IRoDType:
        rep = self.data._scale_operation(op, *args, scaled_base=scaled_base)
        inst: IRoDType = self._realize_class(rep, self.affine)
        return inst

    # ---------------------------------------------------------------
    # math methods

    def __add__(
        self: IRoDType,
        other: T.Union[BaseRepresentationOrDifferential, IRoDType],
    ) -> IRoDType:
        """Add other to an InterpolatedBaseRepresentationOrDifferential

        If other:
        - point : add to data, keep affine the same, re-interpolate
        - vector : add to data, keep affine the same, re-interpolate
        - interpolated : must be same interpolation!
        """
        if isinstance(other, InterpolatedBaseRepresentationOrDifferential):
            if not array_equal(other.affine, self.affine):
                raise ValueError(
                    f"Can only add two {self._class_}"
                    + " if the interpolation variables are the same.",
                )

        # add
        newrep = self.data.__add__(other)

        # now re-interpolate
        return self._realize_class(newrep, self.affine)

    def __sub__(
        self: IRoDType,
        other: T.Union[IRoDType, BaseRepresentationOrDifferential],
    ) -> IRoDType:
        """Add other to an InterpolatedBaseRepresentationOrDifferential

        If other:
        - point : add to data, keep affine the same, re-interpolate
        - vector : add to data, keep affine the same, re-interpolate
        - interpolated : must be same interpolation!
        """
        if isinstance(other, InterpolatedBaseRepresentationOrDifferential):
            if not array_equal(other.affine, self.affine):
                raise ValueError(
                    f"Can only subtract two {self._class_}"
                    + " if the interpolation variables are the same.",
                )

        # add
        newrep = self.data.__sub__(other)

        # now re-interpolate
        return self._realize_class(newrep, self.affine)

    def __mul__(
        self: IRoDType,
        other: T.Union[IRoDType, BaseRepresentationOrDifferential],
    ) -> IRoDType:
        """Add other to an InterpolatedBaseRepresentationOrDifferential

        If other:
        - point : add to data, keep affine the same, re-interpolate
        - vector : add to data, keep affine the same, re-interpolate
        - interpolated : must be same interpolation!
        """
        if isinstance(other, InterpolatedBaseRepresentationOrDifferential):
            if not array_equal(other.affine, self.affine):
                raise ValueError(
                    f"Can only multiply two {self._class_}"
                    + " if the interpolation variables are the same.",
                )

        # add
        newrep = self.data.__mul__(other)

        # now re-interpolate
        return self._realize_class(newrep, self.affine)

    def __truediv__(
        self: IRoDType,
        other: T.Union[IRoDType, BaseRepresentationOrDifferential],
    ) -> IRoDType:
        """Add other to an InterpolatedBaseRepresentationOrDifferential

        If other:
        - point : add to data, keep affine the same, re-interpolate
        - vector : add to data, keep affine the same, re-interpolate
        - interpolated : must be same interpolation!
        """
        if isinstance(other, InterpolatedBaseRepresentationOrDifferential):
            if not array_equal(other.affine, self.affine):
                raise ValueError(
                    f"Can only divide two {self._class_}"
                    + " if the interpolation variables are the same.",
                )

        # add
        newrep = self.data.__truediv__(other)

        # now re-interpolate
        return self._realize_class(newrep, self.affine)

    # def _apply(self, method, *args, **kwargs):
    #     """Create a new representation or differential with ``method`` applied
    #     to the component data and the interpolation parameter.

    #     In typical usage, the method is any of the shape-changing methods for
    #     `~numpy.ndarray` (``reshape``, ``swapaxes``, etc.), as well as those
    #     picking particular elements (``__getitem__``, ``take``, etc.), which
    #     are all defined in `~astropy.utils.shapes.ShapedLikeNDArray`. It will be
    #     applied to the underlying arrays (e.g., ``x``, ``y``, and ``z`` for
    #     `~astropy.coordinates.CartesianRepresentation`), with the results used
    #     to create a new instance.

    #     Internally, it is also used to apply functions to the components
    #     (in particular, `~numpy.broadcast_to`).

    #     Parameters
    #     ----------
    #     method : str or callable
    #         If str, it is the name of a method that is applied to the internal
    #         ``components``. If callable, the function is applied.
    #     args : tuple
    #         Any positional arguments for ``method``.
    #     kwargs : dict
    #         Any keyword arguments for ``method``.

    #     """
    #     rep = self.data._apply(method, *args, **kwargs)

    #     if callable(method):
    #         apply_method = affine array: method(array, *args, **kwargs)
    #     else:
    #         apply_method = operator.methodcaller(method, *args, **kwargs)

    #     affine = apply_method(self.affine)

    #     return InterpolatedRepresentation(rep, affine)

    # # /def

    # ---------------------------------------------------------------
    # Specific wrappers

    def from_cartesian(
        self: IRoDType,
        other: T.Union[CartesianRepresentation, CartesianDifferential],
    ) -> IRoDType:
        """Create a representation of this class from a Cartesian one.

        Parameters
        ----------
        other : `CartesianRepresentation` or `CartesianDifferential`
            The representation to turn into this class

            Note: the affine parameter of this class is used. The
            representation must be the same length as the affine parameter.

        Returns
        -------
        representation : object of this class
            A new representation of this class's type.

        Raises
        ------
        ValueError
            If `other` is not same length as the this instance's affine
            parameter.
        """
        rep = self.data.from_cartesian(other)
        return self._class_(rep, self.affine, **self._interp_kwargs)

    # TODO just wrap self.data method with a wrapper?
    def to_cartesian(self: IRoDType) -> IRoDType:
        """Convert the representation to its Cartesian form.

        Note that any differentials get dropped. Also note that orientation
        information at the origin is *not* preserved by conversions through
        Cartesian coordinates. For example, transforming an angular position
        defined at distance=0 through cartesian coordinates and back will lose
        the original angular coordinates::

            >>> import astropy.units as u
            >>> import astropy.coordinates as coord
            >>> rep = coord.SphericalRepresentation(
            ...     lon=15*u.deg,
            ...     lat=-11*u.deg,
            ...     distance=0*u.pc)
            >>> rep.to_cartesian().represent_as(coord.SphericalRepresentation)
            <SphericalRepresentation (lon, lat, distance) in (rad, rad, pc)
                (0., 0., 0.)>

        Returns
        -------
        cartrepr : `CartesianRepresentation` or `CartesianDifferential`
            The representation in Cartesian form.
            If starting from a Cart
        """
        rep = self.data.to_cartesian()
        return self._class_(rep, self.affine, **self._interp_kwargs)

    def copy(self: IRoDType, *args: T.Any, **kwargs: T.Any) -> IRoDType:
        """Return an instance containing copies of the internal data.

        Parameters are as for :meth:`~numpy.ndarray.copy`.

        .. todo::

            this uses BaseRepresentation._apply, see if that may be modified
            instead

        Returns
        -------
        `InterpolatedBaseRepresentationOrDifferential`
            Same type as this instance.
        """
        data = self.data.copy(*args, **kwargs)
        interps = copy.deepcopy(self._interps)
        return self._class_(
            data,
            self.affine,
            interps=interps,
            derivative_type=self.derivative_type,
            **self._interp_kwargs,
        )


# -------------------------------------------------------------------


class InterpolatedRepresentation(InterpolatedBaseRepresentationOrDifferential):
    """Wrapper for Representations, adding affine interpolations.

    .. todo::

        override all the methods, mapping to underlying Representation

        figure out how to do ``from_cartesian`` as a class method

        get ``X_interp`` as properties. Need to do __dict__ manipulation,
        like BaseCoordinateFrame

    Parameters
    ----------
    representation : `~astropy.coordinates.BaseRepresentation` instance
    affine : `~astropy.units.Quantity` array-like
        The affine interpolation parameter.

    interps : Mapping or None (optional, keyword-only)
        Has same structure as a Representation

        .. code-block:: text

            dict(
                component name: interpolation,
                ...
                "differentials": dict(
                    "s" : dict(
                        component name: interpolation,
                        ...
                    ),
                    ...
                )
            )

    **interp_kwargs
        Only used if `interps` is None.
        keyword arguments into interpolation class

    Other Parameters
    ----------------
    interp_cls : Callable (optional, keyword-only)
        option for 'interp_kwargs'.
        If not specified, default is `IUSU`.

    """

    def __new__(
        cls: T.Type[IRType], representation: BaseRepresentation, *args: T.Any, **kwargs: T.Any
    ) -> T.Union[IRType, InterpolatedCartesianRepresentation]:
        self: T.Union[IRType, InterpolatedCartesianRepresentation]
        # need to special case since it has different methods
        if isinstance(representation, CartesianRepresentation):
            self = super().__new__(
                InterpolatedCartesianRepresentation,
                representation,
                *args,
                **kwargs,
            )
        else:
            self = super().__new__(cls, representation, *args, **kwargs)

        return self

    def __call__(self, affine: T.Optional[u.Quantity] = None) -> BaseRepresentation:
        """Evaluate interpolated representation.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like
            The affine interpolation parameter.
            If None, returns representation points.

        Returns
        -------
        :class:`~astropy.coordinates.BaseRepresenation`
            Representation of type ``self.data`` evaluated with `affine`
        """
        if affine is None:  # If None, returns representation as-is.
            return self.data
        # else:

        affine = u.Quantity(affine)  # need to ensure Quantity

        differentials = dict()
        for k, dif in self.data.differentials.items():
            differentials[k] = dif(affine)

        # evaluate the spline on each argument of the position
        params = {n: interp(affine) for n, interp in self._interps.items()}

        return self.data.__class__(**params, differentials=differentials)

    # ---------------------------------------------------------------

    # TODO just wrap self.data method with a wrapper?
    def represent_as(
        self: IRType,
        other_class: BaseRepresentation,
        differential_class: T.Optional[BaseDifferential] = None,
    ) -> InterpolatedRepresentation:
        """Convert coordinates to another representation.

        If the instance is of the requested class, it is returned unmodified.
        By default, conversion is done via Cartesian coordinates. Also note
        that orientation information at the origin is *not* preserved by
        conversions through Cartesian coordinates. See the docstring for
        `~astropy.coordinates.BaseRepresentation.represent_as()` for an
        example.

        Parameters
        ----------
        other_class : `~astropy.coordinates.BaseRepresentation` subclass
            The type of representation to turn the coordinates into.
        differential_class : dict of |Differential|, optional
            Classes in which the differentials should be represented.
            Can be a single class if only a single differential is attached,
            otherwise it should be a `dict` keyed by the same keys as the
            differentials.
        """
        rep = self.data.represent_as(
            other_class,
            differential_class=differential_class,
        )

        # don't pass on the derivative_type
        # can't do self._class_ since InterpolatedCartesianRepresentation
        # only accepts `rep` of Cartesian type.
        return InterpolatedRepresentation(rep, self.affine, **self._interp_kwargs)

    # TODO just wrap self.data method with a wrapper?
    def with_differentials(self: IRType, differentials: T.Sequence[BaseDifferential]) -> IRType:
        """Realize Representation, with new differentials.

        Create a new representation with the same positions as this
        representation, but with these new differentials.

        Differential keys that already exist in this object's differential dict
        are overwritten.

        Parameters
        ----------
        differentials : Sequence of `~astropy.coordinates.BaseDifferential`
            The differentials for the new representation to have.

        Returns
        -------
        newrepr
            A copy of this representation, but with the ``differentials`` as
            its differentials.
        """
        if not differentials:  # (from source code)
            return self

        rep = self.data.with_differentials(differentials)
        return self._realize_class(rep, self.affine)

    # TODO just wrap self.data method with a wrapper?
    def without_differentials(self: IRType) -> IRType:
        """Return a copy of the representation without attached differentials.

        Returns
        -------
        newrepr
            A shallow copy of this representation, without any differentials.
            If no differentials were present, no copy is made.
        """
        if not self._differentials:  # from source code
            return self

        rep = self.data.without_differentials()
        return self._realize_class(rep, self.affine)

    def derivative(self: IRType, n: int = 1) -> InterpolatedDifferential:
        r"""Construct a new spline representing the derivative of this spline.

        Parameters
        ----------
        n : int, optional
            Order of derivative to evaluate. Default: 1
        """
        ideriv: InterpolatedDifferential
        if f"affine {n}" in self._derivatives:
            ideriv = self._derivatives[f"affine {n}"]
            return ideriv

        ideriv = super().derivative(n=n)

        # cache in derivatives
        self._derivatives[f"affine {n}"] = ideriv

        return ideriv

    # ---------------------------------------------------------------
    # Convenience interpolation methods

    def headless_tangent_vectors(self: IRType) -> IRType:
        r"""Headless tangent vector at each point in affine.

        :math:`\vec{x} + \partial_{\affine} \vec{x}(\affine) \Delta\affine`

        .. todo::

            allow for passing my own points
        """
        irep = self.represent_as(CartesianRepresentation)
        ideriv = irep.derivative(n=1)  # (interpolated)

        offset = CartesianRepresentation(*(ideriv.d_xyz * self.affine.unit))
        offset = offset.represent_as(self.__class__)  # transform back

        return self._realize_class(offset, self.affine)

    def tangent_vectors(self: IRType) -> IRType:
        r"""Tangent vectors along the curve, from the origin.

        :math:`\vec{x} + \partial_{\affine} \vec{x}(\affine) \Delta\affine`

        .. todo::

            allow for passing my own points
        """
        irep = self.represent_as(CartesianRepresentation)
        ideriv = irep.derivative(n=1)  # (interpolated)

        offset = CartesianRepresentation(*(ideriv.d_xyz * self.affine.unit))

        newirep = irep + offset
        newirep = newirep.represent_as(self.__class__)

        return self._realize_class(newirep, self.affine)


class InterpolatedCartesianRepresentation(InterpolatedRepresentation):
    def __init__(
        self,
        rep: CartesianRepresentation,
        affine: T.Optional[u.Quantity],
        *,
        interps: T.Optional[T.Dict] = None,
        derivative_type: T.Optional[BaseDifferential] = None,
        **interp_kwargs: T.Any,
    ) -> None:

        # Check its instantiated and right class
        if inspect.isclass(rep) and issubclass(
            rep,
            coord.CartesianRepresentation,
        ):
            raise ValueError("Must instantiate `rep`.")
        elif not isinstance(rep, coord.CartesianRepresentation):
            raise TypeError("`rep` must be a `CartesianRepresentation`.")

        return super().__init__(
            rep,
            affine=affine,
            interps=interps,
            derivative_type=derivative_type,
            **interp_kwargs,
        )

    # TODO just wrap self.data method with a wrapper?
    def transform(self: ICRType, matrix: np.ndarray) -> ICRType:
        """Transform the cartesian coordinates using a 3x3 matrix.

        This returns a new representation and does not modify the original one.
        Any differentials attached to this representation will also be
        transformed.

        Parameters
        ----------
        matrix : `~numpy.ndarray`
            A 3x3 transformation matrix, such as a rotation matrix.


        Examples
        --------

        We can start off by creating a Cartesian representation object:

            >>> from astropy import units as u
            >>> from astropy.coordinates import CartesianRepresentation
            >>> rep = CartesianRepresentation([1, 2] * u.pc,
            ...                               [2, 3] * u.pc,
            ...                               [3, 4] * u.pc)

        We now create a rotation matrix around the z axis:

            >>> from astropy.coordinates.matrix_utilities import (
            ...     rotation_matrix)
            >>> rotation = rotation_matrix(30 * u.deg, axis='z')

        Finally, we can apply this transformation:

            >>> rep_new = rep.transform(rotation)
            >>> rep_new.xyz  # doctest: +FLOAT_CMP
            <Quantity [[ 1.8660254 , 3.23205081],
                       [ 1.23205081, 1.59807621],
                       [ 3.        , 4.        ]] pc>
        """
        newrep = self.data.transform(matrix)

        return self._realize_class(newrep, self.affine)

    def _scale_operation(self: ICRType, op: T.Callable, *args: T.Any) -> ICRType:  # type: ignore
        rep = self.data._scale_operation(op, *args)
        return self._realize_class(rep, self.affine)


# -------------------------------------------------------------------


class InterpolatedDifferential(InterpolatedBaseRepresentationOrDifferential):

    # ---------------------------------------------------------------

    def __new__(
        cls: T.Type[IDType], rep: T.Union[IDType, DType], *args: T.Any, **kwargs: T.Any
    ) -> IDType:
        if not isinstance(rep, InterpolatedDifferential) and not isinstance(
            rep,
            BaseDifferential,
        ):
            raise TypeError("`rep` must be a differential type.")

        return super().__new__(cls, rep, *args, **kwargs)

    # ---------------------------------------------------------------

    def __call__(self, affine: T.Optional[u.Quantity] = None) -> DType:
        """Evaluate interpolated representation.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like
            The affine interpolation parameter.
            If None, returns representation points.

        Returns
        -------
        `~astropy.coordinates.BaseDifferential`
            Representation of type ``self.data`` evaluated with `affine`
        """
        data: DType

        if affine is None:  # If None, returns representation as-is.
            data = self.data
            return data

        # evaluate the spline on each argument of the position
        affine = u.Quantity(affine, copy=False)  # need to ensure Quantity
        params = {n: interp(affine) for n, interp in self._interps.items()}

        data = self.data.__class__(**params)
        return data

    # ---------------------------------------------------------------

    # TODO just wrap self.data method with a wrapper?
    def represent_as(
        self: IDType,
        other_class: BaseDifferential,
        base: BaseRepresentation,
    ) -> IDType:
        """Convert coordinates to another representation.

        If the instance is of the requested class, it is returned unmodified.
        By default, conversion is done via cartesian coordinates.

        Parameters
        ----------
        other_class : `~astropy.coordinates.BaseDifferential` subclass
            The type of representation to turn the coordinates into.
        base : instance of ``self.base_representation``
            Base relative to which the differentials are defined.  If the other
            class is a differential representation, the base will be converted
            to its ``base_representation``.
        """
        rep = self.data.represent_as(other_class, base=base)

        # don't pass on the derivative_type
        return self._class_(rep, self.affine, **self._interp_kwargs)

    def to_cartesian(self) -> InterpolatedCartesianRepresentation:  # type: ignore
        """Convert the differential to its Cartesian form.

        Note that any differentials get dropped. Also note that orientation
        information at the origin is *not* preserved by conversions through
        Cartesian coordinates. For example, transforming an angular position
        defined at distance=0 through cartesian coordinates and back will lose
        the original angular ccoordinates::

            >>> import astropy.units as u
            >>> import astropy.coordinates as coord
            >>> rep = coord.SphericalRepresentation(
            ...     lon=15*u.deg,
            ...     lat=-11*u.deg,
            ...     distance=0*u.pc)
            >>> rep.to_cartesian().represent_as(coord.SphericalRepresentation)
            <SphericalRepresentation (lon, lat, distance) in (rad, rad, pc)
                (0., 0., 0.)>

        Returns
        -------
        `CartesianRepresentation`
            The representation in Cartesian form.
            On Differentials, ``to_cartesian`` returns a Representation
            https://github.com/astropy/astropy/issues/6215
        """
        rep: CartesianRepresentation = self.data.to_cartesian()
        return InterpolatedCartesianRepresentation(rep, self.affine, **self._interp_kwargs)


#####################################################################


class InterpolatedCoordinateFrame:
    """Wrapper for Coordinate Frame, adding affine interpolations.

    .. todo::

        - override all the methods, mapping to underlying CoordinateFrame

        - allow for ICRS(InterpolatedRepresentation()) to work

    Parameters
    ----------
    data : InterpolatedRepresentation or Representation or CoordinateFrame
        For either an InterpolatedRepresentation or Representation
        the kwarg 'frame' must also be specified.
        If CoordinateFrame, then 'frame' is ignored.
    affine : Quantity array-like (optional)
        if not a Quantity, one is assigned.
        Only used if data is not already interpolated.
        If data is NOT interpolated, this is required.


    Other Parameters
    ----------------
    frame : str or CoordinateFrame
        only used if `data` is  an InterpolatedRepresentation or Representation

    Raises
    ------
    Exception
        if `frame` has no error
    ValueError
        if `data` is not an interpolated type and `affine` is None
    TypeError
        if `data` is not one of types specified in Parameters.
    """

    def __init__(
        self,
        data: TH.CoordinateType,
        affine: T.Optional[u.Quantity] = None,
        *,
        interps: T.Optional[T.Dict] = None,
        **interp_kwargs: T.Any,
    ) -> None:
        # get rep from CoordinateType
        rep = data.data

        if isinstance(rep, InterpolatedRepresentation):
            pass
        elif isinstance(rep, BaseRepresentation):
            if affine is None:
                raise ValueError(
                    "`data` is not already interpolated. "
                    "Need to pass a Quantity array for `affine`.",
                )

            rep = InterpolatedRepresentation(rep, affine=affine, interps=interps, **interp_kwargs)
        else:
            raise TypeError(
                "`data` must be type " + "<InterpolatedRepresentation> or <BaseRepresentation>",
            )

        self.frame = data.realize_frame(rep)
        self._interp_kwargs = interp_kwargs

    @property
    def _interp_kwargs(self) -> T.Dict[str, T.Any]:
        ikw: dict = self.data._interp_kwargs
        return ikw

    @_interp_kwargs.setter
    def _interp_kwargs(self, value: dict) -> None:
        self.data._interp_kwargs = value

    def __call__(self, affine: T.Optional[u.Quantity] = None) -> BaseRepresentation:
        """Evaluate interpolated coordinate frame.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like
            The affine interpolation parameter.
            If None, returns representation points.

        Returns
        -------
        BaseRepresenation
            Representation of type ``self.data`` evaluated with `affine`
        """
        return self.frame.realize_frame(self.frame.data(affine))

    @property
    def _class_(self) -> T.Type[InterpolatedCoordinateFrame]:
        return object.__class__(self)

    def _realize_class(self, data: TH.CoordinateType) -> InterpolatedCoordinateFrame:
        return self._class_(data, affine=self.affine, **self._interp_kwargs)

    def realize_frame(
        self, data: BaseRepresentation, affine: T.Optional[u.Quantity] = None, **kwargs: T.Any
    ) -> InterpolatedCoordinateFrame:
        """Generates a new frame with new data from another frame (which may or
        may not have data). Roughly speaking, the converse of
        `replicate_without_data`.

        Parameters
        ----------
        data : `~astropy.coordinates.BaseRepresentation`
            The representation to use as the data for the new frame.

        Any additional keywords are treated as frame attributes to be set on
        the new frame object. In particular, `representation_type` can be
        specified.

        Returns
        -------
        frameobj : same as this frame
            A new object with the same frame attributes as this one, but
            with the ``data`` as the coordinate data.
        """
        frame = self.frame.realize_frame(data, **kwargs)
        return self._class_(frame, affine=affine, **kwargs)

    #################################################################
    # Interpolation Methods
    # Mapped to underlying Representation

    @format_doc(InterpolatedBaseRepresentationOrDifferential.derivative.__doc__)
    def derivative(self, n: int = 1) -> BaseRepresentationOrDifferential:
        """Take nth derivative wrt affine parameter."""
        return self.frame.data.derivative(n=n)

    @property
    def affine(self) -> u.Quantity:  # read-only
        return self.frame.data.affine

    def headless_tangent_vectors(self) -> InterpolatedCoordinateFrame:
        r"""Headless tangent vector at each point in affine.

        :math:`\vec{x} + \partial_{\affine} \vec{x}(\affine) \Delta\affine`

        .. todo::

            allow for passing my own points
        """
        rep = self.frame.data.headless_tangent_vectors()
        return self.realize_frame(rep)

    def tangent_vectors(self) -> InterpolatedCoordinateFrame:
        r"""Tangent vectors along the curve, from the origin.

        :math:`\vec{x} + \partial_{\affine} \vec{x}(\affine) \Delta\affine`

        .. todo::

            allow for passing my own points
        """
        rep = self.frame.data.tangent_vectors()
        return self.realize_frame(rep)

    #################################################################
    # Mapping to Underlying CoordinateFrame

    @property
    def __class__(self) -> type:
        """Make class appear the same as the underlying CoordinateFrame."""
        cls: type = self.frame.__class__
        return cls

    @__class__.setter
    def __class__(self, value: T.Any) -> None:  # needed for mypy  # noqa: F811
        raise TypeError("cannot set the `__class__` attribute.")

    def __getattr__(self, key: str) -> T.Any:
        """Route everything to underlying CoordinateFrame."""
        return getattr(self.frame, key)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, key: T.Union[int, slice, np.ndarray]) -> InterpolatedCoordinateFrame:
        frame = self.frame[key]
        affine = self.affine[key]

        iframe = self._class_(frame, affine=affine, **self._interp_kwargs)
        iframe.representation_type = self.representation_type

        return iframe

    @property
    def representation_type(self) -> BaseRepresentation:
        return self.frame.representation_type

    @representation_type.setter
    def representation_type(self, value: BaseRepresentation) -> None:
        self.frame.representation_type = value

    def represent_as(
        self,
        base: TH.RepLikeType,
        s: T.Union[str, BaseDifferential] = "base",
        in_frame_units: bool = False,
    ) -> InterpolatedRepresentation:
        """Generate and return a new representation of this frame's `data`
        as a Representation object.

        Note: In order to make an in-place change of the representation
        of a Frame or SkyCoord object, set the ``representation``
        attribute of that object to the desired new representation, or
        use the ``set_representation_cls`` method to also set the differential.

        Parameters
        ----------
        base : subclass of BaseRepresentation or string
            The type of representation to generate.  Must be a *class*
            (not an instance), or the string name of the representation
            class.
        s : subclass of `~astropy.coordinates.BaseDifferential`, str, optional
            Class in which any velocities should be represented. Must be
            a *class* (not an instance), or the string name of the
            differential class.  If equal to 'base' (default), inferred from
            the base class.  If `None`, all velocity information is dropped.
        in_frame_units : bool, keyword only
            Force the representation units to match the specified units
            particular to this frame

        Returns
        -------
        newrep : BaseRepresentation-derived object
            A new representation object of this frame's `data`.

        Raises
        ------
        AttributeError
            If this object had no `data`

        Examples
        --------
        >>> from astropy import units as u
        >>> from astropy.coordinates import SkyCoord, CartesianRepresentation
        >>> coord = SkyCoord(0*u.deg, 0*u.deg)
        >>> coord.represent_as(CartesianRepresentation)  # doctest: +FLOAT_CMP
        <CartesianRepresentation (x, y, z) [dimensionless]
                (1., 0., 0.)>

        >>> coord.representation_type = CartesianRepresentation
        >>> coord  # doctest: +FLOAT_CMP
        <SkyCoord (ICRS): (x, y, z) [dimensionless]
            (1., 0., 0.)>
        """
        rep = self.frame.represent_as(base, s=s, in_frame_units=in_frame_units)

        return InterpolatedRepresentation(rep, affine=self.affine, **self._interp_kwargs)

    def transform_to(
        self,
        new_frame: T.Union[coord.BaseCoordinateFrame, SkyCoord],
    ) -> InterpolatedCoordinateFrame:
        """Transform this object's coordinate data to a new frame.

        Parameters
        ----------
        new_frame : frame object or SkyCoord object
            The frame to transform this coordinate frame into.

        Returns
        -------
        transframe
            A new object with the coordinate data represented in the
            ``newframe`` system.

        Raises
        ------
        ValueError
            If there is no possible transformation route.
        """
        newframe = self.frame.transform_to(new_frame)
        return self._realize_class(newframe)

    def copy(self) -> InterpolatedCoordinateFrame:
        interp_kwargs = self._interp_kwargs.copy()
        frame = self.frame.realize_frame(self.data)
        iframe: InterpolatedCoordinateFrame = self._class_(
            frame,
            affine=self.affine.copy(),
            interps=None,
            **interp_kwargs,
        )
        return iframe

    def _frame_attrs_repr(self) -> str:  # FIXME!!
        s: str = self.frame._frame_attrs_repr()
        return s

    def _data_repr(self) -> str:
        """Returns a string representation of the coordinate data.

        Returns
        -------
        str
            string representation of the data
        """
        # if not self.has_data:  # must have data to be interpolated
        #     return ""

        rep_cls = self.representation_type

        if rep_cls:
            if hasattr(rep_cls, "_unit_representation") and isinstance(
                self.frame.data,
                rep_cls._unit_representation,
            ):
                rep_cls = self.frame.data.__class__

            if "s" in self.frame.data.differentials:
                dif_cls = self.get_representation_cls("s")
                dif_data = self.frame.data.differentials["s"]
                if isinstance(dif_data, _UNIT_DIF_TYPES):
                    dif_cls = dif_data.__class__

            else:
                dif_cls = None

            data = self.represent_as(rep_cls, dif_cls, in_frame_units=True)
            data_repr = repr(data)

            # Generate the list of component names out of the repr string
            part1, _, remainder = data_repr.partition("(")
            if remainder != "":
                comp_str, _, part2 = remainder.partition(")")
                comp_names: T.List[str] = comp_str.split(", ")

                affine_name, comp_name_0 = comp_names[0].split("| ")
                comp_names[0] = comp_name_0

                # Swap in frame-specific component names
                rep_comp_names = self.representation_component_names
                invnames = {nmrepr: nmpref for nmpref, nmrepr in rep_comp_names.items()}
                for i, name in enumerate(comp_names):
                    comp_names[i] = invnames.get(name, name)

                # Reassemble the repr string
                data_repr = part1 + "(" + affine_name + "| " + ", ".join(comp_names) + ")" + part2

        # else:  # uncomment when encounter
        #     data = self.frame.data
        #     data_repr = repr(self.data)

        # /if

        data_cls_name = "Interpolated" + data.__class__.__name__
        if data_repr.startswith("<" + data_cls_name):
            # remove both the leading "<" and the space after the name, as well
            # as the trailing ">"
            i = len(data_cls_name) + 2
            data_repr = data_repr[i:-1]
        # else:  # uncomment when encounter
        #     data_repr = "Data:\n" + data_repr

        if "s" in self.data.differentials:
            data_repr_spl = data_repr.split("\n")
            if "has differentials" in data_repr_spl[-1]:
                diffrepr = repr(data.differentials["s"]).split("\n")
                if diffrepr[0].startswith("<"):
                    diffrepr[0] = " " + " ".join(diffrepr[0].split(" ")[1:])
                for frm_nm, rep_nm in self.get_representation_component_names(
                    "s",
                ).items():
                    diffrepr[0] = diffrepr[0].replace(rep_nm, frm_nm)
                if diffrepr[-1].endswith(">"):
                    diffrepr[-1] = diffrepr[-1][:-1]
                data_repr_spl[-1] = "\n".join(diffrepr)

            data_repr = "\n".join(data_repr_spl)

        return data_repr

    def __repr__(self) -> str:
        frameattrs = self._frame_attrs_repr()
        data_repr = self._data_repr()

        if frameattrs:
            frameattrs = f" ({frameattrs})"

        cls_name = self.__class__.__name__
        if data_repr:
            return f"<Interpolated{cls_name} " f"Coordinate{frameattrs}: {data_repr}>"
        # else:  # uncomment when encounter
        #     return f"<Interpolated{cls_name} Frame{frameattrs}>"

        return "TODO!"

    # -----------------------------------------------------
    # Separation

    def separation(
        self,
        point: TH.CoordinateType,
        *,
        interpolate: bool = True,
        affine: T.Optional[u.Quantity] = None,
    ) -> T.Union[coord.Angle, IUSU]:
        """Computes on-sky separation between this coordinate and another.

        .. note::

            If the ``other`` coordinate object is in a different frame, it is
            first transformed to the frame of this object. This can lead to
            unintuitive behavior if not accounted for. Particularly of note is
            that ``self.separation(other)`` and ``other.separation(self)`` may
            not give the same answer in this case.

        Parameters
        ----------
        other : `~astropy.coordinates.BaseCoordinateFrame`
            The coordinate to get the separation to.
        interpolated : bool, optional keyword-only
            Whether to return the separation as an interpolation, or as points.
        affine : `~astropy.units.Quantity` array-like or None, optional
            The affine interpolation parameter. If None (default), return
            angular width evaluated at all "tick" interpolation points.

        Returns
        -------
        sep : `~astropy.coordinates.Angle` or `~.interpolate.InterpolatedUnivariateSplinewithUnits`
            The on-sky separation between this and the ``other`` coordinate.

        Notes
        -----
        The separation is calculated using the Vincenty formula, which
        is stable at all locations, including poles and antipodes [1]_.

        .. [1] https://en.wikipedia.org/wiki/Great-circle_distance
        """
        return self._separation(point, angular=True, interpolate=interpolate, affine=affine)

    def separation_3d(
        self,
        point: TH.CoordinateType,
        *,
        interpolate: bool = True,
        affine: T.Optional[u.Quantity] = None,
    ) -> T.Union[coord.Distance, IUSU]:
        """
        Computes three dimensional separation between this coordinate
        and another.

        For more on how to use this (and related) functionality, see the
        examples in :doc:`astropy:/coordinates/matchsep`.

        Parameters
        ----------
        other : `~astropy.coordinates.BaseCoordinateFrame`
            The coordinate to get the separation to.
        interpolated : bool, optional keyword-only
            Whether to return the separation as an interpolation, or as points.
        affine : `~astropy.units.Quantity` array-like or None, optional
            The affine interpolation parameter. If None (default), return
            angular width evaluated at all "tick" interpolation points.

        Returns
        -------
        sep : `~astropy.coordinates.Distance` or `~.interpolate.InterpolatedUnivariateSplinewithUnits`  # noqa: E501
            The real-space distance between these two coordinates.

        Raises
        ------
        ValueError
            If this or the other coordinate do not have distances.
        """
        return self._separation(point, angular=False, interpolate=interpolate, affine=affine)

    def _separation(
        self,
        point: SkyCoord,
        angular: bool,
        interpolate: bool,
        affine: T.Optional[u.Quantity],
    ) -> T.Union[coord.Angle, coord.Distance, IUSU]:
        """Separation helper function."""
        affine = self.affine if affine is None else affine

        c = self(affine=affine)
        seps = getattr(c, "separation" if angular else "separation_3d")(point)

        if not interpolate:
            return seps

        return IUSU(affine, seps)  # TODO! if 1 point


#####################################################################


class InterpolatedSkyCoord(SkyCoord):
    """Interpolated SkyCoord."""

    def __init__(
        self,
        *args: T.Any,
        affine: T.Optional[u.Quantity] = None,
        copy: bool = True,
        **kwargs: T.Any,
    ) -> None:

        keys = tuple(kwargs.keys())  # needed b/c pop changes size
        interp_kwargs = {k: kwargs.pop(k) for k in keys if k.startswith("interp_")}

        super().__init__(*args, copy=copy, **kwargs)

        # change frame to InterpolatedCoordinateFrame
        if not isinstance(self.frame, InterpolatedCoordinateFrame):
            self._sky_coord_frame = InterpolatedCoordinateFrame(
                self.frame, affine=affine, **interp_kwargs
            )

    def __call__(self, affine: T.Optional[u.Quantity] = None) -> SkyCoord:
        """Evaluate interpolated representation.

        Parameters
        ----------
        affine : `~astropy.units.Quantity` array-like
            The affine interpolation parameter.
            If None, returns representation points.

        Returns
        -------
        `SkyCoord`
            CoordinateFrame of type ``self.frame`` evaluated with `affine`
        """
        newsc = SkyCoord(self.frame(affine))
        return newsc

    def transform_to(
        self,
        frame: TH.FrameLikeType,
        merge_attributes: bool = True,
    ) -> InterpolatedSkyCoord:
        """Transform this coordinate to a new frame.

        The precise frame transformed to depends on ``merge_attributes``.
        If `False`, the destination frame is used exactly as passed in.
        But this is often not quite what one wants.  E.g., suppose one wants to
        transform an ICRS coordinate that has an obstime attribute to FK4; in
        this case, one likely would want to use this information. Thus, the
        default for ``merge_attributes`` is `True`, in which the precedence is
        as follows: (1) explicitly set (i.e., non-default) values in the
        destination frame; (2) explicitly set values in the source; (3) default
        value in the destination frame.

        Note that in either case, any explicitly set attributes on the source
        `SkyCoord` that are not part of the destination frame's definition are
        kept (stored on the resulting `SkyCoord`), and thus one can round-trip
        (e.g., from FK4 to ICRS to FK4 without loosing obstime).

        Parameters
        ----------
        frame : str, `BaseCoordinateFrame` class or instance, or `SkyCoord` instance
            The frame to transform this coordinate into.  If a `SkyCoord`, the
            underlying frame is extracted, and all other information ignored.
        merge_attributes : bool, optional
            Whether the default attributes in the destination frame are allowed
            to be overridden by explicitly set attributes in the source
            (see note above; default: `True`).

        Returns
        -------
        coord : `SkyCoord`
            A new object with this coordinate represented in the `frame` frame.

        Raises
        ------
        ValueError
            If there is no possible transformation route.
        """
        sc = coord.SkyCoord(self, copy=False)  # TODO, less jank
        nsc = sc.transform_to(frame, merge_attributes=merge_attributes)

        return self.__class__(nsc, affine=self.affine, copy=False)

    # -----------------------------------------------------
    # Separation

    def separation(
        self,
        point: TH.CoordinateType,
        *,
        interpolate: bool = True,
        affine: T.Optional[u.Quantity] = None,
    ) -> T.Union[coord.Angle, IUSU]:
        """Computes on-sky separation between this coordinate and another.

        .. note::

            If the ``other`` coordinate object is in a different frame, it is
            first transformed to the frame of this object. This can lead to
            unintuitive behavior if not accounted for. Particularly of note is
            that ``self.separation(other)`` and ``other.separation(self)`` may
            not give the same answer in this case.

        For more on how to use this (and related) functionality, see the
        examples in :doc:`astropy:/coordinates/matchsep`.

        Parameters
        ----------
        other : `~astropy.coordinates.SkyCoord` or `~astropy.coordinates.BaseCoordinateFrame`
            The coordinate to get the separation to.
        interpolated : bool, optional keyword-only
            Whether to return the separation as an interpolation, or as points.
        affine : `~astropy.units.Quantity` array-like or None, optional
            The affine interpolation parameter. If None (default), return
            angular width evaluated at all "tick" interpolation points.

        Returns
        -------
        sep : ~astropy.coordinates.Angle` or `~.interpolate.InterpolatedUnivariateSplinewithUnits`
            The on-sky separation between this and the ``other`` coordinate.

        Notes
        -----
        The separation is calculated using the Vincenty formula, which
        is stable at all locations, including poles and antipodes [1]_.

        .. [1] https://en.wikipedia.org/wiki/Great-circle_distance
        """
        return self._separation(point, angular=True, interpolate=interpolate, affine=affine)

    def separation_3d(
        self,
        point: TH.CoordinateType,
        *,
        interpolate: bool = True,
        affine: T.Optional[u.Quantity] = None,
    ) -> T.Union[coord.Distance, IUSU]:
        """
        Computes three dimensional separation between this coordinate
        and another.

        For more on how to use this (and related) functionality, see the
        examples in :doc:`astropy:/coordinates/matchsep`.

        Parameters
        ----------
        other : `~astropy.coordinates.SkyCoord` or `~astropy.coordinates.BaseCoordinateFrame`
            The coordinate to get the separation to.
        interpolated : bool, optional keyword-only
            Whether to return the separation as an interpolation, or as points.
        affine : `~astropy.units.Quantity` array-like or None, optional
            The affine interpolation parameter. If None (default), return
            angular width evaluated at all "tick" interpolation points.

        Returns
        -------
        sep : `~astropy.coordinates.Distance` or `~.interpolate.InterpolatedUnivariateSplinewithUnits`  # noqa: E501
            The real-space distance between these two coordinates.

        Raises
        ------
        ValueError
            If this or the other coordinate do not have distances.
        """
        return self._separation(point, angular=False, interpolate=interpolate, affine=affine)

    def _separation(
        self,
        point: SkyCoord,
        angular: bool,
        interpolate: bool,
        affine: T.Optional[u.Quantity],
    ) -> T.Union[coord.Angle, coord.Distance, IUSU]:
        """Separation helper function."""
        return InterpolatedCoordinateFrame._separation(
            self,
            point,
            angular=angular,
            interpolate=interpolate,
            affine=affine,
        )


##############################################################################
# END
