# -*- coding: utf-8 -*-

"""Interpolated Coordinates, Representations, and SkyCoords.

Astropy coordinate objects are a collection of points.
This module provides wrappers to interpolate each dimension with an affine
parameter

**TODO: Basic Example**

>>> affine = np.linspace(0, 1) * u.Myr


**TODO: and a plot**


**TODO: The interpolation can be transformed**


**TODO: and other operations**


**TODO: We can also do nice things like take derivatives**


**TODO: compute tangent vectors along the path**




"""

__all__ = [
    # interpolation classes
    "InterpolatedRepresentation",
    "InterpolatedDifferential",
    "InterpolatedCoordinateFrame",
    "InterpolatedSkyCoord",
]


##############################################################################
# IMPORTS

# BUILT-IN
import copy
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy.lib.recfunctions as rfn
from astropy.coordinates import CartesianRepresentation, SkyCoord
from astropy.coordinates.representation import _array2string
from astropy.utils.decorators import format_doc
from numpy import array_equal

# PROJECT-SPECIFIC
from .generic_coordinates import (
    GenericDifferential,
    _make_generic_differential,
    _make_generic_differential_for_representation,
)
from .interpolate import InterpolatedUnivariateSplinewithUnits as IUSU

# import operator

##############################################################################
# CODE
##############################################################################


def _find_first_best_compatible_differential(rep, n: int = 1):
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

    else:  # nothing matches, so we make a differential
        derivative_type = _make_generic_differential_for_representation(
            rep.__class__,
            n=n,
        )

    if n != 1:
        derivative_type = _make_generic_differential(derivative_type, n=n)

    return derivative_type


# /def


def _infer_derivative_type(rep, dif_unit, n: int = 1):
    """Infer the Differential class used in a derivative wrt time.

    If it can't infer the correct differential class, defaults
    to `~starkman_thesis.utils.generic_coordinates.GenericDifferential`.

    Checks compatible differentials for classes with matching
    names.


    Parameters
    ----------
    rep : `~astropy.coordinates.RepresentationOrDifferential` instance
        The representation object
    dif_unit : unit-like
        The differential unit

    n : int

    """
    unit = u.Unit(dif_unit)
    rep_cls = rep.__class__  # (store rep class for line length)

    # start by assuming the worst: we can't infer anything
    # this will never be returned
    derivative_type = GenericDifferential

    # Now check we can even do this: if can't make a better Generic
    # 1) can't for `Differentials` and stuff without compatible diffs
    if isinstance(rep, coord.BaseDifferential):
        derivative_type = _make_generic_differential(rep_cls, n=n + 1)
    # 2) can't for non-time derivatives
    elif unit.physical_type != "time":
        derivative_type = _make_generic_differential_for_representation(
            rep_cls,
            n=n,
        )

    else:  # Differentiating a Representation wrt time
        derivative_type = _find_first_best_compatible_differential(rep, n=n)

    return derivative_type


# /def

##############################################################################


class InterpolatedRepresentationOrDifferential:
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

    """

    def __new__(cls, *args, **kwargs):
        if cls is InterpolatedRepresentationOrDifferential:
            raise TypeError(f"Cannot instantiate a {cls}.")

        return super().__new__(cls)

    # /def

    def __init__(
        self,
        rep: coord.BaseRepresentationOrDifferential,
        affine,
        *,
        interps=None,
        derivative_type: T.Optional[coord.BaseDifferential] = None,
        **interp_kwargs,
    ):

        if not isinstance(rep, coord.BaseRepresentationOrDifferential):
            raise ValueError("Must instantiate `rep`.")

        affine = u.Quantity(affine, copy=False)  # ensure Quantity
        if not affine.ndim == 1:
            raise ValueError("`affine` must be 1-D.")
        elif len(affine) != len(rep):
            raise ValueError("`affine` must be same length as `rep`")

        self.data = rep
        self._affine = affine = u.Quantity(affine, copy=True)  # TODO copy?

        # The class to use when differentiating wrt to the affine parameter.
        if derivative_type is None:
            derivative_type = _infer_derivative_type(rep, affine.unit)
        # TODO better detection if derivative_type doesn't work!
        self._derivative_type = derivative_type
        self._derivatives = dict()

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
                self._interps[comp] = interp_cls(
                    affine, getattr(rep, comp), **interp_kwargs
                )

            # differentials information
            # these are stored in a dictionary with keys wrt time
            # ex : rep.differentials["s"] is a Velocity
            if hasattr(rep, "differentials"):
                for k, differential in rep.differentials.items():

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

        # /if

    # /def

    @property
    def affine(self):  # read-only
        return self._affine

    @property
    def _class_(self):
        return object.__class__(self)

    # /def

    def _realize_class(self, *args):
        return self._class_(
            *args, derivative_type=self.derivative_type, **self._interp_kwargs
        )

    # /def

    #################################################################
    # Interpolation Methods

    @property
    def derivative_type(self):
        """The class used when taking a derivative."""
        return self._derivative_type

    # /def

    @derivative_type.setter
    def derivative_type(self, value):
        """The class used when taking a derivative."""
        self._derivative_type = value
        self.clear_derivatives()

    # /def

    def clear_derivatives(self):
        """Return self, clearing cached derivatives."""
        if hasattr(self.data, "differentials"):
            keys = tuple(self.data.differentials.keys())
            for key in keys:
                if key.startswith("lambda "):
                    self.data.differentials.pop(key)

        return self

    def derivative(self, n=1):
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
            (k if k.startswith("d_") else "d_" + k): interp.derivative(n=n)(
                self.affine,
            )
            for k, interp in self._interps.items()
        }

        if n == 1:
            derivative_type = self.derivative_type
        else:
            derivative_type = _infer_derivative_type(
                self.data,
                self.affine.unit,
                n=n,
            )

        # make Differential
        deriv = derivative_type(**params)

        # interpolate
        # derivative_type = _infer_derivative_type(
        #     deriv, self.affine.unit, n=n + 1
        # )
        ideriv = InterpolatedDifferential(
            deriv,
            self.affine,
            # derivative_type=derivative_type,
            **self._interp_kwargs,
        )
        # TODO rare case when differentiating an integral of a Representation
        # then want to return an interpolated Representation!

        return ideriv

    # /def

    # def antiderivative(self, n=1):
    #     r"""Construct a new spline representing the derivative of this spline.

    #     .. todo:

    #         Allow for attaching the differentials?

    #         a differential should become a position!

    #         detect if differentiating wrt time, and use astropy Differential classes

    #     Parameters
    #     ----------
    #     n : int, optional
    #         Order of derivative to evaluate. Default: 1

    #     """
    #     # evaluate the spline on each argument of the position
    #     params = [
    #         interp.antiderivative(n=n)(self.affine)
    #         for k, interp in self._interps.items()
    #     ]

    #     deriv = GenericDifferential(*params)

    #     return self._class_(deriv, self.affine, **self._interp_kwargs)

    # # /def

    # def integral(self, a, b):
    #     """Return definite integral between two given points."""
    #     raise NotImplementedError("What does this even mean?")

    # # /def

    # ---------------------------------------------------------------
    # Convenience interpolation methods

    def headless_tangent_vectors(self):
        r"""

        :math:`\vec{x} + \partial_{\lambda} \vec{x}(\lambda) \Delta\lambda`

        """
        irep = self.represent_as(CartesianRepresentation)
        ideriv = irep.derivative(n=1)  # (interpolated)

        offset = CartesianRepresentation(*(ideriv.d_xyz * self.affine.unit))
        offset = offset.represent_as(self.__class__)  # transform back

        return self._realize_class(offset, self.affine)

    # /def

    def tangent_vectors(self):
        r"""

        :math:`\vec{x} + \partial_{\lambda} \vec{x}(\lambda) \Delta\lambda`

        """
        irep = self.represent_as(CartesianRepresentation)
        ideriv = irep.derivative(n=1)  # (interpolated)

        offset = CartesianRepresentation(*(ideriv.d_xyz * self.affine.unit))

        newirep = irep + offset
        newirep = newirep.represent_as(self.__class__)

        return self._realize_class(newirep, self.affine)

    # /def

    #################################################################
    # Mapping to Underlying Representation

    # ---------------------------------------------------------------
    # hidden methods

    @property
    def __class__(self):
        """Make class appear the same as the underlying Representation."""
        return self.data.__class__

    # /def

    def __getattr__(self, key):
        """Route everything to underlying Representation."""
        return getattr(self.data, key)

    # /def

    def __getitem__(self, key):
        """"""
        rep = self.data[key]
        afn = self.affine[key]
        return self._realize_class(rep, afn)

    # /def

    def __len__(self):
        return len(self.data)

    # /def

    def __repr__(self):
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

        aurep = str(self.affine.unit) or "[1]"

        _unitstr = self.data._unitstr
        if _unitstr:
            if _unitstr[0] == "(":
                unitstr = "in " + "(" + aurep + "| " + _unitstr[1:]
            else:
                unitstr = "in " + aurep + "| " + _unitstr
        else:
            unitstr = f"{aurep}| [1]"

        return "<Interpolated{} (lambda| {}) {:s}\n{}{}{}>".format(
            self.__class__.__name__,
            ", ".join(self.data.components),
            unitstr,
            prefixstr,
            arrstr,
            diffstr,
        )

    # /def

    def _scale_operation(self, op, *args):
        rep = self.data._scale_operation(op, *args)

        return self._realize_class(rep, self.affine)

    # ---------------------------------------------------------------
    # math methods

    def __add__(self, other):
        """Add other to an InterpolatedRepresentationOrDifferential

        If other:
        - point : add to data, keep affine the same, re-interpolate
        - vector : add to data, keep affine the same, re-interpolate
        - interpolated : must be same interpolation!

        """
        if isinstance(other, InterpolatedRepresentationOrDifferential):
            if not array_equal(other.affine, self.affine):
                raise ValueError(
                    "Can only add two InterpolatedRepresentationOrDifferential"
                    + " if the interpolation variables are the same.",
                )

        # add
        newrep = self.data.__add__(other)

        # now re-interpolate
        return self._realize_class(newrep, self.affine)

    # /def

    def __sub__(self, other):
        """Add other to an InterpolatedRepresentationOrDifferential

        If other:
        - point : add to data, keep affine the same, re-interpolate
        - vector : add to data, keep affine the same, re-interpolate
        - interpolated : must be same interpolation!

        """
        if isinstance(other, InterpolatedRepresentationOrDifferential):
            if not array_equal(other.affine, self.affine):
                raise ValueError(
                    "Can only add two InterpolatedRepresentationOrDifferential"
                    + " if the interpolation variables are the same.",
                )

        # add
        newrep = self.data.__sub__(other)

        # now re-interpolate
        return self._realize_class(newrep, self.affine)

    # /def

    def __mul__(self, other):
        """Add other to an InterpolatedRepresentationOrDifferential

        If other:
        - point : add to data, keep affine the same, re-interpolate
        - vector : add to data, keep affine the same, re-interpolate
        - interpolated : must be same interpolation!

        """
        if isinstance(other, InterpolatedRepresentationOrDifferential):
            if not array_equal(other.affine, self.affine):
                raise ValueError(
                    "Can only add two InterpolatedRepresentationOrDifferential"
                    + " if the interpolation variables are the same.",
                )

        # add
        newrep = self.data.__mul__(other)

        # now re-interpolate
        return self._realize_class(newrep, self.affine)

    # /def

    def __div__(self, other):
        """Add other to an InterpolatedRepresentationOrDifferential

        If other:
        - point : add to data, keep affine the same, re-interpolate
        - vector : add to data, keep affine the same, re-interpolate
        - interpolated : must be same interpolation!

        """
        if isinstance(other, InterpolatedRepresentationOrDifferential):
            if not array_equal(other.affine, self.affine):
                raise ValueError(
                    "Can only add two InterpolatedRepresentationOrDifferential"
                    + " if the interpolation variables are the same.",
                )

        # add
        newrep = self.data.__div__(other)

        # now re-interpolate
        return self._realize_class(newrep, self.affine)

    # /def

    def __truediv__(self, other):
        """Add other to an InterpolatedRepresentationOrDifferential

        If other:
        - point : add to data, keep affine the same, re-interpolate
        - vector : add to data, keep affine the same, re-interpolate
        - interpolated : must be same interpolation!

        """
        if isinstance(other, InterpolatedRepresentationOrDifferential):
            if not array_equal(other.affine, self.affine):
                raise ValueError(
                    "Can only add two InterpolatedRepresentationOrDifferential"
                    + " if the interpolation variables are the same.",
                )

        # add
        newrep = self.data.__truediv__(other)

        # now re-interpolate
        return self._realize_class(newrep, self.affine)

    # /def

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
    #         apply_method = lambda array: method(array, *args, **kwargs)
    #     else:
    #         apply_method = operator.methodcaller(method, *args, **kwargs)

    #     affine = apply_method(self.affine)

    #     return InterpolatedRepresentation(rep, affine)

    # # /def

    # ---------------------------------------------------------------
    # Specific wrappers

    # TODO just wrap self.data method with a wrapper?
    def from_cartesian(self, other):
        """Create a representation of this class from a Cartesian one.

        Parameters
        ----------
        other : `CartesianRepresentation` or `CartesianDifferential`
            The representation to turn into this class

            Note: the affine parameter of this class is used.

        Returns
        -------
        representation : object of this class
            A new representation of this class's type.

        """
        rep = self.data.from_cartesian(other)
        return self._class_(rep, self.affine, **self._interp_kwargs)

    # /def

    # TODO just wrap self.data method with a wrapper?
    def to_cartesian(self):
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

        """
        rep = self.data.to_cartesian()
        return self._class_(rep, self.affine, **self._interp_kwargs)

    # /def

    # def transform(self)

    def copy(self, *args, **kwargs):
        """Return an instance containing copies of the internal data.

        Parameters are as for :meth:`~numpy.ndarray.copy`.

        .. todo::

            this uses BaseRepresentation._apply, see if that may be modified
            instead

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

    # /def


# /class


# -------------------------------------------------------------------


class InterpolatedRepresentation(InterpolatedRepresentationOrDifferential):
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

    def __new__(cls, representation, *args, **kwargs):

        # need to special case since it has different methods
        if isinstance(representation, CartesianRepresentation):
            self = super().__new__(InterpolatedCartesianRepresentation)
        else:
            self = super().__new__(cls)

        return self

    # /def

    def __call__(self, affine=None):
        """Evaluate interpolated representation.

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

    # /def

    # ---------------------------------------------------------------

    # TODO just wrap self.data method with a wrapper?
    def represent_as(self, other_class, differential_class=None):
        """Convert coordinates to another representation.

        If the instance is of the requested class, it is returned unmodified.
        By default, conversion is done via Cartesian coordinates.
        Also note that orientation information at the origin is *not* preserved by
        conversions through Cartesian coordinates. See the docstring for
        `~astropy.coordinates.BaseRepresentation.represent_as()` for an example.

        Parameters
        ----------
        other_class : `~astropy.coordinates.BaseRepresentation` subclass
            The type of representation to turn the coordinates into.
        differential_class : dict of `~astropy.coordinates.BaseDifferential`, optional
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
        return self._class_(rep, self.affine, **self._interp_kwargs)

    # /def

    # TODO just wrap self.data method with a wrapper?
    def with_differentials(self, differentials):
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
        if not differentials:  # from source code
            return self

        rep = self.data.with_differentials(differentials)
        return self._realize_class(rep, self.affine)

    # /def

    # TODO just wrap self.data method with a wrapper?
    def without_differentials(self):
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

    # /def

    def derivative(self, n=1):
        r"""Construct a new spline representing the derivative of this spline.

        Parameters
        ----------
        n : int, optional
            Order of derivative to evaluate. Default: 1

        """
        if f"lambda {n}" in self._derivatives:
            return self._derivatives[f"lambda {n}"]

        ideriv = super().derivative(n=n)

        # cache in differentials
        self._derivatives[f"lambda {n}"] = ideriv

        return ideriv

    # /def


# /class


class InterpolatedCartesianRepresentation(InterpolatedRepresentation):

    # TODO just wrap self.data method with a wrapper?
    def transform(self, matrix):
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

        We can start off by creating a cartesian representation object:

            >>> from astropy import units as u
            >>> from astropy.coordinates import CartesianRepresentation
            >>> rep = CartesianRepresentation([1, 2] * u.pc,
            ...                               [2, 3] * u.pc,
            ...                               [3, 4] * u.pc)

        We now create a rotation matrix around the z axis:

            >>> from astropy.coordinates.matrix_utilities import rotation_matrix
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

    # /def


# /class


# -------------------------------------------------------------------


class InterpolatedDifferential(InterpolatedRepresentationOrDifferential):
    def __call__(self, affine=None):
        """Evaluate interpolated representation.

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
        if affine is None:  # If None, returns representation as-is.
            return self.data
        # else:

        affine = u.Quantity(affine)  # need to ensure Quantity

        # evaluate the spline on each argument of the position
        params = {n: interp(affine) for n, interp in self._interps.items()}

        return self.data.__class__(**params)

    # /def

    # ---------------------------------------------------------------

    # TODO just wrap self.data method with a wrapper?
    def represent_as(self, other_class, base):
        """Convert coordinates to another representation.

        If the instance is of the requested class, it is returned unmodified.
        By default, conversion is done via cartesian coordinates.

        Parameters
        ----------
        other_class : `~astropy.coordinates.BaseRepresentation` subclass
            The type of representation to turn the coordinates into.
        base : instance of ``self.base_representation``
            Base relative to which the differentials are defined.  If the other
            class is a differential representation, the base will be converted
            to its ``base_representation``.
        """
        rep = self.data.represent_as(other_class, base=base)

        # don't pass on the derivative_type
        return self._class_(rep, self.affine, **self._interp_kwargs)

    # /def


# /class

#####################################################################


class InterpolatedCoordinateFrame:
    """Wrapper for Coordinate Frame, adding affine interpolations.

    .. todo::

        - override all the methods, mapping to underlying CoordinateFrame

        - Separate from this: allow for ICRS(InterpolatedRepresentation()) to work

    Parameters
    ----------
    data : InterpolatedRepresentation or Representation or CoordinateFrame
        For either an InterpolatedRepresentation or Representation
        the kwarg 'frame' must also be specified.
        If CoordinateFrame, then 'frame' is ignored.
    affine : Quantity array-like (optional)
        if not a Quantity, one is assigned.
        Only used if data is not already interpolated.
        If data is not interpolated, this is needed.


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
        data,
        affine=None,
        *,
        interps=None,
        **interp_kwargs,
    ):

        rep = data.data

        if isinstance(rep, InterpolatedRepresentation):
            pass
        elif isinstance(rep, coord.BaseRepresentation):
            if affine is None:
                raise ValueError(
                    "`data` is not already interpolated. "
                    "Need to pass a Quantity array for `affine`.",
                )

            rep = InterpolatedRepresentation(
                rep, affine=affine, interps=interps, **interp_kwargs
            )
        else:
            raise TypeError(
                "`data` must be type "
                + "<InterpolatedRepresentation> or <BaseRepresentation>",
            )

        self._interp_kwargs = interp_kwargs  # TODO double check need this
        self.frame = data.realize_frame(rep)

    # /def

    def __call__(self, affine=None):
        """Evaluate interpolated representation.

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

    # /def

    @property
    def _class_(self):
        return object.__class__(self)

    # /def

    def _realize_class(self, *args):
        return self._class_(*args, affine=self.affine, **self._interp_kwargs)

    # /def

    #################################################################
    # Interpolation Methods
    # Mapped to underlying Representation

    @format_doc(InterpolatedRepresentationOrDifferential.derivative.__doc__)
    def derivative(self, n=1):
        """Take nth derivative wrt affine parameter."""
        return self.frame.data.derivative(n=n)

    # /def

    #################################################################
    # Mapping to Underlying Representation

    @property
    def affine(self):  # read-only
        return self.frame.affine  # or self.frame.data.affine?

    # /def

    #################################################################
    # Mapping to Underlying CoordinateFrame

    @property
    def __class__(self):
        """Make class appear the same as the underlying Representation."""
        return self.frame.__class__

    # /def

    def __getattr__(self, key):
        """Route everything to underlying Representation."""
        return getattr(self.frame, key)

    # /def

    def __len__(self):
        return len(self.frame)

    # /def

    def __iter__(self):
        return iter(self.frame)

    # /def

    def transform_to(self, new_frame):
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

    # /def

    def copy(self):
        interp_kwargs = self._interp_kwargs.copy()
        frame = self.frame.realize_frame(self.data)
        return InterpolatedCoordinateFrame(
            frame,
            affine=self.affine.copy(),
            interps=None,
            **interp_kwargs,
        )

    # /def

    def _frame_attrs_repr(self):  # FIXME!!
        return "FIXME"

    # /def


# /class


#####################################################################


class InterpolatedSkyCoord(SkyCoord):
    """Interpolated SkyCoord."""

    def __init__(self, *args, affine=None, copy=True, **kwargs):

        keys = tuple(kwargs.keys())  # needed b/c pop changes size
        interp_kwargs = {
            k: kwargs.pop(k) for k in keys if k.startswith("interp_")
        }

        super().__init__(*args, copy=copy, **kwargs)

        # change frame to InterpolatedCoordinateFrame
        if not isinstance(self.frame, InterpolatedCoordinateFrame):
            self.frame = InterpolatedCoordinateFrame(
                self.frame, affine=affine, **interp_kwargs
            )

    # /def

    def __call__(self, affine=None):
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
        newsc = SkyCoord(self, copy=True)
        newsc.frame = self.frame(affine)

        return newsc

    # /def

    # ---------------------------------------------------------------
    # Mapping to Underlying SkyCoord

    def separation(self, other):
        """
        Computes on-sky separation between this coordinate and another.

        .. note::

            If the ``other`` coordinate object is in a different frame, it is
            first transformed to the frame of this object. This can lead to
            unintuitive behavior if not accounted for. Particularly of note is
            that ``self.separation(other)`` and ``other.separation(self)`` may
            not give the same answer in this case.

        For more on how to use this (and related) functionality, see the
        examples in
        https://docs.astropy.org/en/stable/coordinates/matchsep.html.

        Parameters
        ----------
        other : |SkyCoord| or |CoordinateFrame|
            The coordinate to get the separation to.

        Returns
        -------
        sep : `~astropy.coordinates.Angle`
            The on-sky separation between this and the ``other`` coordinate.

        Notes
        -----
        The separation is calculated using the Vincenty formula, which
        is stable at all locations, including poles and antipodes [1]_.

        .. [1] https://en.wikipedia.org/wiki/Great-circle_distance

        """
        return super().separation(other)

    # /def

    def separation_3d(self, other):
        """
        Computes three dimensional separation between this coordinate
        and another.

        For more on how to use this (and related) functionality, see the
        examples in
        https://docs.astropy.org/en/stable/coordinates/matchsep.html.

        Parameters
        ----------
        other : |SkyCoord| or |CoordinateFrame|
            The coordinate to get the separation to.

        Returns
        -------
        sep : `~astropy.coordinates.Distance`
            The real-space distance between these two coordinates.

        Raises
        ------
        ValueError
            If this or the other coordinate do not have distances.

        """
        return super().separation_3d(other)

    # /def

    def match_to_catalog_sky(self, catalogcoord, nthneighbor=1):
        """
        Finds the nearest on-sky matches of this coordinate in a set of
        catalog coordinates.

        For more on how to use this (and related) functionality, see the
        examples in
        https://docs.astropy.org/en/stable/coordinates/matchsep.html.

        Parameters
        ----------
        catalogcoord : |SkyCoord| or |CoordinateFrame|
            The base catalog in which to search for matches. Typically this
            will be a coordinate object that is an array (i.e.,
            ``catalogcoord.isscalar == False``)
        nthneighbor : int, optional
            Which closest neighbor to search for.  Typically ``1`` is
            desired here, as that is correct for matching one set of
            coordinates to another. The next likely use case is ``2``,
            for matching a coordinate catalog against *itself* (``1``
            is inappropriate because each point will find itself as the
            closest match).

        Returns
        -------
        idx : integer array
            Indices into ``catalogcoord`` to get the matched points for
            each of this object's coordinates. Shape matches this
            object.
        sep2d : `~astropy.coordinates.Angle`
            The on-sky separation between the closest match for each
            element in this object in ``catalogcoord``. Shape matches
            this object.
        dist3d : `~astropy.units.Quantity`
            The 3D distance between the closest match for each element
            in this object in ``catalogcoord``. Shape matches this
            object. Unless both this and ``catalogcoord`` have associated
            distances, this quantity assumes that all sources are at a
            distance of 1 (dimensionless).

        Notes
        -----
        This method requires `SciPy <https://www.scipy.org/>`_ to be
        installed or it will fail.

        See Also
        --------
        astropy.coordinates.match_coordinates_sky
        SkyCoord.match_to_catalog_3d

        """
        return super().match_coordinates_sky(
            catalogcoord,
            nthneighbor=nthneighbor,
        )

    # /def

    # just needed to modify the docstring
    def match_to_catalog_3d(self, catalogcoord, nthneighbor=1):
        """
        Finds the nearest 3-dimensional matches of this coordinate to a set
        of catalog coordinates.

        This finds the 3-dimensional closest neighbor, which is only different
        from the on-sky distance if ``distance`` is set in this object or the
        ``catalogcoord`` object.

        For more on how to use this (and related) functionality, see the
        examples in
        https://docs.astropy.org/en/stable/coordinates/matchsep.html.

        Parameters
        ----------
        catalogcoord : |SkyCoord| or |CoordinateFrame|
            The base catalog in which to search for matches. Typically this
            will be a coordinate object that is an array (i.e.,
            ``catalogcoord.isscalar == False``)
        nthneighbor : int, optional
            Which closest neighbor to search for.  Typically ``1`` is
            desired here, as that is correct for matching one set of
            coordinates to another.  The next likely use case is
            ``2``, for matching a coordinate catalog against *itself*
            (``1`` is inappropriate because each point will find
            itself as the closest match).

        Returns
        -------
        idx : integer array
            Indices into ``catalogcoord`` to get the matched points for
            each of this object's coordinates. Shape matches this
            object.
        sep2d : `~astropy.coordinates.Angle`
            The on-sky separation between the closest match for each
            element in this object in ``catalogcoord``. Shape matches
            this object.
        dist3d : `~astropy.units.Quantity`
            The 3D distance between the closest match for each element
            in this object in ``catalogcoord``. Shape matches this
            object.

        Notes
        -----
        This method requires `SciPy <https://www.scipy.org/>`_ to be
        installed or it will fail.

        See Also
        --------
        astropy.coordinates.match_coordinates_3d
        SkyCoord.match_to_catalog_sky

        """
        return super().match_to_catalog_3d(
            catalogcoord,
            nthneighbor=nthneighbor,
        )

    # just needed to modify the docstring
    def search_around_sky(self, searcharoundcoords, seplimit):
        """
        Searches for all coordinates in this object around a supplied set of
        points within a given on-sky separation.

        This is intended for use on `~astropy.coordinates.SkyCoord` objects
        with coordinate arrays, rather than a scalar coordinate.  For a scalar
        coordinate, it is better to use
        `~astropy.coordinates.SkyCoord.separation`.

        For more on how to use this (and related) functionality, see the
        examples in
        https://docs.astropy.org/en/stable/coordinates/matchsep.html.

        Parameters
        ----------
        searcharoundcoords : |SkyCoord| or |CoordinateFrame|
            The coordinates to search around to try to find matching points in
            this `SkyCoord`. This should be an object with array coordinates,
            not a scalar coordinate object.
        seplimit : `~astropy.units.Quantity` with angle units
            The on-sky separation to search within.

        Returns
        -------
        idxsearcharound : integer array
            Indices into ``searcharoundcoords`` that match the
            corresponding elements of ``idxself``. Shape matches
            ``idxself``.
        idxself : integer array
            Indices into ``self`` that match the
            corresponding elements of ``idxsearcharound``. Shape matches
            ``idxsearcharound``.
        sep2d : `~astropy.coordinates.Angle`
            The on-sky separation between the coordinates. Shape matches
            ``idxsearcharound`` and ``idxself``.
        dist3d : `~astropy.units.Quantity`
            The 3D distance between the coordinates. Shape matches
            ``idxsearcharound`` and ``idxself``.

        Notes
        -----
        This method requires `SciPy <https://www.scipy.org/>`_ to be
        installed or it will fail.

        In the current implementation, the return values are always sorted in
        the same order as the ``searcharoundcoords`` (so ``idxsearcharound`` is
        in ascending order).  This is considered an implementation detail,
        though, so it could change in a future release.

        See Also
        --------
        astropy.coordinates.search_around_sky
        SkyCoord.search_around_3d

        """
        return super().search_around_sky(searcharoundcoords, seplimit)

    # /def

    # just needed to modify the docstring
    def search_around_3d(self, searcharoundcoords, distlimit):
        """
        Searches for all coordinates in this object around a supplied set of
        points within a given 3D radius.

        This is intended for use on `~astropy.coordinates.SkyCoord` objects
        with coordinate arrays, rather than a scalar coordinate.  For a scalar
        coordinate, it is better to use
        `~astropy.coordinates.SkyCoord.separation_3d`.

        For more on how to use this (and related) functionality, see the
        examples in
        https://docs.astropy.org/en/stable/coordinates/matchsep.html.

        Parameters
        ----------
        searcharoundcoords : |SkyCoord| or |CoordinateFrame|
            The coordinates to search around to try to find matching points in
            this `SkyCoord`. This should be an object with array coordinates,
            not a scalar coordinate object.
        distlimit : `~astropy.units.Quantity` with distance units
            The physical radius to search within.

        Returns
        -------
        idxsearcharound : integer array
            Indices into ``searcharoundcoords`` that match the
            corresponding elements of ``idxself``. Shape matches
            ``idxself``.
        idxself : integer array
            Indices into ``self`` that match the
            corresponding elements of ``idxsearcharound``. Shape matches
            ``idxsearcharound``.
        sep2d : `~astropy.coordinates.Angle`
            The on-sky separation between the coordinates. Shape matches
            ``idxsearcharound`` and ``idxself``.
        dist3d : `~astropy.units.Quantity`
            The 3D distance between the coordinates. Shape matches
            ``idxsearcharound`` and ``idxself``.

        Notes
        -----
        This method requires `SciPy <https://www.scipy.org/>`_ to be
        installed or it will fail.

        In the current implementation, the return values are always sorted in
        the same order as the ``searcharoundcoords`` (so ``idxsearcharound`` is
        in ascending order).  This is considered an implementation detail,
        though, so it could change in a future release.

        See Also
        --------
        astropy.coordinates.search_around_3d
        SkyCoord.search_around_sky

        """
        return super().search_around_3d(searcharoundcoords, distlimit)

    # /def


# /class


##############################################################################
# END
