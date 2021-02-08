# -*- coding: utf-8 -*-

"""Generic Coordinates."""


__all__ = [
    "GenericRepresentation",
    "GenericDifferential",
]


##############################################################################
# IMPORTS

# BUILT-IN
import typing as T

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates.representation import DIFFERENTIAL_CLASSES
from starkman_thesis.type_hints import RepresentationType

##############################################################################
# PARAMETERS

_GENERIC_REGISTRY: T.Dict[T.Union[object, str], object] = dict()


##############################################################################
# CODE
##############################################################################


class GenericRepresentation(coord.BaseRepresentation):
    """Generic representation of a point in a 3D coordinate system.

    Parameters
    ----------
    q1, q2, q3 : `~astropy.units.Quantity` or subclass
        The components of the 3D points. The names are the keys and the
        subclasses the values of the attr_classes attribute.

    differentials : dict, `~astropy.coordinates.BaseDifferential`, optional
        Any differential classes that should be associated with this
        representation. The input must either be a single
        `~astropy.coordinates.BaseDifferential` subclass instance, or a
        dictionary with keys set to a string representation of the SI unit
        with which the differential (derivative) is taken. For example, for a
        velocity differential on a positional representation, the key would be
        ``'s'`` for seconds, indicating that the derivative is a time
        derivative.
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.

    Notes
    -----
    All representation classes should subclass this base representation class,
    and define an ``attr_classes`` attribute, an `~collections.OrderedDict`
    which maps component names to the class that creates them. They must also
    define a ``to_cartesian`` method and a ``from_cartesian`` class method. By
    default, transformations are done via the cartesian system, but classes
    that want to define a smarter transformation path can overload the
    ``represent_as`` method. If one wants to use an associated differential
    class, one should also define ``unit_vectors`` and ``scale_factors``
    methods (see those methods for details).

    """

    attr_classes = dict(q1=u.Quantity, q2=u.Quantity, q3=u.Quantity)


# /class


# -------------------------------------------------------------------


class GenericDifferential(coord.BaseDifferential):
    r"""A base class representing differentials of representations.

    These represent differences or derivatives along each component.
    E.g., for physics spherical coordinates, these would be
    :math:`\delta r, \delta \theta, \delta \phi`.

    Parameters
    ----------
    d_q1, d_q2, d_q3 : `~astropy.units.Quantity` or subclass
        The components of the 3D differentials.  The names are the keys and the
        subclasses the values of the ``attr_classes`` attribute.
    copy : bool, optional
        If `True` (default), arrays will be copied. If `False`, arrays will
        be references, though possibly broadcast to ensure matching shapes.

    """

    base_representation: RepresentationType = GenericRepresentation


# /class


##############################################################################
# Factories


def _make_generic_representation(
    rep_cls: T.Union[RepresentationType, GenericRepresentation],
) -> GenericRepresentation:
    """Factory for making a generic form of a representation.

    Parameters
    ----------
    rep_cls : |Representation| or `GenericRepresentation`
        Representation class for which to make generic.

    Returns
    -------
    `GenericRepresentation` subclass
        Generic form of `rep_cls`.
        If `rep_cls` is already generic, return it unchanged.
        Subclasses are cached in a registry.

    """
    # 1) check if it's already generic
    if issubclass(rep_cls, GenericRepresentation):
        return rep_cls

    # 2) check if it's cached
    elif rep_cls in _GENERIC_REGISTRY:
        return _GENERIC_REGISTRY[rep_cls]

    # 3) need to make
    else:
        # dynamically define class
        # name: Generic{X}
        # bases: both generic and actual representation
        # attributes: copies `attr_classes`
        cls = type(
            f"Generic{rep_cls.__name__}",
            (GenericRepresentation, rep_cls),
            dict(attr_classes=rep_cls.attr_classes),
        )

        # cache b/c can only define the same Rep/Dif once
        _GENERIC_REGISTRY[rep_cls] = cls

        # also store in locals
        # THIRD PARTY
        from starkman_thesis.utils import generic_coordinates

        setattr(generic_coordinates, cls.__name__, cls)
        generic_coordinates.__all__.append(cls.__name__)

        return cls


# /def


def _ordinal(n: int) -> str:
    """Return suffix for ordinal.

    https://codegolf.stackexchange.com/a/74047

    Parameters
    ----------
    n : int
        Must be >= 2

    Returns
    -------
    str
        Ordinal form `n`. Ex 1 -> '1st', 2 -> '2nd', 3 -> '3rd'.

    """
    i: int = n % 5 * (n % 100 ^ 15 > 4 > n % 10)
    return str(n) + "tsnrhtdd"[i::4]  # noqa: E203


# /def


def _make_generic_differential(
    dif_cls: T.Union[coord.BaseDifferential, GenericDifferential],
    n: int = 1,
) -> GenericDifferential:
    """Make Generic Differential.

    Parameters
    ----------
    dif_cls : |Differential| or `GenericDifferential` class
        Differential class for which to make generic.

    n : int
        The differential level.
        Not used if `dif_cls` is GenericDifferential

    Returns
    -------
    `GenericDifferential`
        Generic form of `dif_cls`.
        If `dif_cls` is already generic, return it unchanged.
        Subclasses are cached in a registry.

    """
    # 1) check if it's already generic
    if issubclass(dif_cls, GenericDifferential):
        return dif_cls

    # 2) check if `n` is too small to make a differential
    elif n < 1:
        raise ValueError("n < 1")

    # 3) make name for generic.
    # a) special case for n=1
    if n == 1:
        name = f"Generic{dif_cls.__name__}"
    # b) higher ordinal
    else:
        dif_type = dif_cls.__name__[: -len("Differential")]
        name = f"Generic{dif_type}{_ordinal(n)}Differential"

    # A) check if cached
    # i) special case for n=1
    if dif_cls in _GENERIC_REGISTRY and n == 1:
        return _GENERIC_REGISTRY[dif_cls]
    # ii) higher ordinal
    elif name in _GENERIC_REGISTRY:
        return _GENERIC_REGISTRY[name]

    # B) make generic
    # get base representation from differential class.
    base_rep = dif_cls.base_representation
    # and then get the generic form
    if base_rep in _GENERIC_REGISTRY:
        generic_base_rep = _GENERIC_REGISTRY[base_rep]
    else:  # need to make Generic for base representation
        generic_base_rep = _make_generic_representation(base_rep)

    # make generic differential
    # name: constructed in 3)
    # bases: both generic and actual differential
    # attributes: attr_classes, base_representation
    cls = type(
        name,
        (GenericDifferential, dif_cls),
        dict(
            attr_classes=dif_cls.attr_classes,
            base_representation=generic_base_rep,
        ),
    )

    # cache. either by class or by name
    if n == 1:
        _GENERIC_REGISTRY[dif_cls] = cls
    else:
        _GENERIC_REGISTRY[name] = cls

    # also store in locals
    # THIRD PARTY
    from starkman_thesis.utils import generic_coordinates

    setattr(generic_coordinates, cls.__name__, cls)
    generic_coordinates.__all__.append(cls.__name__)

    return cls


# /def


def _make_generic_differential_for_representation(
    rep_cls: RepresentationType,
    n: int = 1,
) -> GenericDifferential:
    """Make generic differential given a representation.

    Parameters
    ----------
    rep_cls : |Representation|
    n : int
        Must be >= 1

    Returns
    -------
    `GenericDifferential`
        Of ordinal `n`

    """
    rep_cls_name: str = rep_cls.__name__[: -len("Representation")]

    if n == 1:
        name = f"Generic{rep_cls_name}Differential"
    else:
        name = f"Generic{rep_cls_name}{_ordinal(n)}Differential"

    if name in _GENERIC_REGISTRY:
        return _GENERIC_REGISTRY[name]
    elif rep_cls_name.lower() in DIFFERENTIAL_CLASSES:
        return _make_generic_differential(
            DIFFERENTIAL_CLASSES[rep_cls_name.lower()],
            n=n,
        )

    # else:

    cls = type(
        name,
        (GenericDifferential, rep_cls),
        dict(base_representation=rep_cls),
    )

    _GENERIC_REGISTRY[name] = cls

    return cls


# /def


##############################################################################
# END
