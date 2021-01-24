# -*- coding: utf-8 -*-

"""Generic Coordinates."""


__all__ = [
    "GenericRepresentation",
    "GenericDifferential",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
from astropy.coordinates.representation import DIFFERENTIAL_CLASSES

##############################################################################
# PARAMETERS

# dict[coord.RepresentationOrDifferential, coord.RepresentationOrDifferential]
_GENERIC_REGISTRY = dict()


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

    base_representation = GenericRepresentation


# /class


##############################################################################
# Factories


def _make_generic_representation(rep_cls):

    if issubclass(rep_cls, GenericRepresentation):
        return rep_cls
    elif rep_cls in _GENERIC_REGISTRY:
        return _GENERIC_REGISTRY[rep_cls]

    cls = type(
        f"Generic{rep_cls.__name__}",
        (GenericRepresentation, rep_cls),
        dict(attr_classes=rep_cls.attr_classes),
    )

    _GENERIC_REGISTRY[rep_cls] = cls

    return cls


# /def


def _d_nth_suffix(n: int):
    """only works on n>=2"""
    if n == 2:
        suffix = "2nd"
    elif n == 3:
        suffix = "3rd"
    else:
        suffix = f"{n}th"

    return suffix


def _d_nth_prefix(k, n: int):
    """k is label d_X"""
    return k if n == 1 else f"d{n}_{k[2:]}"


def _make_generic_differential(dif_cls, n: int = 1):
    """Make Generic Differential.

    Parameters
    ----------
    dif_cls : class
        not instance

    n : int
        the differential level
        not used if dif_cls is GenericDifferential

    """
    if issubclass(dif_cls, GenericDifferential):
        return dif_cls
    elif n < 1:
        raise ValueError("n < 1")

    if n == 1:
        name = f"Generic{dif_cls.__name__}"
    else:
        dif_type = dif_cls.__name__[: -len("Differential")]
        name = f"Generic{dif_type}{_d_nth_suffix(n)}Differential"

    if dif_cls in _GENERIC_REGISTRY and n == 1:
        return _GENERIC_REGISTRY[dif_cls]
    elif name in _GENERIC_REGISTRY:
        return _GENERIC_REGISTRY[name]

    base_rep = dif_cls.base_representation

    if base_rep in _GENERIC_REGISTRY:
        generic_base_rep = _GENERIC_REGISTRY[base_rep]
    else:  # need to make Generic for base representation
        generic_base_rep = _make_generic_representation(base_rep)

    cls = type(
        name,
        (GenericDifferential, dif_cls),
        dict(
            base_representation=generic_base_rep,
            attr_classes=dif_cls.attr_classes,
        ),
    )

    if n == 1:
        _GENERIC_REGISTRY[dif_cls] = cls
    else:
        _GENERIC_REGISTRY[name] = cls

    return cls


# /def


def _make_generic_differential_for_representation(rep_cls, n: int = 1):

    rep_cls_name = rep_cls.__name__[: -len("Representation")]

    if n == 1:
        name = f"Generic{rep_cls_name}Differential"
    else:
        name = f"Generic{rep_cls_name}{_d_nth_suffix(n)}Differential"

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
