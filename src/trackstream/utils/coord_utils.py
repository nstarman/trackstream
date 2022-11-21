"""Coordinates Utilities."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import functools
from typing import TYPE_CHECKING, Any, Literal, TypeVar

# THIRD PARTY
import astropy.units as u
import numpy as np
import numpy.lib.recfunctions as rfn
from astropy.coordinates import (
    BaseCoordinateFrame,
    BaseDifferential,
    BaseRepresentation,
    SkyCoord,
    frame_transform_graph,
)

if TYPE_CHECKING:
    # THIRD PARTY
    from typing_extensions import TypeAlias

__all__ = ["parse_framelike", "get_frame", "deep_transform_to", "f2q"]

##############################################################################
# PARAMETERS

_FT = TypeVar("_FT", bound=BaseCoordinateFrame)
_RT = TypeVar("_RT", bound=BaseRepresentation)

##############################################################################
# CODE
##############################################################################


@functools.singledispatch
def parse_framelike(frame: object) -> Any:  # https://github.com/python/mypy/issues/11727
    """Determine the frame and return a blank instance.

    Parameters
    ----------
    frame : |Frame| or str or Any, positional-only
        If |Frame|, replicates without data. If `str`, uses astropy parsers to
        determine frame class.

    type_error : bool, optional
        Whether to raise TypeError if ``frame`` is not one of the allowed types.

    Returns
    -------
    frame : |Frame| instance
        Replicated without data.

    Raises
    ------
    TypeError
        If ``frame`` is not one of the allowed types and 'type_error' is True.

    See Also
    --------
    get_frame
        Determine the frame and return a blank instance.

    Parameters
    ----------
    frame : |Frame| or str or Any, positional-only
        If |Frame|, replicates without data. If `str`, uses astropy parsers to
        determine frame class.

    type_error : bool, optional
        Whether to raise TypeError if ``frame`` is not one of the allowed types.

    Returns
    -------
    frame : |Frame| instance
        Replicated without data.

    Raises
    ------
    TypeError
        If ``frame`` is not one of the allowed types and 'type_error' is True.

    See Also
    --------
    get_frame

    Examples
    --------
    ``parse_framelike`` single-dispatches on the argument, so these examples are
    incomplete.

        >>> try: parse_framelike(object())
        ... except NotImplementedError: print("NotImplemented")
        NotImplemented

        >>> parse_framelike('icrs')
        <ICRS Frame>

        >>> import astropy.units as u
        >>> from astropy.coordinates import ICRS, SkyCoord
        >>> parse_framelike(ICRS())
        <ICRS Frame>

        >>> parse_framelike(ICRS(ra=10 * u.deg, dec=10*u.deg))
        <ICRS Frame>
    """
    raise NotImplementedError(f"frame type {type(frame)} not dispatched")


@functools.singledispatch
def get_frame(frame: object) -> BaseCoordinateFrame:
    """Determine the frame and return a blank instance.

    Parameters
    ----------
    frame : |Frame| or str or Any, positional-only
        If |Frame|, replicates without data.
        If `str`, uses astropy parsers to determine frame class.

    type_error : bool, optional
        Whether to raise TypeError if ``frame`` is not one of the allowed types.

    Returns
    -------
    frame : |Frame| instance
        Replicated without data.

    Raises
    ------
    TypeError
        If ``frame`` is not one of the allowed types and 'type_error' is True.

    See Also
    --------
    parse_framelike

    Examples
    --------
    ``parse_framelike`` single-dispatches on the argument, so these examples are
    incomplete.

        >>> try: get_frame(object())
        ... except NotImplementedError: print("NotImplemented")
        NotImplemented

        >>> get_frame('icrs')
        <ICRS Frame>

        >>> import astropy.units as u
        >>> from astropy.coordinates import ICRS, SkyCoord
        >>> get_frame(ICRS())
        <ICRS Frame>

        >>> get_frame(ICRS(ra=10 * u.deg, dec=10*u.deg))
        <ICRS Frame>

        >>> get_frame(SkyCoord(ICRS(ra=10 * u.deg, dec=10*u.deg)))
        <ICRS Frame>
    """
    raise NotImplementedError(f"frame type {type(frame)} not dispatched")


@get_frame.register(str)
@parse_framelike.register(str)
def _parse_framelike_str(name: str) -> BaseCoordinateFrame:
    frame_cls = frame_transform_graph.lookup_name(name)

    if frame_cls is None:
        frame_names = frame_transform_graph.get_names()
        raise ValueError(f"Coordinate frame name {name!r} is not a known coordinate frame ({sorted(frame_names)})")

    return frame_cls()


@get_frame.register(BaseCoordinateFrame)
@parse_framelike.register(BaseCoordinateFrame)
def _parse_framelike_frame(frame: BaseCoordinateFrame) -> BaseCoordinateFrame:
    return frame.replicate_without_data(
        representation_type=frame.representation_type, differential_type=frame.differential_type
    )


@get_frame.register
def _get_frame_skycoord(frame: SkyCoord) -> BaseCoordinateFrame:
    return frame.frame.replicate_without_data(
        representation_type=frame.frame.representation_type,
        differential_type=frame.frame.differential_type,
    )


# ===================================================================

_DifT: TypeAlias = type[BaseDifferential] | None | Literal["base"]


@functools.singledispatch
def deep_transform_to(
    crd: object,
    frame: BaseCoordinateFrame,
    representation_type: type[BaseRepresentation],
    differential_type: _DifT,
) -> Any:  # https://github.com/python/mypy/issues/11727
    """Transform a coordinate to a frame and representation type.

    For speed, Astropy transformations can be shallow. This function does
    ``.transform_to(frame, representation_type=representation_type)`` and makes
    sure all the underlying data is actually in the desired representation type.

    Parameters
    ----------
    crd : SkyCoord or BaseCoordinateFrame
    frame : BaseCoordinateFrame
        The frame to which to tranform `crd`.
    representation_type : BaseRepresentationresentation class
        The type of representation.
    differential_type : BaseDifferentialferential class or None or 'base', optional
        Class in which any velocities should be represented. If equal to `base`
        (default), inferred from the base class.If `None`, all velocity
        information is dropped.

    Returns
    -------
    crd : SkyCoord or BaseCoordinateFrame
        Transformed to ``frame`` and ``representation_type``.
    """
    raise NotImplementedError("not dispatched")


@deep_transform_to.register
def _deep_transform_frame(
    crd: BaseCoordinateFrame,
    frame: BaseCoordinateFrame,
    representation_type: type[BaseRepresentation],
    differential_type: _DifT,
) -> BaseCoordinateFrame:
    # Get representation, with differential possibly determined by the representation.
    dt = None if "s" not in crd.data.differentials else differential_type
    r = crd.represent_as(representation_type, s=dt)

    # Get the actual differential.
    dt = dt if dt != "base" else type(r.differentials["s"])

    # Transform to frame, then realize from data.
    f = crd.transform_to(frame)
    frame = f.realize_frame(r, representation_type=representation_type, differential_type=dt, copy=False)

    return frame


@deep_transform_to.register
def _deep_transform_skycoord(
    crd: SkyCoord,
    frame: BaseCoordinateFrame,
    representation_type: type[BaseRepresentation],
    differential_type: _DifT,
) -> SkyCoord:
    # SkyCoord from transformation
    return SkyCoord(
        deep_transform_to(
            crd.frame, frame=frame, representation_type=representation_type, differential_type=differential_type
        ),
        copy=False,
    )


# ===================================================================


def _f2q_helper(crds: BaseCoordinateFrame | SkyCoord, which: str) -> u.Quantity:
    """Helper for ``f2q``.

    Parameters
    ----------
    crds : BaseCoordinateFrame or SkyCoord
        The coordinates for which to get the array.
    which : str
        Which coordinate to get, e.g 'base' for positions, 's' for
        differentials.

    Returns
    -------
    Quantity
        Structured quantity.
    """
    # Get representation components.
    #
    # Note: ``get_representation_component_names`` sometimes returns names which
    # aren't actually an attribute, so need to filter.
    rcls = crds.get_representation_cls(which)
    comps = tuple(c for c, rc in crds.get_representation_component_names(which).items() if rc in rcls.attr_classes)

    # Build structured array
    dt = np.dtype([(c, float) for c in comps])
    arr = np.empty(crds.shape, dtype=dt)
    us = [None] * len(comps)
    for i, c in enumerate(comps):
        v = getattr(crds, c)
        arr[c] = v
        us[i] = v.unit

    return u.Quantity(arr, dtype=dt, unit=u.StructuredUnit(tuple(us)))


def f2q(crds: BaseCoordinateFrame | SkyCoord, flatten: bool = False) -> u.Quantity:
    """Return coordinate as a structured, flattened Quantity.

    Parameters
    ----------
    crds : Frame or SkyCoord
        Coordinates.
    flatten : bool, optional
        Whether to flatten, by default `False`.

    Returns
    -------
    Quantity
        Flattened coordinates.

    Examples
    --------
    >>> import astropy.units as u
    >>> from astropy.coordinates import ICRS
    >>> c = ICRS(ra=1*u.deg, dec=2*u.deg, pm_ra_cosdec=3*u.mas/u.yr, pm_dec=4*u.mas/u.yr)
    >>> f2q(c)
    <Quantity ((1., 2., 1.), (3., 4., 0)) ((deg, deg, ), (mas / yr, mas / yr, mas / (rad yr)))>
    """
    q = _f2q_helper(crds, "base")  # positions

    # Velocities
    HAS_V = False if ("s" not in crds.data.differentials) else True

    # Output, possibly flattening
    if flatten and not HAS_V:
        out = q
    elif flatten and HAS_V:
        p = _f2q_helper(crds, "s")
        out = rfn.merge_arrays((q, p), flatten=True)
    elif HAS_V:  # not flattened
        p = _f2q_helper(crds, "s")  # kinemtatics
        su = u.StructuredUnit((q.unit, p.unit))
        arr = np.empty(crds.shape, dtype=[("length", q.dtype), ("speed", p.dtype)])
        out = u.Quantity(arr, unit=su)
        out["length"] = q
        out["speed"] = p
    else:  # no flattened and has no velocities
        su = u.StructuredUnit((q.unit,))
        out = u.Quantity(np.empty(crds.shape, dtype=[("length", q.dtype)]), unit=su)
        out["length"] = q

    return out
