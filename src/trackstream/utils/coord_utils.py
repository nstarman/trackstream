"""Coordinates Utilities."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import functools
from typing import TYPE_CHECKING, Literal, NoReturn, TypeVar, Union

# THIRD PARTY
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

__all__ = [
    "parse_framelike",
    "get_frame",
    "deep_transform_to",
]

##############################################################################
# PARAMETERS

_FT = TypeVar("_FT", bound=BaseCoordinateFrame)
_RT = TypeVar("_RT", bound=BaseRepresentation)

##############################################################################
# CODE
##############################################################################


@functools.singledispatch
def parse_framelike(frame: object) -> NoReturn:
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
    get_frame
    """ """Determine the frame and return a blank instance.

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
    get_frame
    """
    raise NotImplementedError(f"frame type {type(frame)} not dispatched")


@functools.singledispatch
def get_frame(frame: object) -> NoReturn:
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
    """
    raise NotImplementedError(f"frame type {type(frame)} not dispatched")


@get_frame.register
@parse_framelike.register
def _parse_framelike_str(name: str) -> BaseCoordinateFrame:  # noqa: F811
    frame_cls = frame_transform_graph.lookup_name(name)

    if frame_cls is None:
        frame_names = frame_transform_graph.get_names()
        raise ValueError(f"Coordinate frame name {name!r} is not a known coordinate frame ({sorted(frame_names)})")

    return frame_cls()


@get_frame.register
@parse_framelike.register
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

_DifT: TypeAlias = Union[type[BaseDifferential], None, Literal["base"]]


@functools.singledispatch
def deep_transform_to(
    crd: object,
    frame: BaseCoordinateFrame,
    representation_type: type[BaseRepresentation],
    differential_type: _DifT,
) -> NoReturn:
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
        Class in which any velocities should be represented. If equal to ‘base’
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
    f = crd.transform_to(frame)

    dt = None if "s" not in crd.data.differentials else differential_type
    r = crd.represent_as(representation_type, s=dt)

    dt = dt if dt != "base" else type(r.differentials["s"])

    frame = f.realize_frame(r, representation_type=representation_type, differential_type=dt, copy=False)

    return frame


@deep_transform_to.register
def _deep_transform_skycoord(
    crd: SkyCoord,
    frame: BaseCoordinateFrame,
    representation_type: type[BaseRepresentation],
    differential_type: _DifT,
) -> SkyCoord:
    return SkyCoord(
        deep_transform_to(
            crd.frame, frame=frame, representation_type=representation_type, differential_type=differential_type
        ),
        copy=False,
    )
