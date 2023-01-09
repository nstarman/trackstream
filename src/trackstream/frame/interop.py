"""Interface frame fitting with main stream library."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import asdict, replace
from functools import singledispatch
from typing import Any, Callable, TypeVar

# THIRD PARTY
import astropy.units as u

# LOCAL
from trackstream._typing import SupportsFrame
from trackstream.stream.core import StreamArm
from trackstream.stream.plural import StreamArms, StreamArmsBase
from trackstream.stream.stream import Stream, StreamArmsDescriptor

__all__: list[str] = []


##############################################################################
# TYPING

Self = TypeVar("Self", bound=SupportsFrame)


##############################################################################
# CODE
##############################################################################


def _add_method_to_cls(cls: type, attr: str | None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Add a method to a class.

    .. todo::

        This is unknown to static type checkers.
        Add a mypy plugin to fix this.

    Parameters
    ----------
    cls : type
        The class to which the attribute will be added.
    attr : str or None
        The name of the added attribute.
        If `None` (default), use the method name.

    Returns
    -------
    decorator : callable[[Callable[..., Any]], Callable[..., Any]]
        One argument function that accepts and returns the method.
        The method is added to ``cls`` as ``setattr(cls, name, meth)``
    """

    def decorator(meth: Callable[..., Any]) -> Callable[..., Any]:
        """Add method to specified class as specified name.

        Parameters
        ----------
        meth : callable[..., Any]
            The method to add to the class.

        Returns
        -------
        metho : callable[..., Any]
            The same method, unmodified.
        """
        name = attr if attr is not None else meth.__name__
        setattr(cls, name, meth)

        return meth

    return decorator


# see https://github.com/python/mypy/issues/11727 for return Any
@singledispatch
def fit_stream(
    self: object,  # noqa: ARG001
    rot0: u.Quantity[u.deg] | None = u.Quantity(0, u.deg),  # noqa: ARG001
    *,
    force: bool = False,  # noqa: ARG001
    **kwargs: Any,  # noqa: ARG001
) -> Any:
    """Convenience method to fit a frame to a stream.

    The frame is an on-sky rotated frame. To prevent a frame from being fit, the
    desired frame should be passed to the Stream constructor at initialization.

    Parameters
    ----------
    rot0 : Quantity or None.
        Initial guess for rotation.

    force : bool, optional keyword-only
        Whether to force a frame fit. Default `False`. Will only fit if a frame
        was not specified at initialization.

    **kwargs : Any
        Passed to fitter.

    Returns
    -------
    object
        A copy of the stream, with `frame` replaced.

    Raises
    ------
    TypeError
        If a system frame was given at the object's initialization and ``force``
        is not True.
    """
    msg = "not dispatched"
    raise NotImplementedError(msg)


@_add_method_to_cls(StreamArm, "fit_frame")
@fit_stream.register(StreamArm)
def _fit_stream_StreamArm(
    arm: StreamArm,
    rot0: u.Quantity[u.deg] | None = u.Quantity(0, u.deg),
    *,
    force: bool = False,
    **kwargs: Any,
) -> StreamArm:
    if arm.frame is not None and not force:
        msg = "a system frame was given at initialization. Use ``force`` to re-fit."
        raise TypeError(msg)

    # LOCAL
    from trackstream.frame.fit import fit_frame

    # fit_frame uses single-dispatch to get the correct output type
    result = fit_frame(arm, rot0=rot0, **kwargs)

    # Make new stream(arm)
    newstream = replace(arm, frame=result.frame, origin=arm.origin.transform_to(result.frame))
    newstream._cache["frame_fit_result"] = result
    newstream.flags.set(**asdict(arm.flags))

    return newstream


_fit_stream_StreamArm.__doc__ = fit_stream.__doc__
# NOTE: `format_doc` is untyped, so untypes wrapped funcs


@_add_method_to_cls(StreamArmsBase, "fit_frame")
@fit_stream.register(StreamArmsBase)
def _fit_stream_StreamArmsBase(
    arms: StreamArmsBase,
    rot0: u.Quantity[u.deg] | None = u.Quantity(0, u.deg),
    *,
    force: bool = False,
    **kwargs: Any,
) -> StreamArmsBase:
    if not force:
        for n, f in arms.frame.items():
            if f is not None:
                msg = f"a system frame was given for {n} at initialization. Use ``force`` to re-fit."
                raise TypeError(msg)

    # LOCAL
    from trackstream.frame.fit import fit_frame

    # fit_frame uses single-dispatch to get the correct output type
    results = fit_frame(arms, rot0=rot0, **kwargs)

    # New Stream, with frame set. Need to set for each contained arm.
    data = {}
    for k, arm in arms.items():
        newarm = replace(arm, frame=results[k].frame, origin=arm.origin.transform_to(results[k].frame))
        newarm._cache["frame_fit_result"] = results[k]
        newarm.flags.set(**asdict(arm.flags))
        data[k] = newarm

    return type(arms)(data)


_fit_stream_StreamArmsBase.__doc__ = fit_stream.__doc__
# NOTE: `format_doc` is untyped, so untypes wrapped funcs


@_add_method_to_cls(StreamArmsDescriptor, "fit_frame")
@fit_stream.register(StreamArmsDescriptor)
def _fit_stream_StreamArmsDescriptor(
    arms_descr: StreamArmsDescriptor,
    rot0: u.Quantity[u.deg] | None = u.Quantity(0, u.deg),
    *,
    force: bool = False,
    **kwargs: Any,
) -> StreamArms:
    """Fit StreamArmsDescriptor as a StreamArms (a collection of arms)."""
    return fit_stream(StreamArms(dict(arms_descr.items())), rot0=rot0, force=force, **kwargs)


@_add_method_to_cls(Stream, "fit_frame")
@fit_stream.register(Stream)
def _fit_stream_Stream(
    stream: Stream,
    rot0: u.Quantity[u.deg] | None = u.Quantity(0, u.deg),
    *,
    force: bool = False,
    **kwargs: Any,
) -> Stream:
    if stream.frame is not None and not force:
        msg = "a system frame was given at initialization. Use ``force`` to re-fit."
        raise TypeError(msg)

    # LOCAL
    from trackstream.frame.fit import fit_frame

    # fit_frame uses single-dispatch to get the correct output type
    result = fit_frame(stream, rot0=rot0, **kwargs)

    # New Stream, with frame set. Need to set for each contained arm.
    data = {}
    for k, arm in stream.items():
        newarm = replace(arm, frame=result.frame, origin=stream.origin.transform_to(result.frame))
        newarm._cache["frame_fit_result"] = result
        newarm.flags.set(**asdict(arm.flags))
        data[k] = newarm

    newstream = type(stream)(data, name=stream.name)
    newstream._cache["frame_fit_result"] = result
    newstream.flags.set(**stream.flags.asdict())

    return newstream


_fit_stream_Stream.__doc__ = fit_stream.__doc__
# NOTE: `format_doc` is untyped, so untypes wrapped funcs
