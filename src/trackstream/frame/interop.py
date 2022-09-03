"""Interface frame fitting with main stream library."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import asdict, replace
from functools import singledispatch
from typing import Any, Callable, NoReturn, TypeVar

# THIRD PARTY
import astropy.units as u
from astropy.utils.decorators import format_doc

# LOCAL
from trackstream.stream.core import StreamArm, SupportsFrame
from trackstream.stream.plural import StreamArms, StreamArmsBase
from trackstream.stream.stream import Stream, StreamArmsDescriptor

__all__: list[str] = []


##############################################################################
# TYPING

Self = TypeVar("Self", bound=SupportsFrame)


##############################################################################
# CODE
##############################################################################


def add_method_to_cls(cls: type, attr: str | None) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
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


@singledispatch
def fit_stream(
    self: object,
    rot0: u.Quantity[u.deg] | None = u.Quantity(0, u.deg),
    *,
    force: bool = False,
    **kwargs: Any,
) -> NoReturn:
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
        A copy, with `frame` replaced.

    Raises
    ------
    TypeError
        If a system frame was given at the object's initialization and ``force``
        is not True.
    """
    raise NotImplementedError("not dispatched")


@add_method_to_cls(StreamArm, "fit_frame")
@fit_stream.register(StreamArm)
@format_doc(fit_stream.__doc__)
def _fit_stream_StreamArm(
    self: StreamArm,
    rot0: u.Quantity[u.deg] | None = u.Quantity(0, u.deg),
    *,
    force: bool = False,
    **kwargs: Any,
) -> StreamArm:
    if self.frame is not None and not force:
        raise TypeError("a system frame was given at initialization. Use ``force`` to re-fit.")

    # LOCAL
    from trackstream.frame.fit import fit_frame

    # fit_frame uses single-dispatch to get the correct output type
    result = fit_frame(self, rot0=rot0, **kwargs)

    # Make new stream(arm)
    newstream = replace(self, frame=result.frame)
    newstream._cache["frame_fit_result"] = result
    newstream.flags.set(**asdict(self.flags))

    return newstream


@add_method_to_cls(StreamArmsBase, "fit_frame")
@fit_stream.register(StreamArmsBase)
@format_doc(fit_stream.__doc__)
def _fit_stream_StreamArmsBase(
    self: StreamArmsBase,
    rot0: u.Quantity[u.deg] | None = u.Quantity(0, u.deg),
    *,
    force: bool = False,
    **kwargs: Any,
) -> StreamArmsBase:
    if not force:
        for n, f in self.frame.items():
            if f is not None:
                raise TypeError(f"a system frame was given for {n} at initialization. Use ``force`` to re-fit.")

    # LOCAL
    from trackstream.frame.fit import fit_frame

    # fit_frame uses single-dispatch to get the correct output type
    results = fit_frame(self, rot0=rot0, **kwargs)

    # New Stream, with frame set. Need to set for each contained arm.
    data = {}
    for k, arm in self.items():
        newarm = replace(arm, frame=results[k].frame)
        newarm._cache["frame_fit_result"] = results[k]
        newarm.flags.set(**asdict(arm.flags))
        data[k] = newarm

    newstream = type(self)(data)

    return newstream


@add_method_to_cls(StreamArmsDescriptor, "fit_frame")
@fit_stream.register(StreamArmsDescriptor)
def _fit_stream_StreamArmsDescriptor(
    self: StreamArmsDescriptor,
    rot0: u.Quantity[u.deg] | None = u.Quantity(0, u.deg),
    *,
    force: bool = False,
    **kwargs: Any,
) -> StreamArms:
    return fit_stream(StreamArms(dict(self.items())), rot0=rot0, force=force, **kwargs)


@add_method_to_cls(Stream, "fit_frame")
@fit_stream.register(Stream)
@format_doc(fit_stream.__doc__)
def _fit_stream_Stream(
    self: Stream,
    rot0: u.Quantity[u.deg] | None = u.Quantity(0, u.deg),
    *,
    force: bool = False,
    **kwargs: Any,
) -> Stream:
    if self.frame is not None and not force:
        raise TypeError("a system frame was given at initialization. Use ``force`` to re-fit.")

    # LOCAL
    from trackstream.frame.fit import fit_frame

    # fit_frame uses single-dispatch to get the correct output type
    result = fit_frame(self, rot0=rot0, **kwargs)

    # New Stream, with frame set. Need to set for each contained arm.
    data = {}
    for k, arm in self.items():
        newarm = replace(arm, frame=result.frame)
        newarm._cache["frame_fit_result"] = result
        newarm.flags.set(**asdict(arm.flags))
        data[k] = newarm

    newstream = type(self)(data, name=self.name)
    newstream._cache["frame_fit_result"] = result
    newstream.flags.set(**self.flags.asdict())

    return newstream
