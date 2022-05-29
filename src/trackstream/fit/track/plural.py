# -*- coding: utf-8 -*-

"""Stream track fit result."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from types import MappingProxyType
from typing import TYPE_CHECKING, Optional

# THIRD PARTY
from attrs import define

# LOCAL
from .core import StreamArmTrack, StreamArmTrackBase
from .visualization import StreamArmsTrackBasePlotDescriptor
from trackstream.base import CollectionBase
from trackstream.fit.path import concatenate_paths

if TYPE_CHECKING:
    # LOCAL
    from trackstream.fit.path import Path
    from trackstream.stream.plural import Stream

__all__ = ["StreamTrack"]


##############################################################################
# CODE
##############################################################################


@define(frozen=True, slots=False, init=False)
class StreamArmsTrackBase(CollectionBase[StreamArmTrack], StreamArmTrackBase):

    plot = StreamArmsTrackBasePlotDescriptor()

    @property
    def stream(self) -> MappingProxyType:
        """The `~trackstream.Stream`, or `None` if the weak reference is broken."""
        return MappingProxyType({k: v.stream for k, v in self.items()})


#####################################################################


@define(frozen=True, slots=False, init=False)
class StreamArmsTracks(StreamArmsTrackBase):
    """Collection of ``StreamArmTrack``."""


#####################################################################


@define(frozen=True, slots=False, init=False, kw_only=True)
class StreamTrack(StreamArmTrack, StreamArmsTrackBase):
    """``StreamArmTrack`` for ``Stream``.

    Currently only works for 2 arms.
    """

    plot = StreamArmsTrackBasePlotDescriptor()

    def __init__(
        self, stream: Stream, path: Path, *, name: Optional[str] = None, meta: Optional[dict] = None
    ) -> None:
        tracks = {k: arm.track for k, arm in stream.items()}
        self.__attrs_init__(data=tracks, stream_ref=stream, path=path, name=name, meta=meta)

    @classmethod
    def from_stream(
        cls,
        stream: "Stream",
        *,
        name: str,
        meta: Optional[dict] = None,
    ):
        tracks = {k: arm.track for k, arm in stream.items()}

        # TODO! concatenation that intelligently stitches
        # and works with more than two
        if len(tracks) > 2:
            raise NotImplementedError

        path = concatenate_paths(
            tuple(track.path for track in tracks.values()), name=name + " Path"
        )
        soms = {k: v.som for k, v in tracks.items()}
        kalmans = {k: v.kalman for k, v in tracks.items()}

        if meta is None:
            meta = {}
        meta["som"] = soms
        meta["kalman"] = kalmans

        return cls(stream, path, name=name, meta=meta)

    @property
    def stream(self) -> "Stream":
        """The `~trackstream.Stream`, or `None` if the weak reference is broken."""
        strm = self._stream_ref()
        if strm is None:
            raise AttributeError("the reference to the stream is broken")
        return strm
