"""Stream track fit result."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import weakref
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

# THIRD PARTY
import numpy as np
from astropy.utils.metadata import MetaAttribute

# LOCAL
from trackstream.common import CollectionBase
from trackstream.stream.stream import Stream
from trackstream.track.core import StreamArmTrack, StreamArmTrackBase

if TYPE_CHECKING:
    # LOCAL
    from trackstream.stream.base import StreamLikeT
    from trackstream.track.fit.path import Path

__all__: list[str] = []


##############################################################################
# CODE
##############################################################################


class StreamArmsTrackBase(CollectionBase[StreamArmTrack["StreamLikeT"]], StreamArmTrackBase["StreamLikeT"]):
    """Base class for stream arms tracks."""

    @property
    def stream(self) -> MappingProxyType[str, StreamLikeT]:
        """The `~trackstream.Stream`, or `None` if the weak reference is broken."""
        return MappingProxyType({k: v.stream for k, v in self.items()})


#####################################################################


class StreamArmsTracks(StreamArmsTrackBase["StreamLikeT"]):
    """Collection of ``StreamArmTrack``."""


#####################################################################


class StreamTrack(StreamArmTrack[Stream], StreamArmsTrackBase[Stream]):
    """``StreamArmTrack`` for ``Stream``.

    Currently only works for 2 arms.
    """

    def __init__(
        self,
        stream: Stream,
        tracks: Mapping[str, StreamArmTrack[StreamLikeT]],
        path: Path,
        *,
        name: str | None = None,
        meta: dict | None = None,
    ) -> None:
        object.__setattr__(self, "path", path)
        object.__setattr__(self, "name", name)

        # set the MetaAttribute(s)
        self._meta: dict[str, Any]  # set by MetaData
        object.__setattr__(self, "_meta", {} if meta is None else meta)
        for attr in list(self._meta):
            descr = getattr(self.__class__, attr, None)
            if isinstance(descr, MetaAttribute):
                descr.__set__(self, self._meta.pop(attr))

        # Initialize
        self.__post_init__(stream_ref=stream)
        StreamArmsTrackBase.__init__(self, tracks)

    def __post_init__(self, stream_ref: weakref.ReferenceType[Stream] | Stream) -> None:
        self._stream_ref: weakref.ReferenceType[Stream]
        sref = weakref.ref(stream_ref) if not isinstance(stream_ref, weakref.ReferenceType) else stream_ref
        object.__setattr__(self, "_stream_ref", sref)

    # ---------------------------------------------------------------

    @classmethod
    def from_stream(cls, stream: Stream, *, name: str, meta: dict | None = None) -> StreamTrack:
        """Track from a `trackstream.stream.Stream`."""
        # Get StreamArmTrack from each stream arm.
        tracks = {k: arm.track for k, arm in stream.items()}

        # TODO! concatenation that works with more than two streams
        if len(tracks) == 1:
            path = next(iter(tracks.values())).path
        elif len(tracks) == 2:
            path = np.concatenate(tuple(track.path for track in tracks.values()))
        else:
            raise NotImplementedError

        # Get SOMs & Kalman
        if meta is None:
            meta = {}
        meta["som"] = {k: v.som for k, v in tracks.items()}
        meta["kalman"] = {k: v.kalman for k, v in tracks.items()}

        return cls(stream, tracks, path, name=name, meta=meta)

    @property
    def stream(self) -> Stream:
        """The `~trackstream.Stream`, or `None` if the weak reference is broken."""
        strm = self._stream_ref()
        if strm is None:
            msg = "the reference to the stream is broken"
            raise AttributeError(msg)
        return strm
