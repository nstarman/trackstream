"""Stream arm classes.

Stream arms are descriptors on a `trackstrea.Stream` class.

"""

##############################################################################
# IMPORTS

from __future__ import annotations

# LOCAL
from trackstream.common import CollectionBase
from trackstream.stream.core import StreamArm

__all__ = ["StreamArmsBase", "StreamArms"]


##############################################################################


class StreamArmsBase(CollectionBase[StreamArm]):
    """Base class for a collection of stream arms."""


class StreamArms(StreamArmsBase):
    """A collection of stream arms.

    See Also
    --------
    Stream
        An object that brings together 2 stream arms, but acts like 1 stream.
    """
