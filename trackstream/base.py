# -*- coding: utf-8 -*-

# STDLIB
from abc import ABCMeta
from typing import Optional, Type

# THIRD PARTY
from astropy.coordinates import BaseCoordinateFrame, BaseRepresentation

# LOCAL
from trackstream._type_hints import FrameLikeType
from trackstream.utils import resolve_framelike


class CommonBase(metaclass=ABCMeta):
    """Base class for most objects. Provides a frame and representation type.

    Parameters
    ----------
    frame : `~astropy.coordinates.BaseCoordinateFrame`, keyword-only
        The frame.
    representation_type : `astropy.coordinates.BaseRepresentation` or None, optional keyword-only
        The representation type for the `frame`. If `None` (default) uses
        the current representation type of the `frame`.
    """

    _frame: BaseCoordinateFrame

    def __init__(
        self,
        *,
        frame: FrameLikeType,
        representation_type: Optional[Type[BaseRepresentation]] = None
    ) -> None:
        # First resolve frame
        theframe = resolve_framelike(frame)
        # Now can get representation type
        representation_type = (
            representation_type if representation_type is not None else theframe.representation_type
        )
        # Set the frame, with the representation type
        self._frame: BaseCoordinateFrame
        self._frame = theframe.replicate_without_data()
        self._frame.representation_type = representation_type

    @property
    def frame(self) -> BaseCoordinateFrame:
        return self._frame

    @property
    def representation_type(self) -> Type[BaseRepresentation]:
        rt: Type[BaseRepresentation] = self._frame.representation_type
        return rt
