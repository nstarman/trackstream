# -*- coding: utf-8 -*-

"""Core Functions."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import re
from functools import cached_property
from typing import TYPE_CHECKING, List, Optional, Tuple, cast

# THIRD PARTY
from astropy.coordinates import BaseCoordinateFrame, SkyCoord
from astropy.table import Column, QTable
from numpy import ndarray

# LOCAL
from .base import StreamBase
from trackstream.utils.descriptors import InstanceDescriptor

if TYPE_CHECKING:
    # LOCAL
    from trackstream.stream.core import Stream  # noqa: F401

__all__: List[str] = []


##############################################################################
# CODE
##############################################################################


class StreamArmDescriptor(InstanceDescriptor["Stream"], StreamBase):
    """Descriptor on a `Stream` to have substreams describing a stream arm.

    This is an instance-level descriptor, so most attributes / methods point to
    corresponding methods on the parent instance.

    Attributes
    ----------
    full_name : str
        Full name of the stream arm, including the parent name.
    has_data : bool
        Boolean of whether this arm has data.
    index : `astropy.table.Column`
        Boolean array of which stars in the parent table are in this arm.
    """

    @property
    def full_stream(self) -> "Stream":
        """Return the full stream"""
        return self._enclosing

    @property
    def name(self) -> str:
        # Get the attribute name for this descriptor, filtering out None.
        attr_name = list(filter(None, re.split(r"(\d+)", self._enclosing_attr)))
        # e.g. arm1 -> ["arm", "1"]
        return " ".join(attr_name)

    @cached_property
    def full_name(self) -> str:
        """Full name of the stream arm, including the parent name."""
        parent_name: str = pn if isinstance(pn := self._enclosing.name, str) else "Stream"
        name_parts: Tuple[str, str] = (parent_name, self.name)
        return ", ".join(name_parts)

    # -------------------------------------------

    @property
    def index(self) -> ndarray:
        """Boolean array of which stars in the parent table are in this arm."""
        tailcolumn: Column = self._enclosing.data["tail"]
        idx: ndarray = tailcolumn == self._enclosing_attr
        return idx

    @cached_property
    def has_data(self) -> bool:
        """Boolean of whether this arm has data."""
        return any(self.index)

    @property
    def data(self) -> QTable:
        """Return subset of full stream table that is for this arm."""
        if not self.has_data:
            raise Exception(f"{self._enclosing_attr} has no data")
        return self._enclosing.data[self.index]

    # -------------------------------------------

    @cached_property
    def data_frame(self) -> BaseCoordinateFrame:
        return self._enclosing.data_frame

    @cached_property
    def frame(self) -> Optional[BaseCoordinateFrame]:
        return self._enclosing.frame

    @cached_property
    def origin(self) -> SkyCoord:
        return self._enclosing.origin

    @property
    def coords(self) -> SkyCoord:
        """The coordinates of the arm."""
        arm = cast(SkyCoord, self._enclosing.coords[self.index])
        return arm

    # ===============================================================
    # Misc

    @cached_property
    def _data_max_lines(self) -> int:  # type: ignore
        data_max_lines = self._enclosing._data_max_lines
        return data_max_lines
