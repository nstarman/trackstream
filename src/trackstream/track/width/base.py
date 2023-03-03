"""Widths base class."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

import numpy as np
from overload_numpy import NPArrayOverloadMixin, NumPyOverloader
from override_toformat import ToFormatOverloader, ToFormatOverloadMixin

__all__ = ["WidthBase"]

if TYPE_CHECKING:
    from astropy.units import StructuredUnit


Self = TypeVar("Self", bound="WidthBase")
T = TypeVar("T")


##############################################################################
# PARAMETERS

WB_FUNCS = NumPyOverloader()
FMT_OVERLOADS = ToFormatOverloader()


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class WidthBase(NPArrayOverloadMixin, ToFormatOverloadMixin):
    """ABC for all width classes."""

    NP_OVERLOADS: ClassVar[NumPyOverloader] = WB_FUNCS
    FMT_OVERLOADS: ClassVar[ToFormatOverloader] = FMT_OVERLOADS

    def __post_init__(self) -> None:
        qlen = None
        for f in fields(self):
            # if not inspect.isclass(t) and not issubclass(t, u.Quantity):

            q = getattr(self, f.name)
            qlen = np.shape(q) if qlen is None else qlen

            # if q.unit.physical_type != d:
            if np.shape(q) != qlen:
                msg = f"{f.name!r} must have length {qlen}"
                raise ValueError(msg)

    # ===============================================================

    @property
    @abstractmethod
    def units(self) -> StructuredUnit:
        """Structured unit of the fields' units."""
        raise NotImplementedError

    # ===============================================================
    # I/O

    @singledispatchmethod
    @classmethod
    def from_format(cls, data: object) -> Any:  # noqa: ARG003
        """Construct this width instance from data of given type."""
        # see https://github.com/python/mypy/issues/11727 for return Any
        msg = "not dispatched"
        raise NotImplementedError(msg)

    @from_format.register(Mapping)
    @classmethod
    def _from_format_dict(cls, data: Mapping[str, Any]) -> WidthBase:
        # From a Mapping.
        return cls(**data)

    # ===============================================================
    # Dunder methods

    def __len__(self) -> int:
        return len(getattr(self, fields(self)[0].name))

    @singledispatchmethod
    def __getitem__(self: Self, key: object) -> Self:
        return replace(self, **{f.name: getattr(self, f.name)[key] for f in fields(self)})
