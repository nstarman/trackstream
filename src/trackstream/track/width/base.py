"""Widths base class."""

from __future__ import annotations

# STDLIB
from abc import abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from functools import singledispatchmethod
from typing import Any, ClassVar, TypeVar

# THIRD PARTY
import astropy.units as u
import numpy as np
from overload_numpy import NPArrayOverloadMixin, NumPyOverloader
from override_toformat import ToFormatOverloader, ToFormatOverloadMixin

__all__ = ["WidthBase"]


##############################################################################
# TYPING

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

    def __post_init__(self):
        qlen = None
        for f in fields(self):
            # t, d = _get_q_and_d(f.type)
            # if not inspect.isclass(t) and not issubclass(t, u.Quantity):
            #     continue

            q = getattr(self, f.name)
            qlen = np.shape(q) if qlen is None else qlen

            # if q.unit.physical_type != d:
            #     raise u.UnitsError(f"{f.name!r} must have dimensions of {d!r}")
            if np.shape(q) != qlen:
                raise ValueError(f"{f.name!r} must have length {qlen}")

    # ===============================================================

    @property
    @abstractmethod
    def units(self) -> u.StructuredUnit:
        """Structured unit of the fields' units."""
        raise NotImplementedError

    # ===============================================================
    # I/O

    @singledispatchmethod
    @classmethod
    def from_format(cls, data: object) -> Any:  # https://github.com/python/mypy/issues/11727
        """Construct this width instance from data of given type."""
        raise NotImplementedError("not dispatched")

    @from_format.register(Mapping)
    @classmethod
    def _from_format_dict(cls, data: Mapping[str, Any]) -> WidthBase:
        # From a Mapping.
        return cls(**data)

    # ===============================================================
    # Dunder methods

    def __len__(self):
        return len(getattr(self, fields(self)[0].name))

    @singledispatchmethod
    def __getitem__(self: Self, key: object) -> Self:
        return replace(self, **{f.name: getattr(self, f.name)[key] for f in fields(self)})
