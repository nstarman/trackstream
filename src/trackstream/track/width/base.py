##############################################################################
# IMPORTS

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

# LOCAL
from trackstream.utils.to_format_overload import ToFormatOverloader

__all__ = ["WidthBase"]


##############################################################################
# TYPING

Self = TypeVar("Self", bound="WidthBase")
T = TypeVar("T")


##############################################################################
# PARAMETERS

WB_FUNCS = NumPyOverloader()
TO_FORMAT = ToFormatOverloader()


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class WidthBase(NPArrayOverloadMixin):
    """ABC for all width classes."""

    NP_OVERLOADS: ClassVar[NumPyOverloader] = WB_FUNCS
    TO_FORMAT: ClassVar[ToFormatOverloader] = TO_FORMAT

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

    def to_format(self, format: type[T], /, *args: Any, **kwargs: Any) -> T:
        """Transform width to specified format.

        Parameters
        ----------
        format : type, positional-only
            The format type to which to transform this width.
        *args : Any
            Arguments into ``to_format``.
        **kwargs : Any
            Keyword-arguments into ``to_format``.

        Returns
        -------
        object
            Width transformed to specified type.

        Raises
        ------
        ValueError
            If format is not one of the recognized types.
        """
        if format not in self.TO_FORMAT:
            raise ValueError(f"format {format} is not known -- {self.TO_FORMAT.keys()}")

        out = self.TO_FORMAT[format](self).func(self, *args, **kwargs)
        return out

    # ===============================================================
    # Dunder methods

    def __len__(self):
        return len(getattr(self, fields(self)[0].name))

    @singledispatchmethod
    def __getitem__(self: Self, key: object) -> Self:
        return replace(self, **{f.name: getattr(self, f.name)[key] for f in fields(self)})
