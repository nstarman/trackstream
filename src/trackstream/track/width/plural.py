##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from collections.abc import Mapping
from dataclasses import fields
from functools import singledispatchmethod
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Iterator,
    KeysView,
    MutableMapping,
    NoReturn,
    TypeVar,
    ValuesView,
    cast,
)

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import BaseRepresentation
from numpy.lib.recfunctions import merge_arrays

# LOCAL
from trackstream.track.utils import is_structured
from trackstream.track.width.base import TO_FORMAT, WidthBase
from trackstream.track.width.core import BASEWIDTH_KIND, LENGTH
from trackstream.utils.to_format_overload import ToFormatOverloader

if TYPE_CHECKING:
    # LOCAL
    from .interpolated import InterpolatedWidths

__all__ = ["Widths"]

##############################################################################
# TYPING

T = TypeVar("T")
W1 = TypeVar("W1", bound="WidthBase")
W2 = TypeVar("W2", bound="WidthBase")


##############################################################################
# CODE
##############################################################################


class Widths(MutableMapping[u.PhysicalType, W1]):
    # TODO! make work with Interpolated

    TO_FORMAT: ClassVar[ToFormatOverloader] = TO_FORMAT

    def __init__(self, widths: dict[u.PhysicalType, W1]) -> None:
        # Validate that have the necessary base of the tower
        if LENGTH not in widths:
            raise ValueError("need positions")

        self._spaces: dict[u.PhysicalType, W1] = widths

    @singledispatchmethod
    def __getitem__(self, key: object) -> NoReturn:
        raise NotImplementedError("not dispatched")

    # ===============================================================

    def represent_as(self, width_type: type[W1], point: BaseRepresentation) -> W1:
        # # LOCAL
        # from trackstream.track.width.transforms import represent_as

        # return represent_as(self, width_type, point)
        raise NotImplementedError("TODO!")

    def interpolated(self, affine: u.Quantity) -> InterpolatedWidths:
        if len(self) < 2:
            raise ValueError("cannot interpolate; too short.")
        # LOCAL
        from trackstream.track.width.interpolated import InterpolatedWidths

        return InterpolatedWidths.from_format(self, affine=affine)

    # ===============================================================
    # Mapping

    @staticmethod
    def _get_key(key: str | u.PhysicalType) -> u.PhysicalType:
        k = key if isinstance(key, u.PhysicalType) else cast(u.PhysicalType, u.get_physical_type(key))
        return k

    @__getitem__.register(str)
    @__getitem__.register(u.PhysicalType)
    def _getitem_key(self, key: str | u.PhysicalType) -> W1:
        return self._spaces[self._get_key(key)]

    def __setitem__(self, k: str | u.PhysicalType, v: W1) -> None:
        self._spaces[self._get_key(k)] = v

    def __delitem__(self, k: str | u.PhysicalType) -> None:
        del self._spaces[self._get_key(k)]

    def __iter__(self) -> Iterator[u.PhysicalType]:
        return iter(self._spaces)

    def __len__(self) -> int:
        return len(self._spaces)

    def keys(self) -> KeysView[u.PhysicalType]:
        return self._spaces.keys()

    def values(self) -> ValuesView[W1]:
        return self._spaces.values()

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._spaces!r})"

    # ===============================================================
    # I/O

    @singledispatchmethod
    @classmethod
    def from_format(cls, data: object) -> Widths:
        # copy over from `data`
        ws: dict[u.PhysicalType, W1] = {}
        for wcls, k in BASEWIDTH_KIND.items():
            try:
                w = wcls.from_format(data)
            except Exception:
                continue
            if k in ws and len(fields(ws[k])) > len(fields(w)):
                continue
            ws[k] = w
        return cls(ws)

    @from_format.register(Mapping)
    @classmethod
    def _from_format_mapping(cls, data: Mapping[str | u.PhysicalType, Any]) -> Widths:
        # copy over from `data`
        ws: dict[u.PhysicalType, W1] = {}
        for wcls, k in BASEWIDTH_KIND.items():
            # Try to get a WidthBase from mapping
            w = data.get(k, data.get(str(k._physical_type_list[0]), None))
            # Maybe it's a bunch of fields
            if w is None:
                try:
                    w = wcls.from_format(data)
                except Exception:
                    continue
            # skip if there's a better match
            if k in ws and len(fields(ws[k])) > len(fields(w)):
                continue
            # Add BaseWidth
            ws[k] = w
        return cls(ws)

    @from_format.register(np.void)
    @from_format.register(np.ndarray)
    @classmethod
    def _from_format_structuredarray(cls, data: np.ndarray) -> Widths:
        if not is_structured(data):
            raise ValueError
        return cls.from_format.__wrapped__(cls, data)

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
    # Interoperability

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        arrs_g = ((str(k._physical_type_list[0]), np.array(v, dtype)) for k, v in self.items())
        return merge_arrays(tuple(v.view(np.dtype([(k, v.dtype)])) for k, v in arrs_g))

    def __array_function__(
        self, func: Callable[..., Any], types: tuple[type, ...], args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> Any:
        # LOCAL
        from .interop import WS_FUNCS

        if func not in WS_FUNCS:
            return NotImplemented

        finfo = WS_FUNCS[func](self)
        if not finfo.validate_types(types):
            return NotImplemented
        return finfo.func(*args, **kwargs)

    # ===============================================================
    # Magic Methods

    @__getitem__.register(np.ndarray)
    @__getitem__.register(np.integer)
    @__getitem__.register(bool)
    @__getitem__.register(slice)
    @__getitem__.register(int)
    def _getitem_valid(self, key: Any) -> Widths[W1]:
        return self.__class__({k: v[key] for k, v in self.items()})


##############################################################################


@TO_FORMAT.implements(np.ndarray, dispatch_on=Widths)
def _to_format_ndarray(data, *args):
    return np.array(data, *args)


@TO_FORMAT.implements(u.Quantity, dispatch_on=Widths)
def _to_format_quantity(data, *args):
    unit = u.StructuredUnit(tuple(v.units for v in data.values()))
    out = u.Quantity(np.array(data), unit=unit)
    return out
