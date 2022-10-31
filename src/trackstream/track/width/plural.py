##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import copy as pycopy
from collections.abc import Mapping
from dataclasses import fields
from functools import singledispatchmethod
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Iterator,
    KeysView,
    MutableMapping,
    TypeVar,
    ValuesView,
    cast,
)

# THIRD PARTY
import astropy.units as u
import numpy as np
import numpy.lib.recfunctions as rfn
from overload_numpy import NPArrayOverloadMixin, NumPyOverloader
from override_toformat import ToFormatOverloadMixin

# LOCAL
from trackstream.track.utils import is_structured
from trackstream.track.width.base import FMT_OVERLOADS, WidthBase
from trackstream.track.width.core import BASEWIDTH_KIND, LENGTH

if TYPE_CHECKING:
    # THIRD PARTY
    from astropy.coordinates import BaseRepresentation
    from override_toformat import ToFormatOverloader

    # LOCAL
    from trackstream.track.width.interpolated import InterpolatedWidths

__all__ = ["Widths"]


##############################################################################
# TYPING

T = TypeVar("T")
W1 = TypeVar("W1", bound="WidthBase")
W2 = TypeVar("W2", bound="WidthBase")


##############################################################################
# PARAMETERS

WS_FUNCS = NumPyOverloader()


##############################################################################
# CODE
##############################################################################


class Widths(MutableMapping[u.PhysicalType, W1], NPArrayOverloadMixin, ToFormatOverloadMixin):
    # TODO! make work with Interpolated

    NP_OVERLOADS: ClassVar[NumPyOverloader] = WS_FUNCS
    FMT_OVERLOADS: ClassVar[ToFormatOverloader] = FMT_OVERLOADS

    def __init__(self, widths: dict[u.PhysicalType, W1]) -> None:
        # Validate that have the necessary base of the tower
        if LENGTH not in widths:
            raise ValueError("need positions")
        if any(not isinstance(k, u.PhysicalType) for k in widths.keys()):
            raise ValueError("all keys must be a PhysicalType")

        self._spaces: dict[u.PhysicalType, W1] = widths

    @singledispatchmethod
    def __getitem__(self, key: object) -> Any:  # https://github.com/python/mypy/issues/11727
        raise NotImplementedError("not dispatched")

    @singledispatchmethod
    def __setitem__(self, key: object, value: Any) -> Any:  # https://github.com/python/mypy/issues/11727
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
        k = key if isinstance(key, u.PhysicalType) else cast("u.PhysicalType", u.get_physical_type(key))
        return k

    @__getitem__.register(str)
    @__getitem__.register(u.PhysicalType)
    def _getitem_key(self, key: str | u.PhysicalType) -> W1:
        return self._spaces[self._get_key(key)]

    @__setitem__.register(str)
    @__setitem__.register(u.PhysicalType)
    def _setitem_str_or_PT(self, k: str | u.PhysicalType, v: W1) -> None:
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

    # ===============================================================
    # Interoperability

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        arrs_g = ((str(k._physical_type_list[0]), np.array(v, dtype)) for k, v in self.items())
        return rfn.merge_arrays(tuple(v.view(np.dtype([(k, v.dtype)])) for k, v in arrs_g))

    # ===============================================================
    # Magic Methods

    @__getitem__.register(np.ndarray)
    @__getitem__.register(np.integer)
    @__getitem__.register(bool)
    @__getitem__.register(slice)
    @__getitem__.register(int)
    def _getitem_valid(self, key: Any) -> Widths[W1]:
        return self.__class__({k: v[key] for k, v in self.items()})

    @singledispatchmethod
    def __lt__(self, other: object) -> Any:  # https://github.com/python/mypy/issues/11727
        return NotImplemented

    @__setitem__.register(Mapping)
    def _setitem_mapping(
        self, key: Mapping[u.PhysicalType, Mapping[str, Any]], value: Mapping[u.PhysicalType, W1]
    ) -> None:
        if key.keys() != self.keys():
            return NotImplemented
        elif key.keys() != value.keys():
            return NotImplemented

        # Delegate to contained Width
        for k in self.keys():
            self[k][key[k]] = value[k]

    def __deepcopy__(self, memo: dict[Any, Any]) -> Widths:
        return type(self)(pycopy.deepcopy(self._spaces, memo))


##############################################################################


@FMT_OVERLOADS.implements(to_format=np.ndarray, from_format=Widths)
def _to_format_ndarray(cls, data, *args):
    return np.array(data, *args)


@FMT_OVERLOADS.implements(to_format=u.Quantity, from_format=Widths)
def _to_format_quantity(cls, data, *args):
    unit = u.StructuredUnit(tuple(v.units for v in data.values()))
    out = cls(np.array(data), unit=unit)
    return out


@Widths.__lt__.register(Widths)
def _lt_widths(self, other: Widths) -> dict[u.PhysicalType, np.ndarray]:
    if any(k not in self for k in other.keys()):
        return NotImplemented

    return {k: self[k] < v for k, v in other.items()}
