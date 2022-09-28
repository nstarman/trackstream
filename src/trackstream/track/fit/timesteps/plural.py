##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from collections.abc import Mapping
from functools import singledispatchmethod
from typing import (
    Any,
    ClassVar,
    Iterator,
    KeysView,
    MutableMapping,
    TypeVar,
    ValuesView,
    cast,
    final,
)

# THIRD PARTY
import astropy.units as u
import numpy as np
from numpy.lib.recfunctions import merge_arrays

# LOCAL
from trackstream.track.utils import is_structured
from trackstream.utils.to_format_overload import ToFormatOverloader

__all__: list[str] = []


##############################################################################
# TYPING

T = TypeVar("T")


##############################################################################
# PARAMETERS

TO_FORMAT = ToFormatOverloader()

# Physical Types
LENGTH = u.get_physical_type("length")
ANGLE = u.get_physical_type("angle")
SPEED = u.get_physical_type("speed")
ANGULAR_SPEED = u.get_physical_type("angular speed")


##############################################################################
# CODE
##############################################################################


@final
class Times(MutableMapping[u.PhysicalType, u.Quantity]):

    TO_FORMAT: ClassVar[ToFormatOverloader] = TO_FORMAT

    def __init__(self, ts: dict[u.PhysicalType, u.Quantity]) -> None:
        # Validate
        if any(not isinstance(k, u.PhysicalType) for k in ts.keys()):
            raise ValueError("all keys must be a PhysicalType")

        # Init
        self._ts: dict[u.PhysicalType, u.Quantity] = ts

    @singledispatchmethod
    def __getitem__(self, key: object) -> Any:  # https://github.com/python/mypy/issues/11727
        raise NotImplementedError("not dispatched")

    # ===============================================================
    # Mapping

    @staticmethod
    def _get_key(key: str | u.PhysicalType) -> u.PhysicalType:
        return cast("u.PhysicalType", u.get_physical_type(key))

    @__getitem__.register(str)
    @__getitem__.register(u.PhysicalType)
    def _getitem_key(self, key: str | u.PhysicalType) -> u.Quantity:
        return self._ts[self._get_key(key)]

    def __setitem__(self, k: str | u.PhysicalType, v: u.Quantity) -> None:
        self._ts[self._get_key(k)] = v

    def __delitem__(self, k: str | u.PhysicalType) -> None:
        del self._ts[self._get_key(k)]

    def __iter__(self) -> Iterator[u.PhysicalType]:
        return iter(self._ts)

    def __len__(self) -> int:
        return len(self._ts)

    def keys(self) -> KeysView[u.PhysicalType]:
        return self._ts.keys()

    def values(self) -> ValuesView[u.Quantity]:
        return self._ts.values()

    def __repr__(self) -> str:
        return f"Times({self._ts!r})"

    def __contains__(self, o: object) -> bool:
        return o in self._ts

    # ===============================================================
    # I/O & Intereop

    @singledispatchmethod
    @classmethod
    def from_format(cls, data: object) -> Any:  # https://github.com/python/mypy/issues/11727
        raise NotImplementedError("not dispatched")

    @from_format.register(Mapping)
    @classmethod
    def _from_format_mapping(cls, data: Mapping[str | u.PhysicalType, u.Quantity]) -> Times:
        if not all(isinstance(k, (str, u.PhysicalType)) and isinstance(v, u.Quantity) for k, v in data.items()):
            raise ValueError("data must be a Mapping[str | u.PhysicalType, u.Quantity]")

        times: dict[u.PhysicalType, u.Quantity] = {}
        for k, v in data.items():
            pt = cls._get_key(k)
            if pt in times:
                raise ValueError(f"key {k} repeats physical type {pt}")
            times[pt] = v
        return cls(times)

    @from_format.register(u.Quantity)
    @classmethod
    def _from_format_structuredquantity(cls, data: u.Quantity) -> Times:
        if not is_structured(data):
            raise ValueError("Quantity must be structured")

        ns = list(map(str, data.dtype.names))
        times: dict[u.PhysicalType, u.Quantity] = {}
        for k in ns:
            pt = cls._get_key(k)
            if pt in times:
                raise ValueError(f"key {k} repeats physical type {pt}")
            times[pt] = cast("u.Quantity", data[k])
        return cls(times)

    def to_format(self, format: type[T], /, *args: Any, **kwargs: Any) -> T:
        """Transform times to specified format.

        Parameters
        ----------
        format : type, positional-only
            The format type to which to transform this Times.
        *args : Any
            Arguments into ``to_format``.
        **kwargs : Any
            Keyword-arguments into ``to_format``.

        Returns
        -------
        object
            Times transformed to specified type.

        Raises
        ------
        ValueError
            If format is not one of the recognized types.
        """
        if format not in self.TO_FORMAT:
            raise ValueError(f"format {format} is not known -- {self.TO_FORMAT.keys()}")

        out = self.TO_FORMAT[format](self).func(self, *args, **kwargs)
        return out

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        arrs_g = ((k._physical_type_list[0], np.array(v, dtype)) for k, v in self.items())
        return merge_arrays(tuple(v.view(np.dtype([(k, v.dtype)])) for k, v in arrs_g))

    # ===============================================================
    # Magic Methods

    @__getitem__.register(np.ndarray)
    @__getitem__.register(np.integer)
    @__getitem__.register(bool)
    @__getitem__.register(slice)
    @__getitem__.register(int)
    def _getitem_valid(self, key: Any) -> Times:
        return self.__class__({k: u.Quantity(v[key]) for k, v in self.items()})


##############################################################################


@TO_FORMAT.implements(np.ndarray, dispatch_on=Times)
def _to_format_ndarray(data: Times) -> np.ndarray:
    return np.array(data)


@TO_FORMAT.implements(u.Quantity, dispatch_on=Times)
def _to_format_quantity(data: Times) -> u.Quantity:
    arrs = np.array(data)
    units = u.StructuredUnit(tuple(v.unit for v in data.values()))
    out = u.Quantity(arrs, unit=units)
    return out