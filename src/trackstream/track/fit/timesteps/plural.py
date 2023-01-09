"""Timesteps."""

from __future__ import annotations

# STDLIB
from collections.abc import Mapping
from functools import singledispatchmethod
from typing import Any, ClassVar, TypeVar, cast, final

# THIRD PARTY
import astropy.units as u
import numpy as np
from numpy.lib.recfunctions import merge_arrays
from override_toformat import ToFormatOverloader, ToFormatOverloadMixin

# LOCAL
from trackstream.track.utils import PhysicalTypeKeyMutableMapping, is_structured

__all__: list[str] = []


##############################################################################
# TYPING

T = TypeVar("T")


##############################################################################
# PARAMETERS

FMT_OVERLOADS = ToFormatOverloader()

# Physical Types
LENGTH = u.get_physical_type("length")
ANGLE = u.get_physical_type("angle")
SPEED = u.get_physical_type("speed")
ANGULAR_SPEED = u.get_physical_type("angular speed")


##############################################################################
# CODE
##############################################################################


@final
class Times(PhysicalTypeKeyMutableMapping[u.Quantity], ToFormatOverloadMixin):
    """Times."""

    FMT_OVERLOADS: ClassVar[ToFormatOverloader] = FMT_OVERLOADS

    def __init__(self, ts: dict[u.PhysicalType, u.Quantity]) -> None:
        # Validate
        if any(not isinstance(k, u.PhysicalType) for k in ts):
            msg = "all keys must be a PhysicalType"
            raise ValueError(msg)

        # Init
        self._mapping: dict[u.PhysicalType, u.Quantity] = ts

    @singledispatchmethod
    def __getitem__(self, key: object) -> Any:  # noqa: ARG002
        # see https://github.com/python/mypy/issues/11727 for returning Any
        msg = "not dispatched"
        raise NotImplementedError(msg)

    # ===============================================================
    # Mapping

    @staticmethod
    def _get_key(key: str | u.PhysicalType) -> u.PhysicalType:
        return cast("u.PhysicalType", u.get_physical_type(key))

    @__getitem__.register(str)
    @__getitem__.register(u.PhysicalType)
    def _getitem_key(self, key: str | u.PhysicalType) -> u.Quantity:
        return self._mapping[self._get_key(key)]

    def __contains__(self, o: object) -> bool:
        return o in self._mapping

    # ===============================================================
    # I/O & Intereop

    @singledispatchmethod
    @classmethod
    def from_format(cls, data: object) -> Any:  # noqa: ARG003
        """Construct from data of a given kind."""
        # see https://github.com/python/mypy/issues/11727 for returning Any
        msg = "not dispatched"
        raise NotImplementedError(msg)

    @from_format.register(Mapping)
    @classmethod
    def _from_format_mapping(cls, data: Mapping[str | u.PhysicalType, u.Quantity]) -> Times:
        if not all(isinstance(k, (str, u.PhysicalType)) and isinstance(v, u.Quantity) for k, v in data.items()):
            msg = "data must be a Mapping[str | u.PhysicalType, u.Quantity]"
            raise ValueError(msg)

        times: dict[u.PhysicalType, u.Quantity] = {}
        for k, v in data.items():
            pt = cls._get_key(k)
            if pt in times:
                msg = f"key {k} repeats physical type {pt}"
                raise ValueError(msg)
            times[pt] = v
        return cls(times)

    @from_format.register(u.Quantity)
    @classmethod
    def _from_format_structuredquantity(cls, data: u.Quantity) -> Times:
        if not is_structured(data):
            msg = "Quantity must be structured"
            raise ValueError(msg)

        ns = list(map(str, data.dtype.names))
        times: dict[u.PhysicalType, u.Quantity] = {}
        for k in ns:
            pt = cls._get_key(k)
            if pt in times:
                msg = f"key {k} repeats physical type {pt}"
                raise ValueError(msg)
            times[pt] = cast("u.Quantity", data[k])
        return cls(times)

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


@FMT_OVERLOADS.implements(to_format=np.ndarray, from_format=Times)
def _to_format_ndarray(data: Times) -> np.ndarray:
    return np.array(data)


@FMT_OVERLOADS.implements(to_format=u.Quantity, from_format=Times)
def _to_format_quantity(cls, data: Times) -> u.Quantity:
    arrs = np.array(data)
    units = u.StructuredUnit(tuple(v.unit for v in data.values()))
    return cls(arrs, unit=units)
