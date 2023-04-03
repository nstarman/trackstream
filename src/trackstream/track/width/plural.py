"""Widths."""

from __future__ import annotations

from collections.abc import Mapping
import copy as pycopy
from dataclasses import fields
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar, cast

import astropy.units as u
import numpy as np
import numpy.lib.recfunctions as rfn
from overload_numpy import NPArrayOverloadMixin, NumPyOverloader
from override_toformat import ToFormatOverloadMixin

from trackstream.track.utils import (
    PhysicalTypeKeyMapping,
    PhysicalTypeKeyMutableMapping,
    is_structured,
)
from trackstream.track.width.base import FMT_OVERLOADS, WidthBase
from trackstream.track.width.core import BASEWIDTH_KIND, LENGTH

__all__ = ["Widths"]


if TYPE_CHECKING:
    from types import NotImplementedType

    from astropy.coordinates import BaseRepresentation
    from override_toformat import ToFormatOverloader

    from trackstream.track.width.interpolated import InterpolatedWidths


T = TypeVar("T")
W1 = TypeVar("W1", bound="WidthBase")
W2 = TypeVar("W2", bound="WidthBase")


##############################################################################
# PARAMETERS

WS_FUNCS = NumPyOverloader()


##############################################################################
# CODE
##############################################################################


class Widths(PhysicalTypeKeyMutableMapping[W1], NPArrayOverloadMixin, ToFormatOverloadMixin):
    """Widths."""

    # TODO! make work with Interpolated

    NP_OVERLOADS: ClassVar[NumPyOverloader] = WS_FUNCS
    FMT_OVERLOADS: ClassVar[ToFormatOverloader] = FMT_OVERLOADS

    def __init__(self, widths: dict[u.PhysicalType, W1]) -> None:
        # Validate that have the necessary base of the tower
        if LENGTH not in widths:
            msg = "need positions"
            raise ValueError(msg)
        if any(not isinstance(k, u.PhysicalType) for k in widths):
            msg = "all keys must be a PhysicalType"
            raise ValueError(msg)

        self._mapping: dict[u.PhysicalType, W1] = widths

    @singledispatchmethod
    def __getitem__(self, key: object) -> Any:
        # see https://github.com/python/mypy/issues/11727 for why return Any
        msg = "not dispatched"
        raise NotImplementedError(msg)

    @singledispatchmethod
    def __setitem__(self, key: object, value: Any) -> Any:
        # see https://github.com/python/mypy/issues/11727 for why return Any
        msg = "not dispatched"
        raise NotImplementedError(msg)

    # ===============================================================

    def represent_as(self, width_type: type[W1], point: BaseRepresentation) -> W1:
        """Represent as a new width type.

        Parameters
        ----------
        width_type : type[WidthBase]
            The width type to represent as.
        point : BaseRepresentation
            The point to represent at.

        Returns
        -------
        WidthBase
            The width of the new type, at the point.
        """
        # # LOCAL

        raise NotImplementedError("TODO!")  # noqa: EM101

    def interpolated(self, affine: u.Quantity) -> InterpolatedWidths:
        """Interpolate the widths.

        Parameters
        ----------
        affine : Quantity
            The affine parameter to interpolate at.

        Returns
        -------
        InterpolatedWidths
            The interpolated widths.
        """
        if len(self) < 2:
            msg = "cannot interpolate; too short."
            raise ValueError(msg)
        # LOCAL
        from trackstream.track.width.interpolated import InterpolatedWidths

        return InterpolatedWidths.from_format(self, affine=affine)

    # ===============================================================
    # Mapping

    @staticmethod
    def _get_key(key: str | u.PhysicalType) -> u.PhysicalType:
        return key if isinstance(key, u.PhysicalType) else cast("u.PhysicalType", u.get_physical_type(key))

    @__getitem__.register(str)
    @__getitem__.register(u.PhysicalType)
    def _getitem_key(self, key: str | u.PhysicalType) -> W1:
        return self._mapping[self._get_key(key)]

    @__setitem__.register(str)
    @__setitem__.register(u.PhysicalType)
    def _setitem_str_or_PT(self, k: str | u.PhysicalType, v: W1) -> None:
        self._mapping[self._get_key(k)] = v

    # ===============================================================
    # I/O

    @singledispatchmethod
    @classmethod
    def from_format(cls, data: object) -> Widths:
        """Create from an object.

        Parameters
        ----------
        data : object
            The object to create from.

        Returns
        -------
        Widths
        """
        # copy over from `data`
        ws: dict[u.PhysicalType, W1] = {}
        for wcls, k in BASEWIDTH_KIND.items():
            try:
                w = wcls.from_format(data)
            except (NotImplementedError, ValueError, KeyError):
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
            w = data.get(k, data.get(str(k._physical_type_list[0]), None))  # noqa: SLF001
            # Maybe it's a bunch of fields
            if w is None:
                try:
                    w = wcls.from_format(data)
                except (NotImplementedError, ValueError, KeyError):
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
        arrs_g = ((str(k._physical_type_list[0]), np.array(v, dtype)) for k, v in self.items())  # noqa: SLF001
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
    def __lt__(self, other: object) -> Any:
        # see https://github.com/python/mypy/issues/11727 for why returns Any
        return NotImplemented

    @__setitem__.register(Mapping)
    def _setitem_mapping(
        self,
        key: Mapping[u.PhysicalType, Mapping[str, Any]],
        value: Mapping[u.PhysicalType, W1],
    ) -> None | NotImplementedType:
        if key.keys() != self.keys() or key.keys() != value.keys():
            raise ValueError

        # Delegate to contained Width
        for k in self:
            self[k][key[k]] = value[k]

    def __deepcopy__(self, memo: dict[Any, Any]) -> Widths:
        return type(self)(pycopy.deepcopy(self._mapping, memo))


##############################################################################


@FMT_OVERLOADS.implements(to_format=np.ndarray, from_format=Widths)
def _to_format_ndarray(cls: type[Widths], data: Widths, *args: Any) -> np.ndarray:  # noqa: ARG001
    return np.array(data, *args)


@FMT_OVERLOADS.implements(to_format=u.Quantity, from_format=Widths)
def _to_format_quantity(cls: type[Widths], data: Widths, *args: Any) -> u.Quantity:  # noqa: ARG001
    unit = u.StructuredUnit(tuple(v.units for v in data.values()))
    return cls(np.array(data), unit=unit)


@Widths.__lt__.register(Widths)
def _lt_widths(self: Widths, other: Widths) -> PhysicalTypeKeyMapping[np.ndarray]:
    if not set(other.keys()).issubset(self.keys()):
        return NotImplemented

    return PhysicalTypeKeyMapping({k: self[k] < v for k, v in other.items()})
