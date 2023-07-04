"""Interpolated track width."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields, replace
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

import astropy.units as u
from astropy.units import Quantity
from interpolated_coordinates.utils import (  # noqa: N817
    InterpolatedUnivariateSplinewithUnits as IUSU,
)
import numpy as np
import numpy.lib.recfunctions as rfn

from trackstream.track.utils import is_structured
from trackstream.track.width.base import WidthBase
from trackstream.track.width.core import BaseWidth
from trackstream.track.width.plural import Widths

__all__ = ["InterpolatedWidth"]


if TYPE_CHECKING:
    from astropy.coordinates import BaseRepresentation


Self = TypeVar("Self", bound="InterpolatedWidth")
W1 = TypeVar("W1", bound="BaseWidth")


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class InterpolatedWidth(WidthBase, Generic[W1]):
    """Interpolated width.

    This is a wrapper around a `trackstream.track.width.BaseWidth`,
    adding an ``affine`` parameter.

    Parameters
    ----------
    width : `trackstream.track.width.BaseWidth` subclass instance
        The width to interpolate.
    affine : Quantity
        The affine parameter with which to interpolate the width.
    """

    width: W1
    """Uninterpolated width."""

    affine: Quantity
    """Interpolating parameter."""

    def __post_init__(self) -> None:
        # Munge the width and affine to the correct order, so that it works with
        # reversed data.
        affine = np.atleast_1d(self.affine)
        if affine[0] > affine[-1]:
            affine = affine[::-1]
            width = self.width[::-1]
        else:
            width = self.width

        affine = cast("Quantity", affine)

        # Create interpolations
        interps = {f.name: IUSU(affine, getattr(width, f.name)) for f in fields(width)}
        self._interps: dict[str, IUSU]
        object.__setattr__(self, "_interps", interps)

    # ===============================================================

    @property
    def uninterpolated(self) -> BaseWidth:
        """Uninterpolated width."""
        return self.width

    @property
    def units(self) -> u.StructuredUnit:
        """Structured unit of the uninterpolated fields' units."""
        return self.width.units

    def __call__(self, affine: Quantity | None = None) -> WidthBase:
        """Return the width of the track at affine points.

        Parameters
        ----------
        affine : |Quantity| or None, optional
            The affine interpolation parameter. If None (default), return
            width evaluated at all "tick" interpolation points.

        Returns
        -------
        |Quantity|
            Path width evaluated at ``affine``.
        """
        afn = self.affine if affine is None else Quantity(np.atleast_1d(affine), copy=False)

        ws = {k: v(afn) for k, v in self._interps.items()}
        return replace(self.width, **ws)

    def represent_as(self, width_type: type[W1], point: BaseRepresentation) -> InterpolatedWidth[W1]:
        """Transform the width to another representation type.

        Parameters
        ----------
        width_type : `trackstream.track.width.BaseWidth` subclass class
            The width type to which to transform this width.
        point : `astropy.coordinates.BaseRepresentation` instance
            The base of the vector from which this width is defined.

        Returns
        -------
        `trackstream.track.width.BaseWidth` subclass instance
            This width, transformed to type ``width_type``.
        """
        # LOCAL
        from trackstream.track.width.transforms import represent_as

        return represent_as(self, width_type, point)

    # ===============================================================
    # I/O

    @singledispatchmethod
    @classmethod
    def from_format(cls, _: object, __: Quantity | None) -> Any:  # https://github.com/python/mypy/issues/11727
        """Construct this width instance from data of given type."""
        msg = "not dispatched"
        raise NotImplementedError(msg)

    @from_format.register(WidthBase)
    @classmethod
    def _from_format_self(cls, data: WidthBase, affine: Quantity | None) -> InterpolatedWidth:
        if not isinstance(data, cls):
            msg = "not dispatched"
            raise NotImplementedError(msg)
        if affine is not None and not np.array_equal(data.affine, affine):
            raise ValueError

        return data

    @from_format.register(Mapping)
    @classmethod
    def _from_format_dict(cls, data: Mapping[str, Any], affine: Quantity | None) -> InterpolatedWidth:
        afn = data["affine"] if affine is None else affine
        return cls(data["width"], affine=afn)

    @from_format.register(BaseWidth)
    @classmethod
    def _from_format_plainwidth(cls, data: BaseWidth, affine: Quantity) -> InterpolatedWidth:
        return cls(data, affine=affine)

    @from_format.register(Widths)
    @classmethod
    def _from_format_widths(cls, data: Widths, affine: Quantity) -> InterpolatedWidths:
        ws = {k: cls.from_format(w, affine=affine) for k, w in data.items()}
        return InterpolatedWidths(ws, affine=affine)

    # ===============================================================
    # Dunder Methods

    def __getattr__(self, key: str) -> Any:
        if key in self._interps:
            return self._interps[key]
        return super().__getattr__(key)

    def __dir__(self) -> Iterable[str]:
        return sorted(list(super().__dir__()) + list(self._interps.keys()))

    def __setstate__(self, state: tuple) -> None:
        self.__dict__.update(state)

    @singledispatchmethod
    def __getitem__(self: Self, key: object) -> Self:
        return replace(self, **{f.name: getattr(self, f.name)[key] for f in fields(self)})

    @__getitem__.register(slice)
    def _getitem_slice(self: Self, key: slice) -> Self:
        i, f, s = key.start, key.stop, key.step
        isq = tuple(isinstance(x, u.Quantity) for x in (i, f, s))

        if not any(isq):
            return self.__getitem__.__wrapped__(self, key)

        if all(isq):
            unit = i.unit
            afn = np.arange(i.to_value(unit), f.to_value(unit), s.to_value(unit)) * unit

            return replace(self, width=self(afn), affine=afn)

        if all(isq[:2]) and s is None:
            msg = "TODO!"
            raise NotImplementedError(msg)

        msg = "invalid slice"
        raise ValueError(msg)

    # ===============================================================
    # Interoperability

    # def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:


#####################################################################


class InterpolatedWidths(Widths[InterpolatedWidth[W1]]):
    """Interpolated widths."""

    def __init__(self, widths: dict[u.PhysicalType, InterpolatedWidth[W1]], affine: Quantity) -> None:
        super().__init__(widths)
        self.affine: Quantity = affine

        # post-init verification
        for _k, v in self.items():
            if not np.array_equal(v.affine, self.affine):
                raise ValueError

    # ===============================================================

    @property
    def uninterpolated(self) -> Widths:
        """The uninterpolated widths."""
        return Widths({k: w.uninterpolated for k, w in self.items()})

    def represent_as(self, _: type[W1], __: BaseRepresentation) -> W1:
        """Transform the width to another representation type."""
        msg = "TODO!"
        raise NotImplementedError(msg)

    def __call__(self, affine: Quantity | None = None) -> Widths:
        """Evaluate the widths at the given affine parameter."""
        return Widths({k: iw(affine) for k, iw in self.items()})

    # ===============================================================
    # I/O

    @singledispatchmethod
    @classmethod
    def from_format(cls, data: object, affine: Quantity) -> InterpolatedWidths:
        """Construct this width instance from data of given type."""
        # LOCAL
        from trackstream.track.width.core import BASEWIDTH_KIND

        ws = {}
        for wcls, k in BASEWIDTH_KIND.items():
            try:
                w = InterpolatedWidth.from_format(wcls.from_format(data), affine=affine)
            except (NotImplementedError, ValueError):  # noqa: PERF203
                continue
            if k in ws and len(fields(ws[k])) > len(fields(w)):
                continue
            ws[k] = w
        return cls(ws, affine=affine)

    @from_format.register(Mapping)
    @classmethod
    def _from_format_mapping(cls, data: Mapping[str | u.PhysicalType, Any], affine: Quantity) -> InterpolatedWidths:
        return cls.from_format.__wrapped__(cls, data, affine=affine)

    @from_format.register(np.void)
    @from_format.register(np.ndarray)
    @classmethod
    def _from_format_structuredarray(cls, data: np.ndarray, affine: Quantity) -> InterpolatedWidths:
        if not is_structured(data):
            raise ValueError
        return cls.from_format.__wrapped__(cls, data, affine=affine)

    @from_format.register(Widths)
    @classmethod
    def _from_format_widths(cls, data: Widths, affine: Quantity) -> InterpolatedWidths:
        if isinstance(data, cls):  # more specialized
            if not np.array_equal(data.affine, affine):
                msg = "affine does not match data"
                raise ValueError(msg)

            return data

        return cls({k: InterpolatedWidth.from_format(v, affine=affine) for k, v in data.items()}, affine=affine)

    # ===============================================================
    # Magic Methods

    @singledispatchmethod
    def __getitem__(self, key: object) -> Any:  # https://github.com/python/mypy/issues/11727
        msg = f"not dispatched on {key}"
        raise NotImplementedError(msg)

    @__getitem__.register(u.PhysicalType)
    @__getitem__.register(str)
    def _getitem_key(self, key: str | u.PhysicalType) -> InterpolatedWidth:
        return self._mapping[self._get_key(key)]

    @__getitem__.register(np.ndarray)
    @__getitem__.register(np.integer)
    @__getitem__.register(bool)
    @__getitem__.register(slice)
    @__getitem__.register(int)
    def _getitem_valid(self, key: int | slice | bool | np.integer | np.ndarray) -> InterpolatedWidths[W1]:
        return self.__class__({k: v[key] for k, v in self.items()}, affine=cast("Quantity", self.affine[key]))

    # ===============================================================
    # Interoperability

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        arrs_g = ((k, np.array(v(self.affine), dtype)) for k, v in self.items())
        return rfn.merge_arrays(tuple(v.view(np.dtype([(k, v.dtype)])) for k, v in arrs_g))
