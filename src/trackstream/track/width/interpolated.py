##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Generic, Iterable, TypeVar, cast

# THIRD PARTY
import astropy.units as u
import numpy as np
import numpy.lib.recfunctions as rfn
from astropy.units import Quantity
from interpolated_coordinates.utils import InterpolatedUnivariateSplinewithUnits as IUSU

# LOCAL
from trackstream.track.utils import is_structured
from trackstream.track.width.base import WidthBase
from trackstream.track.width.core import BaseWidth
from trackstream.track.width.plural import Widths

if TYPE_CHECKING:
    # THIRD PARTY
    from astropy.coordinates import BaseRepresentation

__all__ = ["InterpolatedWidth"]


##############################################################################
# TYPING

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
        width = replace(self.width, **ws)

        return width

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
    def from_format(cls, data: object, affine: Quantity | None) -> Any:  # https://github.com/python/mypy/issues/11727
        """Construct this width instance from data of given type."""
        raise NotImplementedError("not dispatched")

    @from_format.register(WidthBase)
    @classmethod
    def _from_format_self(cls, data: WidthBase, affine: Quantity | None) -> InterpolatedWidth:
        if not isinstance(data, cls):
            raise NotImplementedError("not dispatched")
        elif affine is not None and not np.array_equal(data.affine, affine):
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
        return super().__getattr__(key)  # type: ignore

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

        elif all(isq[:2]) and s is None:
            # ii = np.abs(self.affine - i).argmin()
            # ff = np.abs(self.affine - f).argmin()

            raise NotImplementedError("TODO!")

        else:
            raise ValueError("invalid slice")

    # ===============================================================
    # Interoperability

    # def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
    #     dt = np.dtype([("affine", self.affine.dtype)] + [(f.name, dtype) for f in fields(self.width)])
    #     x = np.c_[(self.affine.value,) + tuple(getattr(self.width, f.name).value for f in fields(self.width))]
    #     return rfn.unstructured_to_structured(x, dtype=dt)


#####################################################################


class InterpolatedWidths(Widths[InterpolatedWidth[W1]]):
    def __init__(self, widths: dict[u.PhysicalType, InterpolatedWidth[W1]], affine: Quantity) -> None:
        super().__init__(widths)
        self.affine: Quantity = affine

        # post-init verification
        for k, v in self.items():
            if not np.array_equal(v.affine, self.affine):
                raise ValueError

    # ===============================================================

    @property
    def uninterpolated(self) -> Widths:
        return Widths({k: w.uninterpolated for k, w in self.items()})

    def represent_as(self, width_type: type[W1], point: BaseRepresentation) -> W1:
        # # LOCAL
        # from trackstream.track.width.transforms import represent_as

        # return represent_as(self, width_type, point)
        raise NotImplementedError("TODO!")

    def __call__(self, affine: Quantity | None = None) -> Widths:
        return Widths({k: iw(affine) for k, iw in self.items()})

    # ===============================================================
    # I/O

    @singledispatchmethod
    @classmethod
    def from_format(cls, data: object, affine: Quantity) -> InterpolatedWidths:
        # LOCAL
        from trackstream.track.width.core import BASEWIDTH_KIND

        ws = {}
        for wcls, k in BASEWIDTH_KIND.items():
            try:
                w = InterpolatedWidth.from_format(wcls.from_format(data), affine=affine)
            except Exception:
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
                raise ValueError("affine does not match data")

            return data

        return cls({k: InterpolatedWidth.from_format(v, affine=affine) for k, v in data.items()}, affine=affine)

    # ===============================================================
    # Magic Methods

    @singledispatchmethod
    def __getitem__(self, key: object) -> Any:  # https://github.com/python/mypy/issues/11727
        raise NotImplementedError(f"not dispatched on {key}")

    @__getitem__.register(u.PhysicalType)
    @__getitem__.register(str)
    def _getitem_key(self, key: str | u.PhysicalType) -> InterpolatedWidth:
        return self._spaces[self._get_key(key)]

    @__getitem__.register(np.ndarray)
    @__getitem__.register(np.integer)
    @__getitem__.register(bool)
    @__getitem__.register(slice)
    @__getitem__.register(int)
    def _getitem_valid(self, key: Any) -> InterpolatedWidths[W1]:
        return self.__class__({k: v[key] for k, v in self.items()}, affine=cast("Quantity", self.affine[key]))

    # ===============================================================
    # Interoperability

    def __array__(self, dtype: np.dtype | None = None) -> np.ndarray:
        arrs_g = ((k, np.array(v(self.affine), dtype)) for k, v in self.items())
        return rfn.merge_arrays(tuple(v.view(np.dtype([(k, v.dtype)])) for k, v in arrs_g))
