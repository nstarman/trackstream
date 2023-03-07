"""Utilities for working with units."""

from __future__ import annotations

from functools import reduce
from operator import add
from typing import TYPE_CHECKING, Any

import astropy.units as u
from astropy.units.quantity_helper.function_helpers import function_helper
import numpy.lib.recfunctions as rfn

__all__ = ["merge_units"]

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence


def _izip_units_flat(iterable: Iterable[u.StructuredUnit | u.Unit] | u.StructuredUnit) -> Iterator[tuple[str, u.Unit]]:
    """Return an iterator collapsing any nested unit structure."""
    # Make Structured unit.
    units = iterable if isinstance(iterable, u.StructuredUnit) else u.StructuredUnit(iterable)

    # Yield from structured unit.
    for k, v in units.items():
        if isinstance(v, u.StructuredUnit):
            yield from _izip_units_flat(v)
        else:
            yield (k, v)


def merge_units(*units: u.UnitBase | u.StructuredUnit | None) -> u.StructuredUnit:
    """Merge a sequence of units into a structured unit.

    Parameters
    ----------
    *units : `astropy.units.UnitBase` or `astropy.units.StructuredUnit`
        The units to merge into one structured unit.
        Units that are `None` are ignored.

    Returns
    -------
    `astropy.units.StructuredUnit`
        Structured unit of constituent unit.
    """
    # filter units, excluding None
    actual_units = tuple(unit for unit in units if isinstance(unit, u.UnitBase | u.StructuredUnit))
    flat = tuple(_izip_units_flat(u.StructuredUnit(actual_units)))
    return u.StructuredUnit(tuple(uu for _, uu in flat), names=tuple(n for n, _ in flat))


@function_helper(helps=rfn.merge_arrays)
def merge_arrays(
    seqarrays: u.Quantity | Sequence[u.Quantity],
    fill_value: float = -1,
    *,
    flatten: bool = False,
    usemask: bool = False,
    asrecarray: bool = False,
) -> tuple[tuple[Any, ...], dict[str, Any], u.StructuredUnit, None]:
    """Merge structured Quantities field by field.

    Parameters
    ----------
    seqarrays : sequence of Quantity
        Sequence of arrays.
    fill_value : {float}, optional
        Filling value used to pad missing data on the shorter arrays.
    flatten : {False, True}, optional
        Whether to collapse nested fields.
    usemask : {False, True}, optional
        Whether to return a masked array or not.

        .. warning::
            Not yet supported.
    asrecarray : {False, True}, optional
        Whether to return a recarray (MaskedRecords) or not.

        .. warning::
            Not yet supported.

    Returns
    -------
    Quantity
        Merged structured Quantity.

    Raises
    ------
    ValueError
        If ``asrecarray`` is `True`.
        If ``usemask`` is `True`.
    """
    if asrecarray:
        msg = "asrecarray=True is not supported."
        raise ValueError(msg)
    if usemask:
        msg = "usemask=True is not supported."
        raise ValueError(msg)

    # Do we have a single ndarray as input ?
    if isinstance(seqarrays, u.Quantity):
        arrays = (seqarrays.value,)
        units = (seqarrays.unit,)
    else:
        arrays = tuple(a.value for a in seqarrays)
        units = (q.unit for q in seqarrays)

    unit = merge_units(*units) if flatten else u.StructuredUnit(reduce(add, units))

    return (
        (arrays,),
        {"fill_value": fill_value, "flatten": flatten, "usemask": usemask, "asrecarray": asrecarray},
        unit,
        None,
    )
