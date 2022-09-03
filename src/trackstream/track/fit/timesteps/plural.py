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
    NoReturn,
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


##############################################################################
# CODE
##############################################################################


@final
class Times(MutableMapping[str, u.Quantity]):

    TO_FORMAT: ClassVar[ToFormatOverloader] = TO_FORMAT

    def __init__(self, ts: dict[str, u.Quantity]) -> None:
        self._ts: dict[str, u.Quantity] = ts

        # notphysical = [k for k in ts.keys() if k not in _name_physical_mapping]
        # if any(notphysical):
        #     raise ValueError(f"keys {notphysical} are not known physical types")

    # ===============================================================
    # Mapping

    def __getitem__(self, key: str) -> u.Quantity:
        return self._ts[key]

    def __setitem__(self, k: str, v: u.Quantity) -> None:
        self._ts[k] = v

    def __delitem__(self, k: str) -> None:
        del self._ts[k]

    def __iter__(self) -> Iterator[str]:
        return iter(self._ts)

    def __len__(self) -> int:
        return len(self._ts)

    def keys(self) -> KeysView[str]:
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
    def from_format(cls, data: object) -> NoReturn:
        raise NotImplementedError("not dispatched")

    @from_format.register(Mapping)
    @classmethod
    def _from_format_mapping(cls, data: Mapping[str, u.Quantity]) -> Times:
        if not all(isinstance(k, str) and isinstance(v, u.Quantity) for k, v in data.items()):
            raise ValueError

        return cls(dict(data))

    @from_format.register(u.Quantity)
    @classmethod
    def _from_format_structuredquantity(cls, data: u.Quantity) -> Times:
        if not is_structured(data):
            raise ValueError

        ns = list(map(str, data.dtype.names))
        return cls({k: cast(u.Quantity, data[k]) for k in ns})

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
        arrs_g = ((k, np.array(v, dtype)) for k, v in self.items())
        return merge_arrays(tuple(v.view(np.dtype([(k, v.dtype)])) for k, v in arrs_g))


##############################################################################


@TO_FORMAT.implements(np.ndarray, dispatch_on=Times)
def _to_format_ndarray(data: Times) -> np.ndarray:
    return np.array(data)


@TO_FORMAT.implements(u.Quantity, dispatch_on=Times)
def _to_format_quantity(data: Times) -> u.Quantity:
    arrs = np.array(data)
    units = u.StructuredUnit(tuple(v.steps.unit for v in data.values()))
    out = u.Quantity(arrs, unit=units)
    return out
