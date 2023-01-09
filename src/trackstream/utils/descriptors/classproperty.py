"""Class-level property."""

from __future__ import annotations

# STDLIB
from typing import Callable, Generic, TypeVar

##############################################################################
# TYPING

T = TypeVar("T")


##############################################################################
# CODE
##############################################################################


class classproperty(Generic[T]):
    """Class-level property.

    Parameters
    ----------
    fget : callable[..., T] or None, optional
        Getter for the class property.
    doc : str or None, optional
        Docstring of class property. If None, tries to get from ``fget``.
    """

    def __init__(self, fget: Callable[..., T] | None = None, doc: str | None = None) -> None:
        thedoc = fget.__doc__ if (doc is None and fget is not None) else doc

        self.fget = fget
        self.__doc__ = thedoc
        self._name = ""  # in case ``__set_name__`` is not called.

    def __set_name__(self, _: type, name: str) -> None:
        self._name = name

    def __get__(self, obj: object | None, objtype: None | type = None) -> T:
        if self.fget is None:
            msg = f"unreadable attribute {self._name}"
            raise AttributeError(msg)
        return self.fget(type(obj) if objtype is None else objtype)

    def getter(self, fget: Callable[..., T]) -> classproperty[T]:
        """Descriptor to obtain a copy of the property with a different getter.

        Parameters
        ----------
        fget : Callable[..., T]
            New getter.

        Returns
        -------
        classproperty[T]
            With new getter set.
        """
        prop = type(self)(fget, self.__doc__)
        prop._name = self._name
        return prop
