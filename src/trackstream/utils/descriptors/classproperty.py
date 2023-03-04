"""Class-level property."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

if TYPE_CHECKING:
    from collections.abc import Callable

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

    fget: Callable[..., T] | None

    def __init__(self: Any, fget: Callable[..., T] | None = None, doc: str | None = None) -> None:
        self.fget: Callable[..., T] | None = fget
        self.__doc__: str | None = fget.__doc__ if (doc is None and fget is not None) else doc
        self._name: str = ""  # in case ``__set_name__`` is not called.

    def __set_name__(self: Any, _: type, name: str) -> None:
        self._name = name

    def __get__(self: Any, obj: object | None, objtype: None | type = None) -> T:
        if self.fget is None:
            msg = f"unreadable attribute {self._name}"
            raise AttributeError(msg)
        return cast("T", self.fget(type(obj) if objtype is None else objtype))

    def getter(self: Any, fget: Callable[..., T]) -> classproperty[T]:
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
        prop: classproperty[T] = type(self)(fget, self.__doc__)
        prop._name = self._name  # noqa: SLF001
        return prop
