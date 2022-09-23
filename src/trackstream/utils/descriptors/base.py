"""Bound classes."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import sys
import weakref
from typing import Any, Callable, Generic, Protocol, TypeVar

__all__ = ["BoundClass", "BoundClassRef"]

##############################################################################
# PARAMETERS

Self = TypeVar("Self")  # from typing_extensions import Self

BndTo = TypeVar("BndTo")


##############################################################################
# CODE
##############################################################################


if sys.version_info >= (3, 9):

    class ReferenceTypeShim(weakref.ReferenceType[BndTo]):
        pass

else:  # TODO! remove when py3.9+

    class ReferenceTypeShim(weakref.ReferenceType):
        def __class_getitem__(cls: type[ReferenceTypeShim], item: Any) -> type[ReferenceTypeShim]:
            return cls


class BoundClassRef(ReferenceTypeShim[BndTo]):
    """`weakref.ref` keeping a `BoundClass` connected to its referant.

    Attributes
    ----------
    _bound_ref : `weakref.ReferenceType`[`BoundClass`]
        A weak reference to a |BoundClass| instance. See Notes for details.

    Notes
    -----
    `weakref.ProxyType` autodetects and cleans up deletion of the referent.
    However, unlike a dereferenced `weakref.ReferenceType`, `~weakref.ProxyType`
    fails ``is`` and ``issubclass`` checks. To emulate the auto-cleanup of
    `weakref.ProxyType`, this class adds a custom finalizer to
    `~weakref.ReferenceType` that will clean the referent on the bound instance.
    It is therefore also necessary to store a weak reference (using the base
    `weakref.ref`) to the bound object in the attribute ``_bound_ref``.::

        bound object  --> BoundClassRef  --> referent
            ^------- ref <-----|
    """

    __slots__ = ("_bound_ref",)

    # `__new__` is needed for type hint tracing because the superclass defines `__new__` without `bound`.
    def __new__(
        cls: type[Self],
        ob: BndTo,
        callback: Callable[[weakref.ReferenceType[BndTo]], Any] | None = None,
        *,
        bound: BoundClass[BndTo],
    ) -> Self:
        ref: Self = super().__new__(cls, ob, callback)  # type: ignore
        return ref

    def __init__(
        self,
        ob: BndTo,
        callback: Callable[[weakref.ReferenceType[BndTo]], Any] | None = None,
        *,
        bound: BoundClass[BndTo],
    ) -> None:
        # Add a reference to the BoundClass object (it holds ``ob``)
        self._bound_ref = weakref.ref(bound)
        # Create a finalizer that will be called when the referant is deleted,
        # setting ``bound.__selfref__ = None``.
        weakref.finalize(ob, self._finalizer_callback)

    def _finalizer_callback(self) -> None:
        """Callback for finalizer that sets ``bound.__selfref__ = None``."""
        bound = self._bound_ref()
        if bound is not None:  # check that reference to bound is alive.
            # del bound.__self__
            type(bound).__self__.fdel(bound)


class BoundClass(Generic[BndTo]):
    """Base class for a class bound to an instance of another class.

    Attributes
    ----------
    __self__ : object
        The instance of a class to which this class is bound.

    Notes
    -----
    This class is modeled after bound methods, which gain a ``__self__``
    attribute when they are on an instance. As this is a base class, assigning
    ``self.__self__ = <X>`` is left to subclasses.

    Examples
    --------
    Methods on classes are unbound:

        >>> class Example:
        ...     def method(self):
        ...         pass
        >>> Example.method
        <function Example.method at ...>

    When the class is instantiated the method becomes bound:

        >>> ex = Example()
        >>> ex.method
        <bound method Example.method of <core.base.Example object at ...>>

    ``BoundClass`` allows this to be extended this so that a class can be
    ``bound`` to another class. Remember that ``BoundClass`` is a baseclass, so
    the specific implementation is determined by which subclass is used. As a
    quick example:

        >>> class Example2:
        ...     @property
        ...     def attribute(self):
        ...         bcb = BoundClass()
        ...         bcb.__self__ = self
        ...         return bcb
        >>> ex2 = Example2()
        >>> ex2.attribute
        <core.base.BoundClass object at ...>
        >>> ex2.attribute.__self__ is ex2
        True

    Behind the scenes ``BoundClass`` uses :mod:`weakref` to ensure that classes
    do not unexpectedly keep each other from being :external+python:ref:`garbage
    collected`. For details of this implementation, see
    `bound_class.core.base.BoundClassRef`.

        >>> attribute = ex2.attribute  # survives deletion
        >>> del ex2
        >>> try: attribute.__self__
        ... except ReferenceError: print("ex2 has been deleted")
        ex2 has been deleted
    """

    @property
    def __self__(self) -> BndTo:
        """Return object to which this one is bound.

        Returns
        -------
        object

        Raises
        ------
        `weakref.ReferenceError`
            If no referant was assigned, if it was deleted, or if it was
            de-refenced (e.g. by ``del self.__self__``).
        """
        try:
            selfref = self.__selfref__
        except AttributeError:
            raise ReferenceError("no weakly-referenced object")

        if isinstance(selfref, BoundClassRef):
            boundto = selfref()  # dereference
            if boundto is not None:
                return boundto

            raise ReferenceError("weakly-referenced object no longer exists")

    @__self__.setter
    def __self__(self, value: BndTo) -> None:
        # Set the reference.
        self.__selfref__: BoundClassRef[BndTo] | None
        object.__setattr__(self, "__selfref__", BoundClassRef(value, bound=self))
        # Note: we use ReferenceType over ProxyType b/c the latter fails ``is``
        # and ``issubclass`` checks. ProxyType autodetects and cleans up
        # deletion of the referent, which ReferenceType does not, so we need a
        # custom ReferenceType subclass to emulate this behavior.

    @__self__.deleter
    def __self__(self) -> None:
        # Romove reference without deleting the attribute.
        object.__setattr__(self, "__selfref__", None)


class BoundClassLike(Protocol[BndTo]):
    __selfref__: BoundClassRef[BndTo] | None
    __self__: BndTo
