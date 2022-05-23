# -*- coding: utf-8 -*-

"""Descriptors for :mod:`~trackstream`."""


__all__ = ["InstanceDescriptor", "TypedMetaAttribute"]


##############################################################################
# IMPORTS

# STDLIB
import sys
import weakref
from typing import Any, Generic, Optional, Type, TypeVar, Union, overload

# THIRD PARTY
from astropy.utils.metadata import MetaAttribute
from typing_extensions import Self

##############################################################################
# PARAMETERS

T = TypeVar("T")
EnclType = TypeVar("EnclType")


##############################################################################
# CODE
##############################################################################


class InstanceDescriptor(Generic[EnclType]):
    """

    Notes
    -----
    This is a non-data descriptor (see
    https://docs.python.org/3/howto/descriptor.html#descriptor-protocol). When
    ``__get__`` is first called it will make a copy of this descriptor instance
    and place it in the enclosing object's ``__dict__``. Thereafter attribute
    access will return the instance in ``__dict__`` without calling this
    descriptor.
    """

    _enclosing_attr: str
    """The enclosing instance's attribute name for this descriptor.

    Set in ``__set_name__``. Initialized as '' (a blank str) so that the
    attribute always exists, even if this class is not instantated as a
    descriptor and ``__set_name__`` is not called.
    """

    if sys.version_info >= (3, 9):
        _enclosing_ref: Optional[weakref.ReferenceType[EnclType]]
        """Reference to the enclosing instance."""
    else:
        _enclosing_ref: Optional[weakref.ReferenceType]
        """Reference to the enclosing instance."""

    def __init__(self) -> None:
        # Setting these attributes here so they always exist, even if the class
        # is not instantiated as a descriptor.
        self._enclosing_attr = ""  # set in __set_name__
        self._enclosing_ref = None  # set in __get__

    # ------------------------------------
    # Enclosing instance

    def _get_enclosing(self) -> Optional[EnclType]:
        """Get the enclosing instance or `None` if none is found.

        Returns
        -------
        None or object
        """
        if isinstance(self._enclosing_ref, weakref.ReferenceType):
            enclosing: Optional[EnclType] = self._enclosing_ref()
        else:
            enclosing = None
        return enclosing

    @property
    def _enclosing(self) -> EnclType:
        """Enclosing instance.

        Returns
        -------
        object
            Of type set by the generic protocol.

        Raises
        ------
        ValueError
            If no reference exists (or resolves to `None`).
        """
        enclosing = self._get_enclosing()

        if enclosing is None:
            raise ValueError("no reference exists to the original enclosing object")

        return enclosing

    # ------------------------------------
    # Descriptor properties

    def __set_name__(self, _: Type[EnclType], name: str) -> None:
        self._enclosing_attr = name

    def __get__(self: Self, obj: Optional[EnclType], _: Optional[Type[EnclType]]) -> Self:
        # accessed from a class
        if obj is None:
            return self

        # accessed from an obj
        descriptor: Optional[Self] = obj.__dict__.get(self._enclosing_attr)  # get from obj
        if descriptor is None:  # hasn't been created on the obj
            descriptor = self.__class__()
            descriptor._enclosing_attr = self._enclosing_attr
            obj.__dict__[self._enclosing_attr] = descriptor

        # We set `_enclosing_ref` on every call, since if one makes copies of objs,
        # 'descriptor' will be copied as well, which will lose the reference.
        descriptor._enclosing_ref = weakref.ref(obj)  # type: ignore

        return descriptor


class TypedMetaAttribute(MetaAttribute, Generic[T]):
    @overload
    def __get__(self: Self, obj: None, objtype: type) -> Self:
        ...

    @overload
    def __get__(self: Self, obj: Any, objtype: Optional[type]) -> T:
        ...

    def __get__(self: Self, obj: Optional[Any], objtype: Optional[type] = None) -> Union[Self, T]:
        return super().__get__(obj, objtype)
