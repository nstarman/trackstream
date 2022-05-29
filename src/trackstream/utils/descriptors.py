# -*- coding: utf-8 -*-

"""Descriptors for :mod:`~trackstream`."""


__all__ = ["InstanceDescriptor"]


##############################################################################
# IMPORTS

# STDLIB
import weakref
from typing import Generic, Optional, Type, TypeVar

# THIRD PARTY
from attrs import define, field
from typing_extensions import Self

##############################################################################
# PARAMETERS

T = TypeVar("T")
EnclType = TypeVar("EnclType")


##############################################################################
# CODE
##############################################################################


@define(frozen=True, repr=False, slots=False)
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

    _enclosing_attr: str = field(init=False, default="", repr=False)
    """The enclosing instance's attribute name for this descriptor.

    Set in ``__set_name__``. Initialized as '' (a blank str) so that the
    attribute always exists, even if this class is not instantated as a
    descriptor and ``__set_name__`` is not called.
    """

    _enclosing_ref: Optional[weakref.ReferenceType] = field(init=False, default=None, repr=False)
    """Reference to the enclosing instance."""

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
        object.__setattr__(self, "_enclosing_attr", name)

    def __get__(
        self: Self, obj: Optional[EnclType], _: Optional[Type[EnclType]], *args, **kwargs
    ) -> Self:
        # accessed from a class
        if obj is None:
            return self

        # accessed from an obj
        descriptor: Optional[Self] = obj.__dict__.get(self._enclosing_attr)  # get from obj
        if descriptor is None:  # hasn't been created on the obj
            descriptor = self.__class__(*args, **kwargs)
            object.__setattr__(descriptor, "_enclosing_attr", self._enclosing_attr)
            obj.__dict__[self._enclosing_attr] = descriptor

        # We set `_enclosing_ref` on every call, since if one makes copies of objs,
        # 'descriptor' will be copied as well, which will lose the reference.
        object.__setattr__(descriptor, "_enclosing_ref", weakref.ref(obj))  # type: ignore

        return descriptor
