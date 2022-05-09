# -*- coding: utf-8 -*-

"""Utilities for :mod:`~trackstream.utils`."""


__all__ = ["InstanceDescriptor"]


##############################################################################
# IMPORTS

# STDLIB
import weakref
from typing import Generic, Optional, Type, TypeVar

##############################################################################
# PARAMETERS

EnclType = TypeVar("EnclType")
IDT = TypeVar("IDT", bound="InstanceDescriptor")


##############################################################################
# CODE
##############################################################################


class InstanceDescriptor(Generic[EnclType]):

    _enclosing_attr: str
    _enclosing_ref: weakref.ReferenceType[EnclType]

    @property
    def _enclosing(self) -> EnclType:
        """Enclosing instance."""
        if isinstance(getattr(self, "_enclosing_ref", None), weakref.ReferenceType):
            enclosing: Optional[EnclType] = self._enclosing_ref()
        else:
            enclosing = None

        if enclosing is None:
            raise ValueError("no reference exists to the original enclosing object")

        return enclosing

    # ------------------------------------

    def __set_name__(self, _: Type[EnclType], name: str) -> None:
        self._enclosing_attr = name

    def __get__(self: IDT, obj: Optional[EnclType], _: Optional[Type[EnclType]]) -> IDT:
        # accessed from a class
        if obj is None:
            return self

        # accessed from an obj
        descriptor: Optional[IDT] = obj.__dict__.get(self._enclosing_attr)  # get from obj
        if descriptor is None:  # hasn't been created on the obj
            descriptor = self.__class__()
            descriptor._enclosing_attr = self._enclosing_attr
            obj.__dict__[self._enclosing_attr] = descriptor

        # We set `_enclosing_ref` on every call, since if one makes copies of objs,
        # 'descriptor' will be copied as well, which will lose the reference.
        descriptor._enclosing_ref = weakref.ref(obj)  # type: ignore

        return descriptor
