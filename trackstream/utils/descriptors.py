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


ParentType = TypeVar("ParentType")
IDT = TypeVar("IDT", bound="InstanceDescriptor")


##############################################################################
# CODE
##############################################################################


class InstanceDescriptor(Generic[ParentType]):

    _parent_attr: str
    _parent_cls: Type[ParentType]
    _parent_ref: weakref.ReferenceType

    @property
    def _parent(self) -> ParentType:
        """Parent instance."""
        if isinstance(getattr(self, "_parent_ref", None), weakref.ReferenceType):
            parent: Optional[ParentType] = self._parent_ref()
        else:
            parent = None

        if parent is None:
            raise ValueError("no reference exists to the original parent object")

        return parent

    # ------------------------------------

    def __set_name__(self, objcls: Type[ParentType], name: str) -> None:
        self._parent_attr = name

    def __get__(self: IDT, obj: Optional[ParentType], objcls: Optional[Type[ParentType]]) -> IDT:
        # accessed from a class
        if obj is None:
            return self

        # accessed from an obj
        descriptor: Optional[IDT] = obj.__dict__.get(self._parent_attr)  # get from obj
        if descriptor is None:  # hasn't been created on the obj
            descriptor = self.__class__()
            descriptor._parent_attr = self._parent_attr
            obj.__dict__[self._parent_attr] = descriptor

        # We set `_parent_ref` on every call, since if one makes copies of objs,
        # 'descriptor' will be copied as well, which will lose the reference.
        descriptor._parent_ref = weakref.ref(obj)  # type: ignore

        return descriptor
