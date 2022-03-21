# -*- coding: utf-8 -*-

"""Utilities for :mod:`~trackstream.utils`."""


__all__ = ["InstanceDescriptor"]


##############################################################################
# IMPORTS

# STDLIB
import weakref
from typing import Optional, Type, TypeVar, Union

##############################################################################
# PARAMETERS


ParentType = TypeVar("ParentType")


##############################################################################
# CODE
##############################################################################


class InstanceDescriptor:

    _parent_attr: Optional[str]
    _parent_cls: Optional[Type[ParentType]]
    _parent_ref: Optional[weakref.ref]

    def __init__(self) -> None:
        # references to parent class and instance
        self._parent_attr = None  # set in __set_name__
        self._parent_cls = None
        self._parent_ref = None

    @property
    def _parent(self) -> Union:
        """Parent instance."""
        return self._parent_ref() if self._parent_ref is not None else self._parent_cls

    # ------------------------------------

    def __set_name__(self, objcls: Type[ParentType], name: str):
        self._parent_attr = name

    def __get__(self, obj: Optional[ParentType], objcls: Optional[Type[ParentType]]):
        # accessed from a class
        if obj is None:
            self._parent_cls: Type[ParentType] = objcls
            return self

        # accessed from an obj
        descriptor = obj.__dict__.get(self._parent_attr)  # get from obj
        if descriptor is None:  # hasn't been created on the obj
            descriptor = self.__class__()
            descriptor._parent_cls = obj.__class__
            descriptor._parent_attr = self._parent_attr
            obj.__dict__[self._parent_attr] = descriptor

        # We set `_parent_ref` on every call, since if one makes copies of objs,
        # 'descriptor' will be copied as well, which will lose the reference.
        descriptor._parent_ref = weakref.ref(obj)
        return descriptor
