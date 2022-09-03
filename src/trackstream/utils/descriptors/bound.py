"""Descriptors on the instance, not the class."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
from dataclasses import dataclass, replace
from typing import (
    TYPE_CHECKING,
    Any,
    Literal,
    MutableMapping,
    NoReturn,
    TypeVar,
    overload,
)

# LOCAL
from .base import BndTo, BoundClass, BoundClassRef

if TYPE_CHECKING:
    # THIRD PARTY
    from typing_extensions import TypeAlias


__all__ = ["BoundDescriptor"]


##############################################################################
# PARAMETERS

Self = TypeVar("Self")  # mypy not yet compatible with Self

CacheLoc: TypeAlias = Literal["__dict__", "__cache__", None]


##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class BoundDescriptorBase(BoundClass[BndTo]):
    """Base class for instance-level descriptors.

    Attributes
    ----------
    enclosing : BndTo
        Returns the enclosing instance to which this one is bound.

    Notes
    -----
    Normally descriptors are bound to a class and are used on instances of that
    class according to the ``__get__`` method. While very useful for, e.g.
    performing validation, this essentially makes descriptors just fancy
    methods. Using |BoundClass| descriptors can now be easily bound to class
    instances, not the class.

    There are currently some limitations:

    1. The class must have a ``__dict__`` attribute. This doesn't preclude
       slots, but few slotted classes also have a ``__dict__``.
    2. The class must have a ``__name__`` attribute. Pretty much all classes do,
       so don't worry about this one.

    This is a base class and mostly exists because MyPy complains that
    :class:`bound_class.descriptors.BoundDescriptor` and
    :class:`bound_class.descriptors.BoundDescriptor` do not have matching
    signatures for ``__get__``.
    """

    store_in: Literal["__dict__", "_attrs_"] | None = "__dict__"

    def __post_init__(self) -> None:
        self.__selfref__: BoundClassRef[BndTo] | None
        object.__setattr__(self, "__selfref__", None)

    # ===============================================================
    # Descriptor

    def __set_name__(self, _: Any, name: str) -> None:
        # Store the name of the attribute on the enclosing object
        self._enclosing_attr: str
        object.__setattr__(self, "_enclosing_attr", name)

    # @abstractmethod
    # def __get__(
    #     self,
    #     enclosing: BndTo | None,
    #     enclosing_type: None | type[BndTo],
    # ):
    #     ...

    def __set__(self, _: str, __: Any) -> NoReturn:
        raise AttributeError  # TODO! useful error message

    # ===============================================================

    @property
    def enclosing(self) -> BndTo:
        """Return the enclosing instance to which this one is bound.

        Each access of this properety dereferences a `weakref.RefernceType`, so
        it is sometimes better to assign this property to a variable and work
        with that.

        .. code-block:: python

            obj = accessor.accessee obj...
        """
        return self.__self__


##############################################################################


@dataclass(frozen=True)
class BoundDescriptor(BoundDescriptorBase[BndTo]):
    """Descriptor stored on and accessess its enclosing instance.

    When attached as a descriptor this class will return itself if accesssed
    from the class and the instance-bound descriptor if accessed from an
    instance. Working with the class-bound descriptor is dangerous as
    modifications made there will not propagate to existing instances of the
    class. For this reason it is advised to use
    :class:`bound_class.descriptors.BoundDescriptor`.

    Examples
    --------
    Descriptors are constructed within an enclosing class.

        >>> class ExampleBoundDescriptor(BoundDescriptor):
        ...     def print_info(self):
        ...         print(f"this is attached to {self.enclosing.name!r}")

        >>> class Example:
        ...     attribute = ExampleBoundDescriptor()
        ...     def __init__(self, name):
        ...         self.name = name

    Through various dunder methods (see
    https://docs.python.org/3/howto/descriptor.html) descriptors know about the
    class on which they are defined

        >>> Example.attribute
        ExampleBoundDescriptor(store_in="__dict__")

    Descriptors also work on instances.

        >>> ex = Example("ex_instance")
        >>> ex.attribute
        ExampleBoundDescriptor(store_in='__dict__')

    What's special about |BoundClass|-derived descriptors is that their
    instances are bound to the instance of the enclosing class, not the class
    itself.

        >>> ex.attribute is Example.attribute
        False

    Consquently :class:`bound_class.descriptors.BoundDescriptor` can access the
    enclosing instance.

        >>> ex.attribute.enclosing is ex
        True

        >>> ex.attribute.print_info()
        this is attached to 'ex_instance'

    Notes
    -----
    Normally descriptors are bound to a class and are used on instances of that
    class according to the ``__get__`` method. While very useful for, e.g.
    performing validation, this essentially makes descriptors just fancy
    methods. Using |BoundClass| descriptors can now be easily bound to class
    instances, not the class.

    There are currently some limitations:

    1. The class must have a ``__dict__`` attribute. This doesn't preclude
       slots, but few slotted classes also have a ``__dict__``.
    2. The class must have a ``__name__`` attribute. Pretty much all classes do,
       so don't worry about this one.

    This is a data descriptor (see
    https://docs.python.org/3/howto/descriptor.html#descriptor-protocol). When
    ``__get__`` is first called it will make a copy of this descriptor instance
    and place it in the enclosing object's ``__dict__``. Thereafter attribute
    access will return the instance in ``__dict__``, first passing through this
    descriptor to make sure references are kept up-to-date.

    See Also
    --------
    bound_class.descriptors.BoundDescriptor
        A version of this descriptor that only permits access from the instance.
    """

    @overload
    def __get__(self: BoundDescriptor[BndTo], enclosing: BndTo, _: None) -> BoundDescriptor[BndTo]:
        ...

    @overload
    def __get__(self: BoundDescriptor[BndTo], enclosing: None, _: type[BndTo]) -> BoundDescriptor[BndTo]:
        ...

    def __get__(self: BoundDescriptor[BndTo], enclosing: BndTo | None, _: type[BndTo] | None) -> BoundDescriptor[BndTo]:
        # When called without an instance, return self to allow access
        # to descriptor attributes.
        if enclosing is None:
            return self

        # accessed from an enclosing
        if self.store_in is None:
            dsc = replace(self)
        else:  # try to get from cache
            cache: MutableMapping[str, Any] = getattr(enclosing, self.store_in)
            obj = cache.get(self._enclosing_attr)  # get from enclosing.

            if obj is None:  # hasn't been created on the enclosing
                dsc = replace(self)
                # transfer any other information
                dsc.__set_name__(dsc, self._enclosing_attr)
                # store on enclosing instance
                cache[self._enclosing_attr] = dsc
            elif not isinstance(obj, type(self)):
                raise TypeError(f"descriptor must be type <{type(self)}> not <{type(obj)}>")
            else:
                dsc = obj

        # We set `__self__` on every call, since if one makes copies of objs,
        # 'dsc' will be copied as well, which will lose the reference.
        # dsc.__self__ = enclosing
        type(dsc).__self__.fset(dsc, enclosing)

        return dsc
