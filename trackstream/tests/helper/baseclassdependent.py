# -*- coding: utf-8 -*-

"""Baseclass for tests which rely on a class."""

__all__ = [
    "BaseClassDependentTests",
]


##############################################################################
# CODE
##############################################################################


class BaseClassDependentTests:
    """Base class for tests which rely on a class.

    Subclasses must specify a class in the class definition.

        >>> class SubClass(BaseClassDependentTests, klass=object):
        ...     '''A SubClass with class=object.'''
        ...     pass
        >>> SubClass.klass == object
        True

    """

    def __init_subclass__(cls, klass: object, **kwargs):
        """Initialize subclass.

        Parameters
        ----------
        klass : object
            The class that will be tested.
        **kwargs
            Arguments into subclass initialization.

        """
        super().__init_subclass__(**kwargs)

        cls.klass = klass

    # /def


# /class


##############################################################################
# END
