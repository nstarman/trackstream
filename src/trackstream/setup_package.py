# -*- coding: utf-8 -*-
# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Set up package."""


##############################################################################
# IMPORTS

from __future__ import absolute_import

# STDLIB
import copyreg
from types import MappingProxyType
from typing import Tuple, Type

try:
    # THIRD PARTY
    import tqdm  # noqa: F401
except ImportError:
    HAS_TQDM = False
else:
    HAS_TQDM = True


__all__ = ["HAS_TQDM"]


def pickle_mappingproxytype(mpt: MappingProxyType) -> Tuple[Type[MappingProxyType], Tuple[dict]]:
    """:mod:`pickle` a `~types.MappingProxyType`.

    Parameters
    ----------
    mpt : `~types.MappingProxyType`
        Object to pickle.

    Returns
    -------
    type
    tuple[dict]
    """
    return type(mpt), (dict(mpt),)


copyreg.pickle(MappingProxyType, pickle_mappingproxytype)
