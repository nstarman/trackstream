# Licensed under a 3-clause BSD style license - see LICENSE.rst

"""Set up package."""

from __future__ import annotations

# STDLIB
import copyreg
from types import MappingProxyType

try:
    # THIRD PARTY
    import tqdm  # noqa: F401
except ImportError:
    HAS_TQDM = False
else:
    HAS_TQDM = True


__all__ = ["HAS_TQDM"]


def pickle_mappingproxytype(mpt: MappingProxyType) -> tuple[type[MappingProxyType], tuple[dict]]:
    """:mod:`pickle` a `~types.MappingProxyType`.

    .. warning::

        Unfortunately unpickled MappingProxyType do not point back to the
        original object.

    Parameters
    ----------
    mpt : `~types.MappingProxyType`
        Object to pickle.

    Returns
    -------
    type, tuple[dict]
    """
    return type(mpt), (dict(mpt),)


copyreg.pickle(MappingProxyType, pickle_mappingproxytype)
