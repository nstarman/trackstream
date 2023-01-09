"""Progress bar, modified from :mod:`~emcee`."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import logging
from typing import TYPE_CHECKING, Any

# LOCAL
from trackstream.setup_package import HAS_TQDM

if HAS_TQDM:
    # THIRD PARTY
    import tqdm

if TYPE_CHECKING:
    # THIRD PARTY
    from typing_extensions import Self


__all__ = ["get_progress_bar"]
__credits__ = ["emcee"]

##############################################################################
# CODE
##############################################################################


class _NoOpPBar:
    """A non-operable progress bar for compatability."""

    def __enter__(self: Self, *_: Any, **__: Any) -> Self:
        return self

    def __exit__(self, *_: Any, **__: Any) -> None:
        pass

    def update(self, _: int) -> None:
        pass


def get_progress_bar(*, display: bool, total: int) -> _NoOpPBar | tqdm.tqdm:
    """Get a progress bar.

    If :mod:`tqdm` is not installed, this will return a no-op.
    Function modified from :mod:`~emcee`.

    Parameters
    ----------
    display : boo
        Should the bar actually show the progress?
    total : int
        The total size of the progress bar.

    Returns
    -------
    `_NoOpPBar` or `tqdm.tqdm`
    """
    if display is False:
        return _NoOpPBar()
    elif not HAS_TQDM:
        logging.warning("install the tqdm library to have progress bars.")
        return _NoOpPBar()

    return tqdm.tqdm(total=total)
