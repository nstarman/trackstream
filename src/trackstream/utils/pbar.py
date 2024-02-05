"""Progress bar, modified from :mod:`~emcee`."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from trackstream.setup_package import HAS_TQDM

if HAS_TQDM:
    import tqdm

__all__ = ["get_progress_bar"]
__credits__ = ["emcee"]

if TYPE_CHECKING:
    from typing_extensions import Self


##############################################################################
# CODE
##############################################################################


class _NoOpPBar:
    """A non-operable progress bar for compatability."""

    def __enter__(self: Self, *_: Any, **__: Any) -> Self:
        return self

    def __exit__(self: Any, *_: object, **__: Any) -> None:
        pass

    def update(self: Any, _: int) -> None:
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

    if not HAS_TQDM:
        logging.warning("install the tqdm library to have progress bars.")
        return _NoOpPBar()

    return tqdm.tqdm(total=total)
