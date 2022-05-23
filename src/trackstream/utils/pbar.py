# -*- coding: utf-8 -*-

"""Progress bar, modified from :mod:`~emcee`."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import logging
from typing import Any, Union

# LOCAL
from trackstream.setup_package import HAS_TQDM

if HAS_TQDM:
    # THIRD PARTY
    import tqdm


__all__ = ["get_progress_bar"]
__credits__ = ["emcee"]

##############################################################################
# CODE
##############################################################################


class _NoOpPBar:
    """This class implements the progress bar interface but does nothing."""

    def __enter__(self, *_: Any, **__: Any) -> _NoOpPBar:
        return self

    def __exit__(self, *_: Any, **__: Any) -> None:
        pass

    def update(self, _: int) -> None:
        pass


def get_progress_bar(display: Union[bool, str], total: int) -> Union[_NoOpPBar, tqdm.tqdm]:
    """Get a progress bar.

    If :mod:`tqdm` is not installed, this will return a no-op.
    Function modified from :mod:`~emcee`.

    Parameters
    ----------
    display : bool or str
        Should the bar actually show the progress?
        Or a string to indicate which tqdm bar to use.
    total : int
        The total size of the progress bar.

    Returns
    -------
    `_NoOpPBar` or `tqdm.tqdm`
    """
    if display is False:
        return _NoOpPBar()
    elif not HAS_TQDM:
        logging.warning("You must install the tqdm library to have progress bars.")
        return _NoOpPBar()
    elif display is True:
        return tqdm.tqdm(total=total)
    else:
        return getattr(tqdm, "tqdm_" + str(display))(total=total)
