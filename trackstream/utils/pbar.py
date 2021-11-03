# -*- coding: utf-8 -*-

"""Progress bar, modified from :mod:`~emcee`."""

# STDLIB
import logging

__all__ = ["get_progress_bar"]
__credits__ = ["emcee"]

##############################################################################
# IMPORTS

# LOCAL
from trackstream.setup_package import HAS_TQDM

if HAS_TQDM:
    # THIRD PARTY
    import tqdm

##############################################################################
# CODE
##############################################################################


class _NoOpPBar(object):
    """This class implements the progress bar interface but does nothing."""

    def __init__(self):
        pass

    def __enter__(self, *args, **kwargs):
        return self

    def __exit__(self, *args, **kwargs):
        pass

    def update(self, count):
        pass


def get_progress_bar(display, total):
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

    """
    if not display:
        return _NoOpPBar()
    elif not HAS_TQDM:
        logging.warning("You must install the tqdm library to have progress bars.")
        return _NoOpPBar()
    elif display:
        return tqdm.tqdm(total=total)
    else:
        return getattr(tqdm, "tqdm_" + display)(total=total)
