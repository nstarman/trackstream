# -*- coding: utf-8 -*-

"""TrackStream Configurations."""


__all__ = [
    "conf",  # configuration instance
]


##############################################################################
# IMPORTS

# THIRD PARTY
from astropy import config as _config

##############################################################################
# CODE
##############################################################################


class Conf(_config.ConfigNamespace):
    """Configuration parameters for `trackstream`."""

    # lmfit

    use_lmfit = _config.ConfigItem(
        False,
        description="Use lmfit.",
        cfgtype="boolean(default=False)",
    )


conf = Conf()
# /class


# -------------------------------------------------------------------


##############################################################################
# END
