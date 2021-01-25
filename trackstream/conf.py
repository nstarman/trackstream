# -*- coding: utf-8 -*-

"""TrackStream Configurations."""


__all__ = [
    "conf",  # configuration instance
]


##############################################################################
# IMPORTS

# THIRD PARTY
from astropy import config as mod  # configuration module

##############################################################################
# CODE
##############################################################################


class Conf(mod.ConfigNamespace):
    """Configuration parameters for `trackstream`."""

    # lmfit
    use_lmfit = mod.ConfigItem(
        False,
        description="Use lmfit.",
        cfgtype="boolean(default=False)",
    )


# /class


# -------------------------------------------------------------------
# configuration instance

conf = Conf()

##############################################################################
# END
