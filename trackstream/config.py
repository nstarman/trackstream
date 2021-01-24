# -*- coding: utf-8 -*-
# see LICENSE.rst

"""Configuration."""


##############################################################################
# IMPORTS

# THIRD PARTY
from astropy import config as _config

__all__ = [
    "conf",
]

#############################################################################
# CONFIGURATIONS


class Conf(_config.ConfigNamespace):
    """Configuration parameters for :mod:`~starkman_thesis`."""

    default_frame = _config.ConfigItem(
        "icrs",
        description="Default Frame.",
        cfgtype="string",
    )


conf = Conf()
# /class

#############################################################################
# END
