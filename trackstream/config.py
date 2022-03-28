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
    """Configuration parameters for :mod:`~trackstream`."""

    default_frame = _config.ConfigItem(
        "icrs",
        description="Default Frame.",
        cfgtype="string",
    )

    # Preprocessing
    # -------------

    use_lmfit = _config.ConfigItem(
        False,
        description="Use lmfit.",
        cfgtype="boolean(default=False)",
    )

    # Processing
    # ----------

    use_filterpy = _config.ConfigItem(
        False,
        description="Use FilterPy.",
        cfgtype="boolean(default=False)",
    )


conf = Conf()
# /class

#############################################################################
# END
