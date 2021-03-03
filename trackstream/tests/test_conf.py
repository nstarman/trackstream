# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.conf`."""

__all__ = [
    "Test_conf",
]


##############################################################################
# IMPORTS

# PROJECT-SPECIFIC
from trackstream.config import Conf

##############################################################################
# TESTS
##############################################################################


class Test_conf(object):
    """Test module configuration instance."""

    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing.

        A new configuration instance is created so as not to interfere
        with the one used in the code.

        """
        cls.conf = Conf()

    # /def

    # -------------------------------

    def test_use_lmfit(self):
        """Test ``use_lmfit`` configuration."""
        assert self.conf.use_lmfit is False

    # /def


# /class


# -------------------------------------------------------------------


##############################################################################
# END
