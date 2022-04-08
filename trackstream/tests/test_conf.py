# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.conf`."""

__all__ = ["Test_conf"]


##############################################################################
# IMPORTS

# LOCAL
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
