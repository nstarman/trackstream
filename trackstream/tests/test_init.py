# -*- coding: utf-8 -*-

"""Initiation Tests for :mod:`~trackstream`."""

__all__ = [
    "test_has_version",
]


##############################################################################
# IMPORTS


##############################################################################
# TESTS
##############################################################################


def test_has_version():
    """Test :mod:`~trackstream` has attribute __version__."""
    # PROJECT-SPECIFIC
    import trackstream

    assert hasattr(trackstream, "__version__"), "No version!"


# /def


# -------------------------------------------------------------------


##############################################################################
# END
