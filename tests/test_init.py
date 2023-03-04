"""Initiation Tests for :mod:`~trackstream`."""

##############################################################################
# TESTS
##############################################################################


def test_has_version():
    """Test :mod:`~trackstream` has attribute __version__."""
    import trackstream

    assert hasattr(trackstream, "__version__"), "No version!"
