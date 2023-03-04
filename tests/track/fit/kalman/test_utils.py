"""Testing :mod:`~trackstream.utils.coord_utils`."""


import numpy as np

from trackstream.track.fit.kalman.utils import intermix_arrays

##############################################################################
# TESTS
##############################################################################


def test_intermix_arrays():
    """Test `trackstream.utils.misc.intermix_arrays`."""
    # Mix single scalar array (does nothing)
    x = np.arange(5)
    got = intermix_arrays(x)
    expect = np.array([0, 1, 2, 3, 4])
    assert np.all(got == expect)

    # Mix two scalar arrays
    y = np.arange(5, 10)
    got = intermix_arrays(x, y)
    expect = np.array([0, 5, 1, 6, 2, 7, 3, 8, 4, 9])
    assert np.all(got == expect)

    # Mix multiple scalar arrays
    z = np.arange(10, 15)
    got = intermix_arrays(x, y, z)
    expect = np.array([0, 5, 10, 1, 6, 11, 2, 7, 12, 3, 8, 13, 4, 9, 14])
    assert np.all(got == expect)

    # Mix single ND array
    xx = np.c_[x, y]
    got = intermix_arrays(xx)
    expect = np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]])
    assert np.all(got == expect)

    # Mix two ND arrays
    yy = np.c_[z, np.arange(15, 20)]
    got = intermix_arrays(xx, yy)
    expect = np.array([[0, 10, 1, 11, 2, 12, 3, 13, 4, 14], [5, 15, 6, 16, 7, 17, 8, 18, 9, 19]])
    assert np.all(got == expect)
