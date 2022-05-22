# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.utils.misc`."""


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.units as u
import numpy as np
import pytest
from astropy.coordinates import Angle

# LOCAL
from trackstream.utils.misc import (
    ABCwAMeta,
    abstract_attribute,
    covariance_ellipse,
    intermix_arrays,
    is_structured,
    make_shuffler,
)

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


def test_make_shuffler():
    """Test `trackstream.utils.misc.make_shuffler`."""
    # default Generator
    shuffler, undo = make_shuffler(10, rng=None)
    assert isinstance(shuffler, np.ndarray) & isinstance(undo, np.ndarray)
    assert np.all(shuffler[undo] == np.arange(10))

    # Generator
    shuffler, undo = make_shuffler(10, rng=np.random.default_rng())
    assert isinstance(shuffler, np.ndarray) & isinstance(undo, np.ndarray)
    assert np.all(shuffler[undo] == np.arange(10))

    # RandomState
    shuffler, undo = make_shuffler(10, rng=np.random.RandomState())
    assert isinstance(shuffler, np.ndarray) & isinstance(undo, np.ndarray)
    assert np.all(shuffler[undo] == np.arange(10))


def test_abstract_attribute():
    """
    Test :func:`trackstream.utils.misc.abstract_attribute` and
    :class:`trackstream.utils.misc.ABCwMeta`.
    """
    # It doesn't work if wrong metaclass
    class ABClass1:
        attr: int = abstract_attribute()

    assert ABClass1()  # instantiate

    # It will work if the metaclass is ABCwAMeta
    class ABClass2(metaclass=ABCwAMeta):
        attr: int = abstract_attribute()

    with pytest.raises(NotImplementedError, match="cannot instantiate"):
        ABClass2()

    class Class(ABClass2):
        def __init__(self) -> None:
            self.attr = 2  # not abstract

    inst = Class()
    assert inst.attr == 2


def test_is_structured():
    """Test :func:`trackstream.utils.misc.is_structured`."""
    # Not structured
    assert not is_structured(None)
    assert not is_structured(np.array(1))

    # structured
    assert is_structured(np.array((1, 2), dtype=[("f1", int), ("f2", int)]))


@pytest.mark.parametrize(
    ("P", "nstd", "expected"),
    [
        ([[1, 0], [0, 1]], 1, (0, [1, 1])),
        ([[1, 0], [0, 1]], 2, (0, [2, 2])),
        ([[0, 1], [1, 0]], 1, (Angle(-90, u.deg), [1, 1])),
        ([[0, 1], [1, 0]], 2, (Angle(-90, u.deg), [2, 2])),
    ],
)
def test_covariance_ellipse(P, nstd, expected):
    """Test :func:`trackstream.utils.misc.covariance_ellipse`."""
    (orientation, wh) = covariance_ellipse(P, nstd=nstd)
    assert orientation == expected[0]
    assert np.array_equal(wh, expected[1])
