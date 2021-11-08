# -*- coding: utf-8 -*-
"""Configure Test Suite.

This file is used to configure the behavior of pytest when using the Astropy
test infrastructure. It needs to live inside the package in order for it to
get picked up when running the tests inside an interpreter using
packagename.test

"""

# STDLIB
import os

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest

# LOCAL
import trackstream.utils.interpolated_coordinates as icoord
from trackstream.utils.path import Path

try:
    # THIRD PARTY
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS

    ASTROPY_HEADER = True
except ImportError:
    ASTROPY_HEADER = False

# /if


def pytest_configure(config):
    """Configure Pytest with Astropy.

    Parameters
    ----------
    config : pytest configuration

    """
    if ASTROPY_HEADER:

        config.option.astropy_header = True

        # Customize the following lines to add/remove entries from the list of
        # packages for which version numbers are displayed when running the
        # tests.
        PYTEST_HEADER_MODULES.pop("Pandas", None)
        PYTEST_HEADER_MODULES["scikit-image"] = "skimage"

        # LOCAL
        from . import __version__

        packagename = os.path.basename(os.path.dirname(__file__))
        TESTED_VERSIONS[packagename] = __version__

    return


# /def


# ------------------------------------------------------
# doctest fixtures


@pytest.fixture(autouse=True)
def add_numpy(doctest_namespace):
    """Add NumPy to Pytest.

    Parameters
    ----------
    doctest_namespace : namespace

    """
    # THIRD PARTY
    import numpy

    # add to namespace
    doctest_namespace["np"] = numpy


@pytest.fixture(autouse=True)
def add_astropy(doctest_namespace):
    """Add Astropy stuff to Pytest.

    Parameters
    ----------
    doctest_namespace : namespace

    """
    # THIRD PARTY
    import astropy.coordinates as coord
    import astropy.units

    # add to namespace
    doctest_namespace["coord"] = coord
    doctest_namespace["u"] = astropy.units

    # extras
    # THIRD PARTY
    from astropy.visualization import quantity_support, time_support

    quantity_support()
    time_support()


# ------------------------------------------------------


@pytest.fixture
def num():
    return 40


@pytest.fixture
def affine(num):
    return np.linspace(0, 10, num=num) * u.deg


@pytest.fixture
def frame():
    return coord.ICRS()


@pytest.fixture
def dif(num):
    d = coord.CartesianDifferential(
        d_x=np.linspace(3, 4, num=num) * (u.km / u.s),
        d_y=np.linspace(4, 5, num=num) * (u.km / u.s),
        d_z=np.linspace(5, 6, num=num) * (u.km / u.s),
    )
    return d


@pytest.fixture
def idif(dif, affine):
    return icoord.InterpolatedDifferential(dif, affine=affine)


@pytest.fixture
def rep(dif, num):
    r = coord.CartesianRepresentation(
        x=np.linspace(0, 1, num=num) * u.kpc,
        y=np.linspace(1, 2, num=num) * u.kpc,
        z=np.linspace(2, 3, num=num) * u.kpc,
        differentials=dif,
    )
    return r


@pytest.fixture
def irep(rep, affine):
    return icoord.InterpolatedRepresentation(rep, affine=affine)


@pytest.fixture
def crd(frame, rep):
    c = frame.realize_frame(rep)
    return c


@pytest.fixture
def icrd(crd, affine):
    return icoord.InterpolatedCoordinateFrame(crd, affine=affine)


@pytest.fixture
def scrd(crd):
    return coord.SkyCoord(crd)


@pytest.fixture
def iscrd(scrd, affine):
    return icoord.InterpolatedSkyCoord(scrd, affine=affine)


@pytest.fixture
def path_cls():
    return Path


@pytest.fixture
def path(path_cls, iscrd, width):
    p = path_cls(iscrd, width, name="conftest")
    return p


@pytest.fixture
def width():
    return 100 * u.pc  # TODO! have variable function
