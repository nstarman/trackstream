# -*- coding: utf-8 -*-
"""Configure Test Suite.

This file is used to configure the behavior of pytest when using the Astropy
test infrastructure. It needs to live inside the package in order for it to
get picked up when running the tests inside an interpreter using
packagename.test

"""

# STDLIB
import os
from typing import Any, Dict, Type, cast

# THIRD PARTY
import astropy.units as u
import pytest
from astropy.coordinates import ICRS, Angle, BaseCoordinateFrame, BaseDifferential
from astropy.coordinates import BaseRepresentation, CartesianDifferential, CartesianRepresentation
from astropy.coordinates import SkyCoord
from astropy.units import Quantity
from interpolated_coordinates import InterpolatedCoordinateFrame, InterpolatedDifferential
from interpolated_coordinates import InterpolatedRepresentation, InterpolatedSkyCoord
from numpy import linspace

# LOCAL
from trackstream.track.path import Path

try:
    # THIRD PARTY
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS  # type: ignore

    ASTROPY_HEADER = True
except ImportError:
    ASTROPY_HEADER = False


def pytest_configure(config: pytest.Config) -> None:
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


# ------------------------------------------------------
# doctest fixtures


@pytest.fixture(autouse=True)
def add_numpy(doctest_namespace: Dict[str, Any]) -> None:
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
def add_astropy(doctest_namespace: Dict[str, Any]) -> None:
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


@pytest.fixture(scope="session")
def num() -> int:
    """Fixture returning the number of affine ticks"""
    return 40


@pytest.fixture(scope="session")
def affine(num: int) -> Angle:
    """Fixture returning the affine |Angle|."""
    afn = Angle(linspace(0, 10, num=num), u.deg)
    return afn


@pytest.fixture(scope="session")
def dif_type() -> Type[BaseDifferential]:
    """Fixture returning the differential type."""
    dt: Type[BaseDifferential] = CartesianDifferential
    return dt


@pytest.fixture(scope="session")
def dif(dif_type: Type[BaseDifferential], num: int) -> BaseDifferential:
    """Fixture returning the differential."""
    d = dif_type(
        d_x=linspace(3, 4, num=num) * (u.km / u.s),
        d_y=linspace(4, 5, num=num) * (u.km / u.s),
        d_z=linspace(5, 6, num=num) * (u.km / u.s),
    )
    return d


@pytest.fixture(scope="session")
def idif(dif: BaseDifferential, affine: Angle) -> InterpolatedDifferential:
    """Fixture returning the interpolated differential."""
    return InterpolatedDifferential(dif, affine=affine)


@pytest.fixture(scope="session")
def rep_type() -> Type[CartesianRepresentation]:
    """Fixture returning the differential type."""
    return CartesianRepresentation


@pytest.fixture(scope="session")
def rep(
    rep_type: Type[CartesianRepresentation], dif: BaseDifferential, num: int
) -> CartesianRepresentation:
    """Fixture returning the representation, with attached differentials."""
    r = rep_type(
        x=linspace(0, 1, num=num) * u.kpc,
        y=linspace(1, 2, num=num) * u.kpc,
        z=linspace(2, 3, num=num) * u.kpc,
        differentials=dif,
    )
    return r


@pytest.fixture(scope="session")
def irep(rep: BaseRepresentation, affine: Angle) -> InterpolatedRepresentation:
    """Fixture returning the interpolated representation."""
    return InterpolatedRepresentation(rep, affine=affine)


@pytest.fixture(scope="session")
def frame(
    rep_type: Type[BaseRepresentation], dif_type: Type[BaseDifferential]
) -> BaseCoordinateFrame:
    """Fixture returning the frame, |ICRS|."""
    frame = ICRS(representation_type=rep_type, differential_type=dif_type)
    return frame


@pytest.fixture(scope="session")
def crd(frame: BaseCoordinateFrame, rep: BaseRepresentation) -> BaseCoordinateFrame:
    """Fixture returning the coordinate frame."""
    c = frame.realize_frame(rep, representation_type=type(rep))
    return c


@pytest.fixture(scope="session")
def icrd(crd: BaseCoordinateFrame, affine: Angle) -> InterpolatedCoordinateFrame:
    """Fixture returning the interpolated coordinate frame."""
    return InterpolatedCoordinateFrame(crd, affine=affine)


@pytest.fixture(scope="session")
def scrd(crd: BaseCoordinateFrame) -> SkyCoord:
    """Fixture returning the |SkyCoord|."""
    return SkyCoord(crd)


@pytest.fixture(scope="session")
def iscrd(scrd: SkyCoord, affine: Angle) -> InterpolatedSkyCoord:
    """Fixture returning the interpolated |SkyCoord|."""
    return InterpolatedSkyCoord(scrd, affine=affine)


@pytest.fixture(scope="session")
def path_cls() -> Type[Path]:
    """Fixture returning the Path class."""
    return Path


@pytest.fixture(scope="session")
def path(path_cls: Type[Path], iscrd: InterpolatedSkyCoord, width: Quantity) -> Path:
    """Fixture returning the Path instance."""
    p = path_cls(iscrd, width, name="conftest")
    return p


@pytest.fixture(scope="session")
def width() -> Quantity:
    """Fixture returning the width."""
    return Quantity(100, u.pc)  # TODO! have variable function


@pytest.fixture(scope="session")
def index_on() -> int:
    """Fixture returning the int."""
    return 10


@pytest.fixture(scope="session")
def affine_on(affine: Angle, index_on: int) -> Angle:
    """Fixture returning the affine parameter at one point."""
    return affine[index_on]


@pytest.fixture(scope="session")
def point_on(crd: BaseCoordinateFrame, index_on: int) -> BaseCoordinateFrame:
    """Fixture returning the coordinate at one point."""
    c = cast(BaseCoordinateFrame, crd[index_on])
    return c


# @pytest.fixture
# def point_off(affine):
#     i = affine[10]
#     return crd[i]
