"""Configure Test Suite.

This file is used to configure the behavior of pytest when using the Astropy
test infrastructure. It needs to live inside the package in order for it to
get picked up when running the tests inside an interpreter using
packagename.test

"""

from __future__ import annotations

# STDLIB
import os
from typing import Any, cast

# THIRD PARTY
import astropy.coordinates as coords
import astropy.units as u
import numpy as np
import pytest
from astropy.units import Quantity
from interpolated_coordinates import (
    InterpolatedCoordinateFrame,
    InterpolatedDifferential,
    InterpolatedRepresentation,
    InterpolatedSkyCoord,
)

# LOCAL
from trackstream.track.path import Path
from trackstream.track.width import Cartesian1DWidth
from trackstream.track.width.interpolated import InterpolatedWidths

__all__: list[str] = []


try:
    # THIRD PARTY
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS

    ASTROPY_HEADER = True
except ImportError:
    ASTROPY_HEADER = False


def pytest_configure(config: pytest.Config) -> None:
    """Configure Pytest with Astropy.

    Parameters
    ----------
    config : pytest configuration
    """
    if not ASTROPY_HEADER:
        return

    config.option.astropy_header = True

    # Customize the following lines to add/remove entries from the list of
    # packages for which version numbers are displayed when running the
    # tests.
    PYTEST_HEADER_MODULES.pop("Pandas", None)

    # STDLIB
    from importlib.metadata import version

    packagename = os.path.basename(os.path.dirname(__file__))
    TESTED_VERSIONS[packagename] = version(packagename)


# ------------------------------------------------------
# doctest fixtures


@pytest.fixture(autouse=True)  # type: ignore
def add_numpy(doctest_namespace: dict[str, Any]) -> None:
    """Add NumPy to Pytest.

    Parameters
    ----------
    doctest_namespace : namespace

    """
    # THIRD PARTY
    import numpy

    # add to namespace
    doctest_namespace["np"] = numpy


@pytest.fixture(autouse=True)  # type: ignore
def add_astropy(doctest_namespace: dict[str, Any]) -> None:
    """Add Astropy stuff to Pytest.

    Parameters
    ----------
    doctest_namespace : namespace

    """
    # THIRD PARTY
    import astropy.coordinates as coords
    import astropy.units

    # add to namespace
    doctest_namespace["coords"] = coords
    doctest_namespace["u"] = astropy.units

    # extras
    # THIRD PARTY
    from astropy.visualization import quantity_support, time_support

    quantity_support()
    time_support()


#####################################################################


@pytest.fixture(scope="session")
def num() -> int:
    """Fixture returning the number of affine ticks."""
    return 40


@pytest.fixture(scope="session")
def affine(num: int) -> coords.Angle:
    """Fixture returning the affine |Angle|."""
    afn = coords.Angle(np.linspace(0, 10, num=num), u.deg)
    return afn


@pytest.fixture(scope="session")
def dif_type() -> type[coords.BaseDifferential]:
    """Fixture returning the differential type."""
    dt: type[coords.BaseDifferential] = coords.CartesianDifferential
    return dt


@pytest.fixture(scope="session")
def dif(dif_type: type[coords.BaseDifferential], num: int) -> coords.BaseDifferential:
    """Fixture returning the differential."""
    return dif_type(
        d_x=np.linspace(3, 4, num=num) * (u.km / u.s),
        d_y=np.linspace(4, 5, num=num) * (u.km / u.s),
        d_z=np.linspace(5, 6, num=num) * (u.km / u.s),
    )


@pytest.fixture(scope="session")
def idif(dif: coords.BaseDifferential, affine: coords.Angle) -> InterpolatedDifferential:
    """Fixture returning the interpolated differential."""
    return InterpolatedDifferential(dif, affine=affine)


@pytest.fixture(scope="session")
def rep_type() -> type[coords.CartesianRepresentation]:
    """Fixture returning the differential type."""
    return coords.CartesianRepresentation


@pytest.fixture(scope="session")
def rep(
    rep_type: type[coords.CartesianRepresentation], dif: coords.BaseDifferential, num: int
) -> coords.CartesianRepresentation:
    """Fixture returning the representation, with attached differentials."""
    r = rep_type(
        x=np.linspace(0, 1, num=num) * u.kpc,
        y=np.linspace(1, 2, num=num) * u.kpc,
        z=np.linspace(2, 3, num=num) * u.kpc,
        differentials=dif,
    )
    return r


@pytest.fixture(scope="session")
def irep(rep: coords.BaseRepresentation, affine: coords.Angle) -> InterpolatedRepresentation:
    """Fixture returning the interpolated representation."""
    return InterpolatedRepresentation(rep, affine=affine)


@pytest.fixture(scope="session")
def frame(
    rep_type: type[coords.BaseRepresentation], dif_type: type[coords.BaseDifferential]
) -> coords.BaseCoordinateFrame:
    """Fixture returning the frame, |ICRS|."""
    frame = coords.ICRS(representation_type=rep_type, differential_type=dif_type)
    return frame


@pytest.fixture(scope="session")
def crd(frame: coords.BaseCoordinateFrame, rep: coords.BaseRepresentation) -> coords.BaseCoordinateFrame:
    """Fixture returning the coordinate frame."""
    c = frame.realize_frame(rep, representation_type=type(rep))
    return c


@pytest.fixture(scope="session")
def icrd(crd: coords.BaseCoordinateFrame, affine: coords.Angle) -> InterpolatedCoordinateFrame:
    """Fixture returning the interpolated coordinate frame."""
    return InterpolatedCoordinateFrame(crd, affine=affine)


@pytest.fixture(scope="session")
def scrd(crd: coords.BaseCoordinateFrame) -> coords.SkyCoord:
    """Fixture returning the |SkyCoord|."""
    return coords.SkyCoord(crd)


@pytest.fixture(scope="session")
def iscrd(scrd: coords.SkyCoord, affine: coords.Angle) -> InterpolatedSkyCoord:
    """Fixture returning the interpolated |SkyCoord|."""
    return InterpolatedSkyCoord(scrd, affine=affine)


@pytest.fixture(scope="session")
def path_cls() -> type[Path]:
    """Fixture returning the Path class."""
    return Path


@pytest.fixture(scope="session")
def path(path_cls: type[Path], iscrd: InterpolatedSkyCoord, width: InterpolatedWidths) -> Path:
    """Fixture returning the Path instance."""
    p = path_cls(iscrd, width, name="conftest")
    return p


@pytest.fixture(scope="session")
def width(num: int) -> InterpolatedWidths:
    """Fixture returning the width."""
    x = Quantity(np.ones(num) * 100, u.pc)
    ws = InterpolatedWidths.from_format({"length": Cartesian1DWidth(x)})
    return ws


@pytest.fixture(scope="session")
def index_on() -> int:
    """Fixture returning the int."""
    return 10


@pytest.fixture(scope="session")
def affine_on(affine: coords.Angle, index_on: int) -> coords.Angle:
    """Fixture returning the affine parameter at one point."""
    return cast("coords.Angle", affine[index_on])


@pytest.fixture(scope="session")
def point_on(crd: coords.BaseCoordinateFrame, index_on: int) -> coords.BaseCoordinateFrame:
    """Fixture returning the coordinate at one point."""
    return cast("coords.BaseCoordinateFrame", crd[index_on])
