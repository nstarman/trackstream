"""Configure Test Suite.

This file is used to configure the behavior of pytest when using the Astropy
test infrastructure. It needs to live inside the package in order for it to
get picked up when running the tests inside an interpreter using
packagename.test

"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING, Any, cast

import astropy.coordinates as coords
from astropy.coordinates import Angle, CartesianDifferential, CartesianRepresentation, SkyCoord
import astropy.units as u
from astropy.units import Quantity
from interpolated_coordinates import (
    InterpolatedCoordinateFrame,
    InterpolatedDifferential,
    InterpolatedRepresentation,
    InterpolatedSkyCoord,
)
import numpy as np
import pytest

from trackstream.track.path import Path
from trackstream.track.width import Cartesian1DWidth
from trackstream.track.width.interpolated import InterpolatedWidths

__all__: list[str] = []

if TYPE_CHECKING:
    from astropy.coordinates import BaseCoordinateFrame, BaseDifferential, BaseRepresentation


try:
    from pytest_astropy_header.display import PYTEST_HEADER_MODULES, TESTED_VERSIONS

    ASTROPY_HEADER = True
except ImportError:
    ASTROPY_HEADER = False


def pytest_configure(config: pytest.Config) -> None:
    """Configure Pytest with Astropy.

    Parameters
    ----------
    config : `pytest.Config`
        Pytest configuration object.

    """
    if not ASTROPY_HEADER:
        return

    config.option.astropy_header = True

    # Customize the following lines to add/remove entries from the list of
    # packages for which version numbers are displayed when running the
    # tests.
    PYTEST_HEADER_MODULES.pop("Pandas", None)

    from importlib.metadata import version

    packagename = pathlib.Path(__file__).parent.name
    TESTED_VERSIONS[packagename] = version(packagename)


# ------------------------------------------------------
# doctest fixtures


@pytest.fixture(autouse=True)
def _add_numpy(doctest_namespace: dict[str, Any]) -> None:
    """Add NumPy to Pytest.

    Parameters
    ----------
    doctest_namespace : namespace
        dictionary to add to.

    """
    import numpy as np

    # add to namespace
    doctest_namespace["np"] = np


@pytest.fixture(autouse=True)
def _add_astropy(doctest_namespace: dict[str, Any]) -> None:
    """Add Astropy stuff to Pytest.

    Parameters
    ----------
    doctest_namespace : namespace
        dictionary to add to.

    """
    import astropy.coordinates as coords

    # add to namespace
    doctest_namespace["coords"] = coords
    doctest_namespace["u"] = u

    # extras
    from astropy.visualization import quantity_support, time_support

    quantity_support()
    time_support()


#####################################################################


@pytest.fixture(scope="session")
def num() -> int:
    """Fixture returning the number of affine ticks."""
    return 40


@pytest.fixture(scope="session")
def affine(num: int) -> Angle:
    """Fixture returning the affine |Angle|."""
    return Angle(np.linspace(0, 10, num=num), u.deg)


@pytest.fixture(scope="session")
def dif_type() -> type[BaseDifferential]:
    """Fixture returning the differential type."""
    return CartesianDifferential


@pytest.fixture(scope="session")
def dif(dif_type: type[BaseDifferential], num: int) -> BaseDifferential:
    """Fixture returning the differential."""
    return dif_type(
        d_x=np.linspace(3, 4, num=num) * (u.km / u.s),
        d_y=np.linspace(4, 5, num=num) * (u.km / u.s),
        d_z=np.linspace(5, 6, num=num) * (u.km / u.s),
    )


@pytest.fixture(scope="session")
def idif(dif: BaseDifferential, affine: Angle) -> InterpolatedDifferential:
    """Fixture returning the interpolated differential."""
    return InterpolatedDifferential(dif, affine=affine)


@pytest.fixture(scope="session")
def rep_type() -> type[CartesianRepresentation]:
    """Fixture returning the differential type."""
    return CartesianRepresentation


@pytest.fixture(scope="session")
def rep(
    rep_type: type[CartesianRepresentation],
    dif: BaseDifferential,
    num: int,
) -> CartesianRepresentation:
    """Fixture returning the representation, with attached differentials."""
    return rep_type(
        x=np.linspace(0, 1, num=num) * u.kpc,
        y=np.linspace(1, 2, num=num) * u.kpc,
        z=np.linspace(2, 3, num=num) * u.kpc,
        differentials=dif,
    )


@pytest.fixture(scope="session")
def irep(rep: BaseRepresentation, affine: Angle) -> InterpolatedRepresentation:
    """Fixture returning the interpolated representation."""
    return InterpolatedRepresentation(rep, affine=affine)


@pytest.fixture(scope="session")
def frame(
    rep_type: type[BaseRepresentation],
    dif_type: type[BaseDifferential],
) -> BaseCoordinateFrame:
    """Fixture returning the frame, |ICRS|."""
    return coords.ICRS(representation_type=rep_type, differential_type=dif_type)


@pytest.fixture(scope="session")
def crd(frame: BaseCoordinateFrame, rep: BaseRepresentation) -> BaseCoordinateFrame:
    """Fixture returning the coordinate frame."""
    return frame.realize_frame(rep, representation_type=type(rep))


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
def path_cls() -> type[Path]:
    """Fixture returning the Path class."""
    return Path


@pytest.fixture(scope="session")
def path(path_cls: type[Path], iscrd: InterpolatedSkyCoord, width: InterpolatedWidths) -> Path:
    """Fixture returning the Path instance."""
    return path_cls(iscrd, width, name="conftest")


@pytest.fixture(scope="session")
def width(num: int) -> InterpolatedWidths:
    """Fixture returning the width."""
    x = Quantity(np.ones(num) * 100, u.pc)
    return InterpolatedWidths.from_format({"length": Cartesian1DWidth(x)})


@pytest.fixture(scope="session")
def index_on() -> int:
    """Fixture returning the int."""
    return 10


@pytest.fixture(scope="session")
def affine_on(affine: Angle, index_on: int) -> Angle:
    """Fixture returning the affine parameter at one point."""
    return cast("Angle", affine[index_on])


@pytest.fixture(scope="session")
def point_on(crd: BaseCoordinateFrame, index_on: int) -> BaseCoordinateFrame:
    """Fixture returning the coordinate at one point."""
    return cast("BaseCoordinateFrame", crd[index_on])
