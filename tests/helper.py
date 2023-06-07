# see LICENSE.rst

"""Test helpers for :mod:`~trackstream`."""
from typing import Any

from astropy.coordinates import Angle, SkyCoord
from astropy.table import QTable
import astropy.units as u
from astropy.utils.data import get_pkg_data_path
import pytest


@pytest.fixture(scope="session")
def IbataEtAl2017() -> dict[str, Any]:
    """Fixture returning data from Ibata et al (2017) [1]_.

    Returns
    -------
    dict[str, Any]

    References
    ----------
    .. [1] Ibata, R., Lewis, G., Thomas, G., Martin, N., & Chapman, S. (2018).
        VizieR Online Data Catalog: Large spectrosc. survey of Palomar 5 stellar
        stream (Ibata+, 2017). VizieR Online Data Catalog, J/ApJ/842/120.
    """
    # Read Data
    path = get_pkg_data_path("data", "IbataEtAl2017.ecsv", package="trackstream")
    data = QTable.read(path)

    # Manually construct origin
    origin = SkyCoord(ra=Angle("15h 16m 05.3s"), dec=Angle("-00:06:41 degree"))  # 23 kpc

    # Manually add arm labels
    data["tail"] = "arm1"
    data["tail"][data["ra"] < origin.ra] = "arm2"

    # Make error table  # TODO: real errors?
    data_err = QTable()
    data_err["ra_err"] = 0 * data["ra"]  # (for the shape)
    data_err["dec_err"] = u.Quantity(0, u.deg)

    return {"name": IbataEtAl2017, "data": data, "origin": origin, "data_error": data_err}


# @pytest.fixture
# def STREAM_DATA_SET(IbataEtAl2017) -> Iterator[Dict[str, Any]]:
#     """Fixture yielding all stream data sets."""
