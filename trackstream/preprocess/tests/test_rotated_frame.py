# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.preprocess.rotated_frame`.

.. todo::

    properly use pytest fixtures

"""

__all__ = [
    "test_cartesian_model",
    "test_residual",
]


##############################################################################
# IMPORTS

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest
from astropy.tests.helper import assert_quantity_allclose

# LOCAL
from trackstream.example_data import example_coords
from trackstream.example_data.tests import data
from trackstream.preprocess import rotated_frame
from trackstream.setup_package import HAS_LMFIT

if HAS_LMFIT:
    # THIRD PARTY
    import lmfit as lf


##############################################################################
# Fixtures


##############################################################################
# TESTS
##############################################################################


@pytest.mark.skipif(not HAS_LMFIT, reason="needs `lmfit`")
def test_scipy_residual_to_lmfit():
    """Test ``scipy_residual_to_lmfit``."""

    @rotated_frame.scipy_residual_to_lmfit(param_order=["var0", "var1"])
    def residual(variables, x, data):
        var0 = variables[0]
        var1 = variables[1]
        return data - (var0 * x + var1)

    # normal residual
    vars = [1, 2]
    x = np.arange(0, 10)
    expect = vars[0] * x + vars[1]
    assert all(np.equal(residual(vars, x, expect), np.zeros_like(x)))

    # scipy residual
    ps = lf.Parameters()
    ps.add_many(("var0", 1), ("var1", 2))
    assert all(np.equal(residual.lmfit(ps, x, expect), np.zeros_like(x)))


@pytest.mark.parametrize(
    "test_data,lon,lat,rotation,deg,expected_data",
    [
        (
            data.icrs,  # TODO as fixture
            example_coords.RA,
            example_coords.DEC,
            example_coords.ICRS_ROTATION,
            True,
            data.ricrs,  # TODO as fixture
        ),
        (
            data.icrs,  # TODO fixtures
            example_coords.RA,
            example_coords.DEC,
            example_coords.ICRS_ROTATION,
            False,
            data.ricrs,  # TODO as fixture
        ),
        (
            data.gcentric,  # TODO as fixture
            example_coords.LON,
            example_coords.LAT,
            example_coords.GALACTOCENTRIC_ROTATION,
            True,
            data.rgcentric,  # TODO as fixture
        ),
        (
            data.gcentric,  # TODO as fixture
            example_coords.LON,
            example_coords.LAT,
            example_coords.GALACTOCENTRIC_ROTATION,
            False,
            data.rgcentric,  # TODO as fixture
        ),
        # TODO the other datasets
    ],
)
def test_cartesian_model(
    test_data,
    lon,
    lat,
    rotation,
    deg: bool,
    expected_data,
):
    """Test `~trackstream.preprocess.rotated_frame.cartesian_model`."""
    # --------------
    # setup

    angle_unit = u.deg if deg else u.rad  # get unit
    # reverse map: value, key map in expected_data
    rev_names = {v: k for k, v in expected_data.representation_component_names.items()}

    # --------------
    # apply model

    r, lon, lat = rotated_frame.cartesian_model(
        test_data.cartesian,
        lon=lon,
        lat=lat,
        rotation=rotation,
        deg=deg,
    )
    # longitude and latitude
    lon = np.mod(coord.Longitude(lon, unit=angle_unit), 180 * u.deg)
    lat = np.mod(coord.Longitude(lat, unit=angle_unit), 180 * u.deg)

    # --------------
    # Testing r, lon, lat
    expected_r = getattr(expected_data, rev_names["distance"])
    expected_lon = getattr(expected_data, rev_names["lon"])
    expected_lat = getattr(expected_data, rev_names["lat"])

    assert_quantity_allclose(r, expected_r, atol=1e-10 * expected_r.unit)
    assert_quantity_allclose(
        lon,
        np.mod(expected_lon, 180 * u.deg),
        atol=1e-10 * expected_lon.unit,
    )
    # there's 1 that's almost 180 deg, but not quite and needs to get
    # modded down.
    i = np.isclose(lat, 180 * u.deg)
    lat[i] = 180 * u.deg + np.abs(180 * u.deg - lat[i])
    assert_quantity_allclose(
        np.mod(lat.to(u.deg), 180 * u.deg),
        np.mod(expected_lat, 180 * u.deg),
        atol=1e-10 * expected_lat.unit,
    )


# /def


# ===================================================================


class TestRotatedFrameFitter:
    def setup_class(self):

        # origin
        origin = coord.ICRS(ra=20 * u.deg, dec=30 * u.deg)

        # icrs
        rsc = example_coords.RotatedICRS(
            phi1=np.linspace(-np.pi, np.pi, 128) * u.radian,
            phi2=np.zeros(128) * u.radian,
            pm_phi1_cosphi2=np.ones(128) * 10 * u.mas / u.yr,
            pm_phi2=np.zeros(128) * u.mas / u.yr,
        )
        self.data = rsc.transform_to(coord.ICRS())

        self.RFF = rotated_frame.RotatedFrameFitter(
            self.data,
            origin=origin,
            origin_lim=0.001 * u.deg,
            fix_origin=True,
        )

    def test_default_fit_options(self):
        """Test ``RotatedFrameFitter.default_fit_options``."""
        got = self.RFF.default_fit_options
        expect = dict(fix_origin=True, use_lmfit=None, leastsquares=False, align_v=True)

        assert (set(got.keys()) == set(expect.keys())) & all(
            [expect[k] == v for k, v in got.items()],
        )

    def test_set_bounds(self):
        """Test ``RotatedFrameFitter.set_bounds``."""
        # original value
        expect = np.array([[-180.0, 180.0], [19.999, 20.001], [29.999, 30.001]])
        assert np.all(np.equal(self.RFF.bounds, expect))

        # set different value: default
        self.RFF.set_bounds()
        expect = np.array([[-180.0, 180.0], [19.995, 20.005], [29.995, 30.005]])
        assert np.all(np.equal(self.RFF.bounds, expect))

        # back to old value
        self.RFF.set_bounds(origin_lim=0.001 * u.deg)
        expect = np.array([[-180.0, 180.0], [19.999, 20.001], [29.999, 30.001]])
        assert np.all(np.equal(self.RFF.bounds, expect))

    @pytest.mark.skip("TODO!")
    def test_align_v_positive_lon(self):
        """Test ``RotatedFrameFitter.align_v_positive_lon``."""
        assert False

    @pytest.mark.parametrize(
        "rotation,expected_lat",
        [(example_coords.ICRS_ROTATION, 0)],  # TODO! more
    )
    def test_residual_scalar(self, rotation, expected_lat):
        """Test ``RotatedFrameFitter.residual``."""
        res = self.RFF.residual(rotation, scalar=True)
        # compare result and expected latitude
        np.testing.assert_allclose(res, expected_lat, atol=1e-12)

    @pytest.mark.parametrize(
        "rotation,expected_lat",
        [(example_coords.ICRS_ROTATION, data.ricrs.phi2)],  # todo internally
    )
    def test_residual_array(self, rotation, expected_lat):
        """Test ``RotatedFrameFitter.residual``."""
        res = self.RFF.residual(rotation, scalar=False)
        # compare result and expected latitude
        np.testing.assert_allclose(res, expected_lat.to_value(u.deg), atol=1e-10)

    # ---------------------------------------------------------------

    def test_fit_no_rot0(self):
        """Test ``fit`` without ``rot0`` specified."""
        with pytest.raises(ValueError, match="no prespecified `rot0`"):
            self.RFF.fit(rot0=None)

    def test_fit_has_rot0(self):
        """
        Test ``fit`` with ``rot0`` specified and all else defaults.
        This triggers all the ``if X is None`` checks, including the 2nd one for
        ``use_lmfit`` (a configuration parameter) since it is not in
        ``_default_options``
        """
        fr = self.RFF.fit(rot0=79 * u.deg)
        assert isinstance(fr, rotated_frame.FitResult)
        assert fr.fitresult.success
        assert u.allclose(fr.residual, 0 * u.deg, atol=1e-6 * u.deg)

    @pytest.mark.skipif(HAS_LMFIT, reason="cannot have lmfit installed")
    def test_fit_lmfit_fail(self):
        """Test ``RotatedFrameFitter.fit`` with lmfit."""
        with pytest.raises(ImportError, match="`lmfit`"):
            self.RFF.fit(rot0=79 * u.deg, use_lmfit=True)

    @pytest.mark.skipif(not HAS_LMFIT, reason="requires lmfit")
    def test_fit_lmfit(self):
        """Test ``RotatedFrameFitter.fit`` with lmfit."""
        fr = self.RFF.fit(rot0=79 * u.deg, use_lmfit=True)
        assert isinstance(fr, rotated_frame.FitResult)
        assert fr.fitresult.success
        assert u.allclose(fr.residual, 0 * u.deg, atol=1e-6 * u.deg)

    # ---------------------------------------------------------------

    @pytest.mark.skip("TODO!")
    def test_plot_data(self):
        """Test ``RotatedFrameFitter.plot_data``."""
        assert False

    @pytest.mark.skip("TODO!")
    def test_plot_residual(self):
        """Test ``RotatedFrameFitter.plot_residual``."""
        assert False


# -------------------------------------------------------------------


@pytest.mark.parametrize(
    "test_data,variables,scalar,expected_lat",
    [
        (
            data.icrs,  # TODO fixtures
            (
                example_coords.ICRS_ROTATION,
                example_coords.RA,
                example_coords.DEC,
            ),
            True,
            0,  # TODO fixtures
        ),
        (
            data.icrs,  # TODO fixtures
            (
                example_coords.ICRS_ROTATION,
                example_coords.RA,
                example_coords.DEC,
            ),
            False,
            data.ricrs.phi2,  # TODO fixtures
        ),
        (
            data.gcentric,  # TODO fixtures
            (
                example_coords.GALACTOCENTRIC_ROTATION,
                example_coords.LON,
                example_coords.LAT,
            ),
            True,
            0,  # TODO fixtures
        ),
        (
            data.gcentric,  # TODO fixtures
            (
                example_coords.GALACTOCENTRIC_ROTATION,
                example_coords.LON,
                example_coords.LAT,
            ),
            False,
            data.rgcentric.phi2,  # TODO fixtures
        ),
        # TODO the other datasets
    ],
)
def test_residual(test_data, variables, scalar, expected_lat):
    """Test `~trackstream.preprocess.rotated_frame.residual`."""
    res = rotated_frame.residual(variables, test_data.cartesian, scalar=scalar)

    # compare result and expected latitude
    # When scalar is False, can have units, so always take value.
    np.testing.assert_allclose(
        res,
        u.Quantity(expected_lat).to_value(),
        atol=1e-12,
    )


# /def


##############################################################################
# END
