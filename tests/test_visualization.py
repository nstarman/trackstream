"""Initiation Tests for :mod:`trackstream.base`."""

from __future__ import annotations

# THIRD PARTY
import astropy.units as u
import matplotlib.pyplot as plt
import pytest
from astropy.coordinates import (
    ICRS,
    Galactocentric,
    SkyCoord,
    SphericalDifferential,
    SphericalRepresentation,
)
from astropy.units import Quantity
from numpy import array_equal

# LOCAL
from trackstream.stream.base import StreamLike
from trackstream.stream.visualization import StreamPlotDescriptorBase
from trackstream.utils.visualization import PlotDescriptorBase

##############################################################################
# PARAMETERS


class StreamLikeExample:
    @property
    def full_name(self):
        return "name"

    @property
    def coords_ord(self):
        return SkyCoord(
            ra=Quantity(1, u.deg),
            dec=Quantity(2, u.deg),
            distance=Quantity(3, u.kpc),
            pm_ra_cosdec=Quantity(1, "mas/yr"),
            pm_dec=Quantity(2, "mas/yr"),
            radial_velocity=Quantity(3, "km/s"),
        )

    @property
    def system_frame(self):
        return self.coords_ord.frame


##############################################################################
# TESTS
##############################################################################


@pytest.mark.skip("TODO!")
def test_StreamLike():
    """Test :class:`trackstream.visualization.StreamLike`."""
    assert isinstance(StreamLikeExample(), StreamLike)


##############################################################################


class Test_PlotDescriptorBase:
    """Test `trackstream.visualization.PlotDescriptorBase`."""

    @pytest.fixture(scope="class")
    def descriptor_cls(self):
        return PlotDescriptorBase

    @pytest.fixture()
    def fig(self):
        return plt.Figure()

    @pytest.fixture()
    def axs(self, fig):
        return fig.subplots(nrows=1, ncols=1)

    # ===============================================================
    # Method Tests

    @pytest.mark.skip("TODO!")
    def test_expected_attributes(self, descriptor):
        """Test the desciptor expects the right attributes."""
        annot = descriptor.__annotations__
        assert annot["default_scatter_kwargs"] == "Dict[str, Any]"

    def test_init(self, descriptor_cls):
        """Test the descriptor initialization."""

        # default_scatter_kwargs
        descriptor = descriptor_cls(default_scatter_kwargs={})
        scatter_style = descriptor.default_scatter_kwargs
        assert scatter_style["s"] == 3

        descriptor = descriptor_cls(default_scatter_kwargs={"s": 4})
        assert descriptor.default_scatter_kwargs["s"] == 4

    @pytest.mark.skip("TODO!")
    def test_get_kw(self, descriptor):
        """Test method ``_get_kw``."""
        # No arguments
        kw = descriptor._get_kw()
        assert kw == descriptor.default_scatter_kwargs

        # With kwargs
        kw = descriptor._get_kw({"s": 0})
        assert kw["s"] == 0

        # With defaults
        kw = descriptor._get_kw(s=0)
        assert kw["s"] == 0

    @pytest.mark.skip("TODO!")
    def test_setup(self, descriptor, axs):
        """Test method ``_setup``."""
        # Process and check axes
        parent, _ax, *_ = descriptor._setup(ax=axs)
        assert parent is descriptor._enclosing
        assert axs is _ax


class Test_StreamPlotDescriptorBase(Test_PlotDescriptorBase):
    """Test `trackstream.visualization.StreamPlotDescriptorBase`."""

    @pytest.fixture(scope="class")
    def descriptor_cls(self):
        return StreamPlotDescriptorBase

    @pytest.fixture(scope="class")
    def enclosing_cls(self, descriptor_cls):
        """Fixture of the enclosing class."""

        class Enclosing(StreamLikeExample):
            """Enclosing Stream-like."""

            attr = descriptor_cls["Enclosing"]()

        return Enclosing

    @pytest.fixture(scope="class")
    def coords(self, enclosing):
        """Fixture of the enclosing object's coordinates."""
        return enclosing.coords_ord

    # ===============================================================
    # Method Tests

    def test_init(self, descriptor_cls):
        """Test the descriptor initialization."""
        super().test_init(descriptor_cls)

        # default_scatter_kwargs
        descriptor = descriptor_cls(default_scatter_kwargs={})
        scatter_style = descriptor.default_scatter_kwargs
        assert scatter_style["s"] == 3
        assert scatter_style["marker"] == "*"

        descriptor = descriptor_cls(default_scatter_kwargs={"s": 4, "marker": "."})
        scatter_style = descriptor.default_scatter_kwargs
        assert scatter_style["s"] == 4
        assert scatter_style["marker"] == "."

    # ---------------------------------------------------------------
    # Private methods

    @pytest.mark.skip("TODO!")
    def test_parse_frame(self, descriptor, enclosing, frame):
        """Test method ``_parse_frame``."""
        # Error
        with pytest.raises(ValueError, match="<class 'object'> is not"):
            descriptor._parse_frame(object)

        # BaseCoordinateFrame
        frame, name = descriptor._parse_frame(frame)
        assert frame is frame
        assert name == frame.__class__.__name__

        # 'stream'
        frame, name = descriptor._parse_frame("stream")
        assert frame == enclosing.system_frame
        assert name == "Stream"

        # str
        frame, name = descriptor._parse_frame("Galactocentric")
        assert isinstance(frame, Galactocentric)
        assert name == "Galactocentric"

    @pytest.mark.skip("TODO!")
    def test_to_frame(self, descriptor, coords):
        """Test method ``to_frame``."""
        # Frame is None
        c, name = descriptor._to_frame(coords, frame=None)

        assert name.lower() == coords.frame.name
        assert c == coords.frame

        c, name = descriptor._to_frame(coords, frame="stream")

        assert name == "Stream"
        assert c == coords.frame
        assert c.representation_type == SphericalRepresentation
        assert c.differential_type == SphericalDifferential

        c, name = descriptor._to_frame(coords, frame="ICRS")

        assert name == "ICRS"
        assert c == coords.icrs.frame
        assert c.representation_type == SphericalRepresentation
        assert c.differential_type == SphericalDifferential

    @pytest.mark.skip("TODO!")
    def test_get_xy_names(self, descriptor):
        """Test method ``_get_xy_name``."""
        with pytest.raises(ValueError, match="kind"):
            descriptor._get_xy_names(frame=ICRS(), kind="other")

        assert descriptor._get_xy_names(frame=ICRS(), kind="positions") == ("ra", "dec")
        assert descriptor._get_xy_names(frame=ICRS(), kind="kinematics") == (
            "pm_ra_cosdec",
            "pm_dec",
        )

        assert descriptor._get_xy_names(frame=Galactocentric(), kind="positions") == ("x", "y")
        assert descriptor._get_xy_names(frame=Galactocentric(), kind="kinematics") == ("v_x", "v_y")

    @pytest.mark.skip("TODO!")
    def test_get_xy(self, descriptor, coords):
        """Test method ``_get_xy``."""
        with pytest.raises(ValueError, match="kind"):
            descriptor._get_xy(coords, kind="other")

        # ICRS
        c = coords.transform_to(ICRS())
        (x, xn), (y, yn) = descriptor._get_xy(c, kind="positions")
        assert array_equal(x, c.ra)
        assert xn == "ra"
        assert array_equal(y, c.dec)
        assert yn == "dec"

        (x, xn), (y, yn) = descriptor._get_xy(c, kind="kinematics")
        assert array_equal(x, c.pm_ra)
        assert xn == "pm_ra"
        assert array_equal(y, c.pm_dec)
        assert yn == "pm_dec"

        # Galactocentric
        c = coords.transform_to(Galactocentric())
        (x, xn), (y, yn) = descriptor._get_xy(c, kind="positions")
        assert array_equal(x, c.x)
        assert xn == "x"
        assert array_equal(y, c.y)
        assert yn == "y"

        (x, xn), (y, yn) = descriptor._get_xy(c, kind="kinematics")
        assert array_equal(x, c.v_x)
        assert xn == "v_x"
        assert array_equal(y, c.v_y)
        assert yn == "v_y"

    @pytest.mark.skip("TODO!")
    def test_format_ax(self, descriptor, axs, coords):
        """Test method ``_format_ax``."""
        c = coords.transform_to(ICRS())
        axs.scatter(c.ra, c.dec)

        descriptor._format_ax(axs, frame="ICRS", x="ra", y="dec")

        assert axs.get_xlabel() == "RA (ICRS) [deg]"
        assert axs.get_ylabel() == "Dec (ICRS) [deg]"

        # TODO! compare the hash of the plot
        pytest.skip("TODO!")

    # ---------------------------------------------------------------

    @pytest.mark.skip("TODO!")
    def test_in_frame(self, descriptor, axs):
        """Test ``in_frame``."""

    @pytest.mark.skip("TODO!")
    def test_origin(self, descriptor, axs):
        """Test ``origin``."""
