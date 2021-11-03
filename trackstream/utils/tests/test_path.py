# -*- coding: utf-8 -*-

"""Testing :mod:`trackstream.utils.path`."""


##############################################################################
# IMPORTS

# THIRD PARTY
import pytest

# LOCAL
from trackstream.utils.path import Path

##############################################################################
# TESTS
##############################################################################


class TestPath:
    """Test :class:`trackstream.utils.path.Path`."""

    @pytest.fixture
    def num(self):
        return 40

    @pytest.fixture
    def affine(self, num):
        return np.linspace(0, 10, num=num) * u.Myr

    @pytest.fixture
    def frame(self):
        return coord.ICRS

    @pytest.fixture
    def iscrd(self, affine, num):
        diff = coord.CartesianDifferential(
            d_x=np.linspace(3, 4, num=num) * (u.km / u.s),
            d_y=np.linspace(4, 5, num=num) * (u.km / u.s),
            d_z=np.linspace(5, 6, num=num) * (u.km / u.s),
        )
        rep = rep_cls(
            x=np.linspace(0, 1, num=num) * u.kpc,
            y=np.linspace(1, 2, num=num) * u.kpc,
            z=np.linspace(2, 3, num=num) * u.kpc,
            differentials=dif,
        )
        irep = icoord.InterpolatedRepresentation(rep, affine=affine)
        icrd = icoord.InterpolatedCoordinateFrame(frame(irep))
        return InterpolatedSkyCoord(icrd)

    @pytest.fixture
    def width(self):
        return 100 * u.pc  # TODO!

    @pytest.fixture
    def path_cls(self):
        return Path

    @pytest.fixture
    def path(self, path_cls, affine, frame):
        return path_cls(self.path, self.width, name="TestPath", affine=affine, frame=frame)

    # ===============================================================
    # Method tests

    def test_init(self, path_cls, iscrd, width, affine, frame):
        """Test initialization."""
        path = path_cls(iscrd, width, name="test_init", affine=affine, frame=frame)
        assert hasattr(path, "_name")
        assert hasattr(path, "_frame")
        assert hasattr(path, "_original_path")
        assert hasattr(path, "_path")

        # TODO! tests for initialize width

    def test_name(self, path):
        """Test :attr:`trackstream.utils.path.Path.name`."""
        assert path.name is path._name

    def test_frame(self, path):
        """Test :attr:`trackstream.utils.path.Path.frame`."""
        assert path.frame is path._frame

    def test_path(self, path):
        """Test :attr:`trackstream.utils.path.Path.path`."""
        assert path.path is path._path

    def test_affine(self, path, affine):
        """Test :attr:`trackstream.utils.path.Path.affine`."""
        assert path.affine is path.path.affine
        assert np.all(path.affine == affine)

    def test_width(self, path, affine):
        """Test :meth:`trackstream.utils.path.Path.width`."""
        # default
        assert np.equal(path.width(None) == path.width(affine))

        # TODO! scalar eval

    @pytest.mark.skip("TODO!")
    def test_call(self, path, affine):
        path.width(affine)

        # TODO! scalar


##############################################################################
# END
