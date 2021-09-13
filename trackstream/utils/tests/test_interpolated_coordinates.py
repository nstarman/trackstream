# -*- coding: utf-8 -*-

"""Testing :mod:`~trackstream.utils.interpolated_coordinates`."""

__all__ = [
    "Test_InterpolatedRepresentationOrDifferential",
    "Test_InterpolatedRepresentation",
    "Test_InterpolatedCartesianRepresentation",
    "Test_InterpolatedDifferential",
    "Test_InterpolatedCoordinateFrame",
    "Test_InterpolatedSkyCoord",
]


##############################################################################
# IMPORTS

# STDLIB
import operator

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
import pytest

# LOCAL
from trackstream.tests.helper import BaseClassDependentTests
from trackstream.utils import generic_coordinates as gcoord
from trackstream.utils import interpolated_coordinates as icoord

##############################################################################
# TESTS
##############################################################################


def test_find_first_best_compatible_differential():
    """Test ``_find_first_best_compatible_differential``."""
    # ----------------------------
    # Test when rep has compatible differentials
    rep = coord.CartesianRepresentation(x=1, y=2, z=3)

    # find differential
    dif = icoord._find_first_best_compatible_differential(rep)

    assert dif is coord.CartesianDifferential

    # ----------------------------
    # test when it doesn't
    # rep = coord.CartesianRepresentation(x=1, y=2, z=3)
    # # find differential
    # dif = self.klass(rep)
    # assert dif is coord.CartesianDifferential


# /class


#####################################################################


def test_infer_derivative_type():
    """Test ``_infer_derivative_type``."""
    # ----------------------------
    # Test when rep is a differential

    rep = coord.CartesianDifferential(d_x=1, d_y=2, d_z=3)
    dif = icoord._infer_derivative_type(rep, u.s)

    assert dif.__name__ == "GenericCartesian2ndDifferential"
    assert dif is gcoord.GenericCartesian2ndDifferential

    # ----------------------------
    # Test when non-time dif unit

    rep = coord.CartesianRepresentation(x=1, y=2, z=3)
    dif = icoord._infer_derivative_type(rep, u.deg)

    assert dif.__name__ == "GenericCartesianDifferential"
    assert dif is gcoord.GenericCartesianDifferential

    # ----------------------------
    # Test when Rep & time

    rep = coord.CartesianRepresentation(x=1, y=2, z=3)
    dif = icoord._infer_derivative_type(rep, u.s)

    assert dif is coord.CartesianDifferential


# /class


#####################################################################


class Test_InterpolatedRepresentationOrDifferential(
    BaseClassDependentTests,
    klass=icoord.InterpolatedRepresentationOrDifferential,
):
    """Test :class:`~{package}.{klass}`."""

    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.num = 40
        cls.affine = np.linspace(0, 10, num=cls.num) * u.Myr

        cls.rep = coord.CartesianRepresentation(
            x=np.linspace(0, 1, num=cls.num) * u.kpc,
            y=np.linspace(1, 2, num=cls.num) * u.kpc,
            z=np.linspace(2, 3, num=cls.num) * u.kpc,
            differentials=coord.CartesianDifferential(
                d_x=np.linspace(3, 4, num=cls.num) * u.km / u.s,
                d_y=np.linspace(4, 5, num=cls.num) * u.km / u.s,
                d_z=np.linspace(5, 6, num=cls.num) * u.km / u.s,
            ),
        )

        if cls.klass is icoord.InterpolatedRepresentationOrDifferential:

            class SubClass(cls.klass):
                pass  # so not abstract & can instantiate

            cls.inst = SubClass(cls.rep, affine=cls.affine)

        else:
            cls.inst = cls.klass(cls.rep, affine=cls.affine)

    # /def

    #######################################################
    # Method tests

    def test___new__(self) -> None:
        """Test method ``__new__``."""
        # ------------------
        # Test it's abstract

        if self.klass is icoord.InterpolatedRepresentationOrDifferential:

            with pytest.raises(TypeError) as e:
                self.klass()

            assert "Cannot instantiate" in str(e.value)

        # ------------------
        # can instantiate subclass
        # tested in all subclasses

        else:

            self.klass(self.rep, affine=self.affine)

    # /def

    def test___init__(self) -> None:
        """Test method ``__init__``."""
        # skip if it's the baseclass.
        if self.klass is icoord.InterpolatedRepresentationOrDifferential:
            return

        # ------------------
        # affine not 1-D

        with pytest.raises(ValueError):
            self.klass(self.rep, affine=self.affine.reshape((-1, 2)))

        # ------------------
        # affine not right length

        with pytest.raises(ValueError):
            self.klass(self.rep, affine=self.affine[::2])

        # ------------------
        # when interps not None

        irep = self.klass(self.rep, affine=self.affine, interps=self.rep)

        assert irep._interps is self.rep

        # ------------------
        # the standard, need to interpolate

        irep = self.klass(self.rep, affine=self.affine)

        # ------------------
        # differentials already interpolated

        # TODO
        # irep = self.klass(rep=self.rep, affine=self.affine)

    # /def

    def test_affine(self) -> None:
        """Test method ``affine`."""
        assert all(self.inst.affine == self.affine)

        # read-only
        with pytest.raises(AttributeError):
            self.inst.affine = 2

    # /def

    def test__class_(self) -> None:
        """Test method ``_class_``."""
        assert issubclass(self.inst._class_, self.klass)

    # /def

    def test__realize_class(self) -> None:
        """Test method ``_realize_class``."""
        assert self.inst._realize_class(self.inst.data, self.affine)

    # /def

    def test___call__(self) -> None:
        """Test method ``__call__``."""
        pass  # it's abstract and empty

    # /def

    def test_derivative_type(self) -> None:
        """Test method ``derivative_type``."""
        assert issubclass(
            self.inst.derivative_type,
            coord.BaseRepresentationOrDifferential,
        )

        # ----------------
        # test setting
        old_derivative_type = self.inst.derivative_type
        self.inst.derivative_type = coord.SphericalDifferential
        # test
        assert issubclass(
            self.inst.derivative_type,
            coord.SphericalDifferential,
        )
        # reset
        self.inst.derivative_type = old_derivative_type

    # /def

    def test_clear_derivatives(self) -> None:
        """Test method ``clear_derivatives``."""
        # calculate derivatives
        self.inst.derivative(n=1)
        self.inst.derivative(n=2)

        # will fail until cache in "differentials"
        if hasattr(self.inst, "_derivatives"):  # skip differentials
            assert not any(["lambda " in self.inst._derivatives.keys()])

    # /def

    def test_derivative(self) -> None:
        """Test method ``derivative``.

        .. todo::

            tests on the Generic Coordinates

        """
        # --------------------

        ideriv = self.inst.derivative(n=1)  # a straight line

        assert all(ideriv.affine == self.affine)
        assert np.allclose(ideriv._values.view(float), 0.1)

        # --------------------

        ideriv = self.inst.derivative(n=2)  # no 2nd deriv

        assert all(ideriv.affine == self.affine)
        assert np.allclose(ideriv._values.view(float), 0.0)

        # --------------------

        ideriv = self.inst.derivative(n=3)  # no 3rd deriv

        assert all(ideriv.affine == self.affine)
        assert np.allclose(ideriv._values.view(float), 0.0)

    # /def

    def test_antiderivative(self) -> None:
        """Test method ``antiderivative``."""
        # not yet implemented!
        assert not hasattr(self.inst, "antiderivative")

    # /def

    def test___class__(self) -> None:
        """Test method ``__class__``."""
        assert self.inst.__class__ is self.inst.data.__class__

    # /def

    def test___getattr__(self) -> None:
        """Test method ``__getattr__``."""
        key = "shape"
        assert self.inst.__getattr__(key) == getattr(self.inst.data, key)

    # /def

    def test___getitem__(self) -> None:
        """Test method ``__getitem__``."""
        assert isinstance(
            self.inst[::2],
            icoord.InterpolatedRepresentationOrDifferential,
        )

    # /def

    def test___len__(self) -> None:
        """Test method ``__len__``."""
        assert self.inst.__len__() == self.inst.data.__len__() == self.num

    # /def

    def test___repr__(self) -> None:
        """Test method ``__repr__``."""
        s = self.inst.__repr__()

        assert isinstance(s, str)
        assert "lambda" in s

        # Also need to test a dimensionless case
        # This is done in InterpolatedCartesianRepresentation

    # /def

    def test__scale_operation(self) -> None:
        """Test method ``_scale_operation``."""
        with pytest.raises(TypeError) as e:
            self.inst._scale_operation(operator.mul, 1.1)

        assert "differentials are attached" in str(e.value)

        # TODO one that works

    # /def

    def test___add__(self) -> None:
        """Test method ``__add__``."""
        # -----------
        # fails

        with pytest.raises(ValueError) as e:
            self.inst.__add__(self.inst[::2])

        assert "Can only add" in str(e.value)

        # -----------
        # succeeds

        # TODO test in subclass

    # /def

    def test___sub__(self) -> None:
        """Test method ``__sub__``."""
        # -----------
        # fails

        with pytest.raises(ValueError) as e:
            self.inst.__sub__(self.inst[::2])

        assert "Can only subtract" in str(e.value)

        # -----------
        # succeeds

        # TODO test in subclass

    # /def

    def test___mul__(self) -> None:
        """Test method ``__mul__``."""
        # -----------
        # fails

        with pytest.raises(ValueError) as e:
            self.inst.__mul__(self.inst[::2])

        assert "Can only multiply" in str(e.value)

        # -----------
        # succeeds

        # TODO test in subclass

    # /def

    def test___truediv__(self) -> None:
        """Test method ``__truediv__``."""
        # -----------
        # fails

        with pytest.raises(ValueError) as e:
            self.inst.__truediv__(self.inst[::2])

        assert "Can only divide" in str(e.value)

        # -----------
        # succeeds

        # TODO test in subclass

    # /def

    def test_from_cartesian(self) -> None:
        """Test method ``from_cartesian``."""
        # -------------------
        # works

        newrep = self.inst.from_cartesian(self.rep)

        assert isinstance(newrep, self.inst.__class__)
        assert isinstance(newrep, self.inst._class_)  # interpolated class

        # -------------------
        # fails

        with pytest.raises(ValueError):
            self.inst.from_cartesian(self.rep[::2])

    # /def

    def test_to_cartesian(self) -> None:
        """Test method ``to_cartesian``."""
        # -------------------
        # works

        newrep = self.inst.to_cartesian()

        assert isinstance(newrep, coord.CartesianRepresentation)
        assert isinstance(
            newrep,
            icoord.InterpolatedRepresentationOrDifferential,
        )

    # /def

    def test_copy(self) -> None:
        """Test method ``copy``."""
        newrep = self.inst.copy()

        assert newrep is not self.inst  # not the same object
        assert isinstance(
            newrep,
            icoord.InterpolatedRepresentationOrDifferential,
        )

        # TODO more tests

    # /def

    #######################################################
    # Usage tests


# /class


#####################################################################


class Test_InterpolatedRepresentation(
    Test_InterpolatedRepresentationOrDifferential,
    klass=icoord.InterpolatedRepresentation,
):
    """Test :class:`~{package}.{klass}`."""

    #######################################################
    # Method tests

    def test___new__(self) -> None:
        """Test method ``__init__``."""
        super().test___new__()

        # ------------------
        # test it redirects

        irep = self.klass(
            self.rep.represent_as(coord.CartesianRepresentation),
            affine=self.affine,
        )
        assert isinstance(irep, self.klass)
        assert isinstance(irep, icoord.InterpolatedCartesianRepresentation)

    # /def

    def test___init__(self) -> None:
        """Test method ``__init__``."""
        super().test___init__()

        # ------------------
        # Test not instantiated

        with pytest.raises(ValueError) as e:
            self.klass(self.inst.__class__, affine=self.affine)

        assert "Must instantiate `rep`" in str(e.value)

        # ------------------
        # Test wrong type

        with pytest.raises(TypeError) as e:
            self.klass(object(), affine=self.affine)

        assert "`rep` must be" in str(e.value)

    # /def

    def test___call__(self) -> None:
        """Test method ``__call__``."""
        super().test___call__()

        rep = self.inst()
        assert isinstance(rep, coord.BaseRepresentation)
        assert not isinstance(rep, icoord.InterpolatedRepresentation)
        assert all(rep._values == self.inst.data._values)

    # /def

    def test_represent_as(self) -> None:
        """Test method ``represent_as``.

        Tested in astropy. Here only need to test it stays interpolated

        """
        # super().test_represent_as()
        rep = self.inst.represent_as(coord.PhysicsSphericalRepresentation)

        assert isinstance(rep, coord.PhysicsSphericalRepresentation)
        assert isinstance(rep, icoord.InterpolatedRepresentation)

    # /def

    def test_with_differentials(self) -> None:
        """Test method ``with_differentials``."""
        # super().test_with_differentials()

        rep = self.inst.with_differentials(
            self.rep.differentials["s"].represent_as(
                coord.CartesianDifferential,
                base=self.rep.represent_as(coord.CartesianRepresentation),
            ),
        )

        assert isinstance(rep, self.inst.__class__)
        assert isinstance(rep, self.inst._class_)

        # --------------
        # bad differential length caught by astropy!

    # /def

    def test_without_differentials(self) -> None:
        """Test method ``without_differentials``."""
        # super().test_without_differentials()

        rep = self.inst.without_differentials()

        assert isinstance(rep, self.inst.__class__)
        assert isinstance(rep, self.inst._class_)
        assert not rep.differentials  # it's empty

    # /def

    def test_clear_derivatives(self) -> None:
        """Test method ``clear_derivatives``."""
        # calculate derivatives
        self.inst.derivative(n=1)
        self.inst.derivative(n=2)

        assert "lambda 1" in self.inst._derivatives.keys()
        assert "lambda 2" in self.inst._derivatives.keys()

        self.inst.clear_derivatives()

        assert "lambda 1" not in self.inst._derivatives.keys()
        assert "lambda 2" not in self.inst._derivatives.keys()
        assert not any(["lambda " in self.inst._derivatives.keys()])

    # /def

    def test_derivative(self) -> None:
        """Test method ``derivative``."""
        super().test_derivative()

        # Testing cache, it's the only thing different between
        # InterpolatedRepresentationOrDifferential and
        # InterpolatedRepresentation
        assert "lambda 1" in self.inst._derivatives.keys()
        assert "lambda 2" in self.inst._derivatives.keys()

        assert self.inst.derivative(n=1) is self.inst._derivatives["lambda 1"]
        assert self.inst.derivative(n=2) is self.inst._derivatives["lambda 2"]

    # /def

    def test_headless_tangent_vector(self) -> None:
        """Test method ``headless_tangent_vector."""
        htv = self.inst.headless_tangent_vectors()

        assert isinstance(htv, icoord.InterpolatedRepresentation)
        assert all(htv.affine == self.inst.affine)

        # given the straight lines...
        for c in htv.components:
            assert np.allclose(getattr(htv, c), 0.1 * u.kpc)

    # /def

    def test_tangent_vector(self) -> None:
        """Test method ``headless_tangent_vector."""
        # BaseRepresentationOrDifferential derivative is not interpolated
        tv = self.inst.tangent_vectors()

        assert isinstance(tv, icoord.InterpolatedRepresentation)
        assert all(tv.affine == self.inst.affine)

        # given the straight lines...
        for c in tv.components:
            assert np.allclose(
                getattr(tv, c) - getattr(self.inst, c),
                0.1 * u.kpc,
            )

    # /def

    def test___add__(self) -> None:
        """Test method ``__add__``."""
        super().test___add__()

        # -----------
        # succeeds
        # requires stripping the differentials

        inst = self.inst.without_differentials()

        got = inst + inst
        expected = inst.data + inst.data

        # affine is the same
        assert all(got.affine == self.affine)
        # and components are the same
        for c in expected.components:
            assert all(getattr(got, c) == getattr(expected, c))

        # and a sanity check
        assert all(got.x == 2 * inst.x)

    # /def

    def test___sub__(self) -> None:
        """Test method ``__sub__``."""
        super().test___sub__()

        # -----------
        # succeeds
        # requires stripping the differentials

        inst = self.inst.without_differentials()

        got = inst - inst
        expected = inst.data - inst.data

        # affine is the same
        assert all(got.affine == self.affine)
        # and components are the same
        for c in expected.components:
            assert all(getattr(got, c) == getattr(expected, c))

    # /def

    def test___mul__(self) -> None:
        """Test method ``__mul__``."""
        super().test___mul__()

        # -----------
        # succeeds
        # requires stripping the differentials

        inst = self.inst.without_differentials()

        got = inst * 2
        expected = inst.data * 2

        # affine is the same
        assert all(got.affine == self.affine)
        # and components are the same
        for c in expected.components:
            assert all(getattr(got, c) == getattr(expected, c))

    # /def

    def test___truediv__(self) -> None:
        """Test method ``__truediv__``."""
        super().test___truediv__()

        # -----------
        # succeeds
        # requires stripping the differentials

        inst = self.inst.without_differentials()

        got = inst / 2
        expected = inst.data / 2

        # affine is the same
        assert all(got.affine == self.affine)
        # and components are the same
        for c in expected.components:
            assert all(getattr(got, c) == getattr(expected, c))

    # /def


# /class


#####################################################################


class Test_InterpolatedCartesianRepresentation(
    Test_InterpolatedRepresentation,
    klass=icoord.InterpolatedCartesianRepresentation,
):
    """Test :class:`~{package}.{klass}`."""

    #######################################################
    # Method tests

    def test___init__(self) -> None:
        """Test method ``__init__``."""
        super().test___init__()

        # ------------------
        # Test not instantiated

        with pytest.raises(ValueError) as e:
            self.klass(self.inst.__class__, affine=self.affine)

        assert "Must instantiate `rep`" in str(e.value)

        # ------------------
        # Test wrong type

        with pytest.raises(TypeError) as e:
            self.klass(object(), affine=self.affine)

        assert "`rep` must be a `CartesianRepresentation`." in str(e.value)

    # /def

    def test_transform(self) -> None:
        """Test method ``transform``.

        Astropy tests the underlying method. Only need to test that
        it is interpolated.

        """
        rep = self.inst.transform(np.eye(3))

        assert isinstance(rep, self.inst.__class__)
        assert isinstance(rep, self.inst._class_)
        assert isinstance(rep, self.klass)

    # /def

    def test___repr__(self) -> None:
        """Test method ``__repr__``."""
        super().test___repr__()

        # Also need to test a dimensionless case
        inst = self.klass(
            coord.CartesianRepresentation(
                x=[1, 2, 3, 4],
                y=[5, 6, 7, 8],
                z=[9, 10, 11, 12],
            ),
            affine=[0, 2, 4, 6],
        )

        s = inst.__repr__()
        assert "[dimensionless]" in s

    # /def


# /class


#####################################################################


class Test_InterpolatedDifferential(
    Test_InterpolatedRepresentationOrDifferential,
    klass=icoord.InterpolatedDifferential,
):
    """Test :class:`~{package}.{klass}`."""

    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.num = 40
        cls.affine = np.linspace(0, 10, num=cls.num) * u.Myr

        cls.rep = coord.CartesianDifferential(
            d_x=np.linspace(3, 4, num=cls.num) * u.km / u.s,
            d_y=np.linspace(4, 5, num=cls.num) * u.km / u.s,
            d_z=np.linspace(5, 6, num=cls.num) * u.km / u.s,
        )

        cls.inst = cls.klass(rep=cls.rep, affine=cls.affine)

    # /def

    #######################################################
    # Method tests

    def test___new__(self) -> None:
        """Test method ``__new__``."""
        super().test___new__()

        # test wrong type
        with pytest.raises(TypeError) as e:
            self.klass(object())

        assert "`rep` must be a differential type." in str(e.value)

    # /def

    def test___call__(self) -> None:
        """Test method ``__call__``."""
        super().test___call__()

        # Test it evaluates to the correct class type
        rep = self.inst()

        assert isinstance(rep, self.inst.__class__)
        assert not isinstance(rep, (self.inst._class_, self.klass))

    # /def

    def test_represent_as(self) -> None:
        """Test method ``represent_as``.

        Astropy tests the underlying method. Only need to test that
        it is interpolated.

        """
        # super().test_represent_as()

        rep = self.inst.represent_as(
            coord.PhysicsSphericalDifferential,
            base=coord.PhysicsSphericalRepresentation(
                0 * u.rad,
                0 * u.rad,
                0 * u.km,
            ),
        )

        assert isinstance(rep, coord.PhysicsSphericalDifferential)
        assert isinstance(rep, icoord.InterpolatedDifferential)
        assert isinstance(rep, self.klass)

    # /def

    def test__scale_operation(self) -> None:
        """Test method ``_scale_operation``."""
        newinst = self.inst._scale_operation(operator.mul, 1.1)

        for c in self.inst.components:
            assert all(getattr(newinst, c) == 1.1 * getattr(self.inst, c))

        # TODO one that works

    # /def

    def test_to_cartesian(self) -> None:
        """Test method ``to_cartesian``.

        On Differentials, ``to_cartesian`` returns a Representation
        https://github.com/astropy/astropy/issues/6215

        """
        # -------------------
        # works

        newrep = self.inst.to_cartesian()

        assert isinstance(newrep, coord.CartesianRepresentation)
        assert isinstance(newrep, icoord.InterpolatedCartesianRepresentation)
        assert not isinstance(newrep, icoord.InterpolatedDifferential)

    # /def


# /class


#####################################################################


class Test_InterpolatedCoordinateFrame(
    BaseClassDependentTests,
    klass=icoord.InterpolatedCoordinateFrame,
):
    """Test :class:`~{package}.{klass}`."""

    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.num = 40
        cls.affine = np.linspace(0, 10, num=cls.num) * u.Myr
        cls.frame = coord.Galactocentric

        cls.rep = coord.CartesianRepresentation(
            x=np.linspace(0, 1, num=cls.num) * u.kpc,
            y=np.linspace(1, 2, num=cls.num) * u.kpc,
            z=np.linspace(2, 3, num=cls.num) * u.kpc,
            differentials=coord.CartesianDifferential(
                d_x=np.linspace(3, 4, num=cls.num) * u.km / u.s,
                d_y=np.linspace(4, 5, num=cls.num) * u.km / u.s,
                d_z=np.linspace(5, 6, num=cls.num) * u.km / u.s,
            ),
        )

        cls.irep = icoord.InterpolatedRepresentation(
            cls.rep,
            affine=cls.affine,
        )

        cls.inst = cls.klass(cls.frame(cls.irep))

    # /def

    #######################################################
    # Method Tests

    def test___init__(self) -> None:
        """Test method ``__init__``."""
        # -------------------
        # rep is interpolated

        c = self.klass(self.frame(self.irep))

        assert isinstance(c, self.klass)
        assert isinstance(c.data, icoord.InterpolatedRepresentation)

        # -------------------
        # rep is base astropy

        # doesn't work b/c no affine
        with pytest.raises(ValueError) as e:
            self.klass(self.frame(self.rep), affine=None)

        assert "`data` is not already interpolated." in str(e.value)

        # ----

        # works with affine
        c = self.klass(self.frame(self.rep), affine=self.affine)

        assert isinstance(c, self.klass)
        assert isinstance(c.data, icoord.InterpolatedRepresentation)

        # -------------------
        # rep is wrong type

        class Obj:
            data = object()

        with pytest.raises(TypeError) as e:
            self.klass(Obj())

        assert "`data` must be type " in str(e.value)

    # /def

    def test__interp_kwargs(self) -> None:
        """Test method ``_interp_kwargs``."""
        # property get
        assert self.inst._interp_kwargs is self.inst.data._interp_kwargs

        # setter
        self.inst._interp_kwargs = {"a": 1}
        assert self.inst.data._interp_kwargs["a"] == 1
        self.inst._interp_kwargs = {}  # reset

    # /def

    def test___call__(self) -> None:
        """Test method ``__call__``.

        Since it returns a BaseCoordinateFrame, and does the evaluation
        through the InterpolatedRepresentation, all we need to test here
        is that it's the right type.

        """
        data = self.inst()

        assert isinstance(data, self.frame)
        assert len(data) == self.num

    # /def

    def test__class_(self) -> None:
        """Test method ``_class_``."""
        assert issubclass(self.inst._class_, self.klass)

    # /def

    @pytest.mark.skip("TODO")
    def test__realize_class(self) -> None:
        """Test method ``_realize_class``."""
        assert False

    # /def

    @pytest.mark.skip("TODO")
    def test_realize_frame(self) -> None:
        """Test method ``realize_frame``."""
        assert False

    # /def

    def test_derivative(self) -> None:
        """Test method ``derivative``.

        Just passes to the Representation.

        """
        # --------------------

        ideriv = self.inst.derivative(n=1)  # a straight line

        assert all(ideriv.affine == self.affine)
        assert np.allclose(ideriv._values.view(float), 0.1)

        # --------------------

        ideriv = self.inst.derivative(n=2)  # no 2nd deriv

        assert all(ideriv.affine == self.affine)
        assert np.allclose(ideriv._values.view(float), 0.0)

        # --------------------

        ideriv = self.inst.derivative(n=3)  # no 3rd deriv

        assert all(ideriv.affine == self.affine)
        assert np.allclose(ideriv._values.view(float), 0.0)

    # /def

    def test_affine(self) -> None:
        """Test method ``affine``.

        Just passes to the Representation.

        """
        assert all(self.inst.affine == self.affine)
        assert all(self.inst.frame.data.affine == self.affine)

    # /def

    def test_headless_tangent_vectors(self) -> None:
        """Test method ``headless_tangent_vectors``.

        Wraps Representation in InterpolatedCoordinateFrame

        """
        htv = self.inst.headless_tangent_vectors()

        assert isinstance(htv, self.klass)  # interp
        assert isinstance(htv, coord.BaseCoordinateFrame)

        for c in htv.data.components:
            assert np.allclose(getattr(htv.data, c), 0.1 * u.kpc)

    # /def

    def test_tangent_vectors(self) -> None:
        """Test method ``tangent_vectors``.

        Wraps Representation in InterpolatedCoordinateFrame

        """
        tv = self.inst.tangent_vectors()

        assert isinstance(tv, self.klass)  # interp
        assert isinstance(tv, coord.BaseCoordinateFrame)

        for c in tv.data.components:
            assert np.allclose(
                getattr(tv.data, c) - getattr(self.inst.data, c),
                0.1 * u.kpc,
            )

    # /def

    def test___class__(self) -> None:
        """Test method ``__class__``.

        Just passes to the CoordinateFrame.

        """
        assert self.inst.__class__ is self.inst.frame.__class__
        assert issubclass(self.inst.__class__, coord.BaseCoordinateFrame)
        assert isinstance(self.inst, self.klass)

    # /def

    def test___getattr__(self) -> None:
        """Test method ``__getattr__``.

        Routes everything to underlying CoordinateFrame.
        Lets just test the ``shape``.

        """
        assert self.inst.shape == self.inst.frame.shape
        assert self.inst.shape == (self.num,)

        assert self.inst.ndim == self.inst.frame.ndim
        assert self.inst.ndim == 1

    # /def

    def test___len__(self) -> None:
        """Test method ``__len__``."""
        assert len(self.inst) == self.num

    # /def

    def test___getitem__(self) -> None:
        """Test method ``__getitem__``."""
        # Test has problem when slicing with <3 elements
        # TODO? fix?
        with pytest.raises(Exception):
            self.inst[:3]

        # works otherwise
        inst = self.inst[:4]

        assert isinstance(inst, coord.BaseCoordinateFrame)
        assert isinstance(inst, self.klass)
        assert isinstance(inst, self.inst.__class__)
        assert isinstance(inst, self.inst._class_)

        assert inst.representation_type == self.inst.representation_type

        assert len(inst) == 4

    # /def

    def test_transform_to(self) -> None:
        """Test method ``transform_to``.

        All the transformation is handled in the frame. Only need to
        test that it's still interpolated.

        """
        newinst = self.inst.transform_to(coord.HeliocentricTrueEcliptic())

        assert isinstance(newinst, coord.HeliocentricTrueEcliptic)
        assert isinstance(newinst, icoord.InterpolatedCoordinateFrame)

        assert isinstance(newinst.frame, coord.HeliocentricTrueEcliptic)

        assert isinstance(newinst.frame.data, coord.CartesianRepresentation)
        assert isinstance(
            newinst.frame.data,
            icoord.InterpolatedRepresentation,
        )

        assert isinstance(
            newinst.frame.data.data,
            coord.CartesianRepresentation,
        )

    # /def

    def test_copy(self) -> None:
        """Test method ``copy``."""
        newrep = self.inst.copy()

        assert newrep is not self.inst  # not the same object
        assert isinstance(newrep, icoord.InterpolatedCoordinateFrame)

        # TODO more tests

    # /def

    def test__frame_attrs_repr(self) -> None:
        """Test method ``_frame_attrs_repr``."""
        assert (
            self.inst._frame_attrs_repr()
            == self.inst.frame._frame_attrs_repr()
        )
        # TODO more tests

    # /def

    def test__data_repr(self) -> None:
        """Test method ``_data_repr``."""
        data_repr = self.inst._data_repr()
        assert isinstance(data_repr, str)
        # TODO more tests

    # /def

    def test___repr__(self) -> None:
        """Test method ``__repr__``."""
        s = self.inst.__repr__()
        assert isinstance(s, str)

        # a test for unit dif types
        self.inst.representation_type = coord.UnitSphericalRepresentation
        s = self.inst.__repr__()
        assert isinstance(s, str)

        # TODO more tests

    # /def


# /class


#####################################################################


class Test_InterpolatedSkyCoord(
    BaseClassDependentTests,
    klass=icoord.InterpolatedSkyCoord,
):
    """Test :class:`~{package}.{klass}`."""

    @classmethod
    def setup_class(cls):
        """Setup fixtures for testing."""
        cls.num = 40
        cls.affine = np.linspace(0, 10, num=cls.num) * u.Myr
        cls.frame = coord.Galactocentric

        cls.rep = coord.CartesianRepresentation(
            x=np.linspace(0, 1, num=cls.num) * u.kpc,
            y=np.linspace(1, 2, num=cls.num) * u.kpc,
            z=np.linspace(2, 3, num=cls.num) * u.kpc,
            differentials=coord.CartesianDifferential(
                d_x=np.linspace(3, 4, num=cls.num) * u.km / u.s,
                d_y=np.linspace(4, 5, num=cls.num) * u.km / u.s,
                d_z=np.linspace(5, 6, num=cls.num) * u.km / u.s,
            ),
        )

        cls.irep = icoord.InterpolatedRepresentation(
            cls.rep,
            affine=cls.affine,
        )

        cls.coord = cls.frame(cls.irep)
        cls.icoord = icoord.InterpolatedCoordinateFrame(cls.coord)

        cls.inst = cls.klass(cls.icoord)

    # /def

    #######################################################
    # Method Tests

    def _test_isc(
        self,
        isc,
        representation_type=coord.UnitSphericalRepresentation,
    ) -> None:
        """Runs through all the levels, testing type."""
        inst = isc.transform_to(coord.FK5())

        assert isinstance(inst, coord.SkyCoord)
        assert isinstance(inst, icoord.InterpolatedSkyCoord)

        assert isinstance(inst.frame, coord.FK5)

        assert isinstance(inst.frame.data, representation_type)
        assert isinstance(inst.frame.data, icoord.InterpolatedRepresentation)

        assert isinstance(inst.frame.data.data, representation_type)

    def test___init__(self) -> None:
        """Test method ``__init__``.

        Copying from astropy docs

        """
        # -----------
        c = icoord.InterpolatedSkyCoord(
            [10] * self.num,
            [20] * self.num,
            unit="deg",
            affine=self.affine,
        )  # defaults to ICRS frame
        self._test_isc(c)

        # -----------
        c = icoord.InterpolatedSkyCoord(
            [1, 2, 3, 4],
            [-30, 45, 8, 16],
            frame="icrs",
            unit="deg",
            affine=self.affine[:4],
        )  # 4 coords
        self._test_isc(c)

        # -----------
        coords = [
            "1:12:43.2 +31:12:43",
            "1:12:43.2 +31:12:43",
            "1:12:43.2 +31:12:43",
            "1 12 43.2 +31 12 43",
        ]
        c = icoord.InterpolatedSkyCoord(
            coords,
            frame=coord.FK4,
            unit=(u.hourangle, u.deg),
            obstime="J1992.21",
            affine=self.affine[:4],
        )
        self._test_isc(c)

        # -----------
        c = icoord.InterpolatedSkyCoord(
            ["1h12m43.2s +1d12m43s"] * self.num,
            frame=coord.Galactic,
            affine=self.affine,
        )  # Units from string
        self._test_isc(c)

        # # -----------
        c = icoord.InterpolatedSkyCoord(
            frame="galactic",
            l=["1h12m43.2s"] * self.num,
            b="+1d12m43s",  # NOT BROADCASTING THIS ONE
            affine=self.affine,
        )
        self._test_isc(c)

        # -----------
        ra = coord.Longitude([1, 2, 3, 4], unit=u.deg)  # Could also use Angle
        dec = np.array([4.5, 5.2, 6.3, 7.4]) * u.deg  # Astropy Quantity
        c = icoord.InterpolatedSkyCoord(
            ra,
            dec,
            frame="icrs",
            affine=self.affine[:4],
        )
        self._test_isc(c)

        # -----------
        c = icoord.InterpolatedSkyCoord(
            frame=coord.ICRS,
            ra=ra,
            dec=dec,
            obstime="2001-01-02T12:34:56",
            affine=self.affine[:4],
        )
        self._test_isc(c)

        # -----------
        c = coord.FK4(
            [1] * self.num * u.deg,
            2 * u.deg,
        )  # Uses defaults for obstime, equinox
        c = icoord.InterpolatedSkyCoord(
            c,
            obstime="J2010.11",
            equinox="B1965",
            affine=self.affine,
        )  # Override defaults
        self._test_isc(c)

        # -----------
        c = icoord.InterpolatedSkyCoord(
            w=[0] * self.num,
            u=1,
            v=2,
            unit="kpc",
            frame="galactic",
            representation_type="cartesian",
            affine=self.affine,
        )
        self._test_isc(c, representation_type=coord.CartesianRepresentation)

        # -----------
        c = icoord.InterpolatedSkyCoord(
            [
                coord.ICRS(ra=1 * u.deg, dec=2 * u.deg),
                coord.ICRS(ra=3 * u.deg, dec=4 * u.deg),
            ]
            * (self.num // 2),
            affine=self.affine,
        )
        self._test_isc(c)

    # /def

    def test___call__(self) -> None:
        """Test method ``__call__``."""
        inst = self.inst()

        assert isinstance(inst, coord.SkyCoord)
        assert len(inst) == self.num

    # /def

    def test_transform_to(self) -> None:
        """Test method ``transform_to``."""
        for frame in (coord.ICRS,):
            inst = self.inst.transform_to(frame())

            assert isinstance(inst, coord.SkyCoord)
            assert isinstance(inst, self.klass)

            assert isinstance(inst.frame, frame)

            assert all(inst.affine == self.affine)

    # /def

    def test_separation(self) -> None:
        """Test method ``separation``."""
        pass  # it just calls super b/c docstring issues

    # /def

    def test_separation_3d(self) -> None:
        """Test method ``separation_3d``."""
        pass  # it just calls super b/c docstring issues

    # /def

    def test_match_to_catalog_sky(self) -> None:
        """Test method ``match_to_catalog_sky``."""
        pass  # it just calls super b/c docstring issues

    # /def

    def test_match_to_catalog_3d(self) -> None:
        """Test method ``match_to_catalog_3d``."""
        pass  # it just calls super b/c docstring issues

    # /def

    def test_search_around_sky(self) -> None:
        """Test method ``search_around_sky``."""
        pass  # it just calls super b/c docstring issues

    # /def

    def test_search_around_3d(self) -> None:
        """Test method ``search_around_3d``."""
        pass  # it just calls super b/c docstring issues

    # /def


# /class


# -------------------------------------------------------------------


#####################################################################
# Tests for embedding an InterpolatedX in something


@pytest.mark.skip("TODO")
def test_InterpolatedRepresentation_in_CoordinateFrame():
    assert False


# /def


@pytest.mark.skip("TODO")
def test_InterpolatedCoordinateFrame_in_SkyCoord():
    assert False


# /def


@pytest.mark.skip("TODO")
def test_InterpolatedRepresentation_in_CoordinateFrame_in_SkyCoord():
    assert False


# /def


##############################################################################
# END
