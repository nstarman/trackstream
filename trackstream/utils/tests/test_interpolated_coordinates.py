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
    # dif = icrd_cls(rep)
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


class InterpolatedCoordinatesBase:
    
    @pytest.fixture
    def num(self):
        return 40

    @pytest.fixture
    def affine(self, num):
        return np.linspace(0, 10, num=num) * u.Myr

    @pytest.fixture
    def dif_cls(self):
        return coord.CartesianDifferential

    @pytest.fixture
    def dif(self, num):
        diff = coord.CartesianDifferential(
            d_x=np.linspace(3, 4, num=num) * (u.km / u.s),
            d_y=np.linspace(4, 5, num=num) * (u.km / u.s),
            d_z=np.linspace(5, 6, num=num) * (u.km / u.s),
        )
        return diff

    @pytest.fixture
    def rep_cls(self):
        return coord.CartesianRepresentation

    @pytest.fixture
    def rep(self, rep_cls, num, dif):
        rep = rep_cls(
            x=np.linspace(0, 1, num=num) * u.kpc,
            y=np.linspace(1, 2, num=num) * u.kpc,
            z=np.linspace(2, 3, num=num) * u.kpc,
            differentials=dif
        )
        return rep

    @pytest.fixture
    def rep_nodif(self, rep):
        return rep.without_differentials()


#####################################################################


class Test_InterpolatedRepresentationOrDifferential(InterpolatedCoordinatesBase):
    """Test InterpolatedRepresentationOrDifferential."""

    @pytest.fixture
    def irep_cls(self):
        return icoord.InterpolatedRepresentationOrDifferential

    @pytest.fixture
    def irep(self, irep_cls, rep, affine):
        class SubClass(irep_cls):  # so not abstract & can instantiate
            # TODO! not use Cartesian `rep`, whic is a special case
            def _scale_operation(self, op, *args):
                rep = self.data._scale_operation(op, *args)
                return self._realize_class(rep, self.affine)

        inst = SubClass(rep, affine=affine)

        return inst

    # TODO? have idif & idif_cls at this level?

    # ===============================================================
    # Method tests

    def test_new(self, irep_cls, rep, affine) -> None:
        """Test method ``__new__``."""
        # Test it's abstract
        with pytest.raises(TypeError, match="Cannot instantiate"):
            irep_cls(rep, affine=affine)

    def test_init(self, irep_cls, rep, affine) -> None:
        """Test method ``__init__``."""
        # skip if it's the baseclass.
        if irep_cls is icoord.InterpolatedRepresentationOrDifferential:
            return

        # ------------------
        # affine not 1-D

        with pytest.raises(ValueError):
            irep_cls(rep, affine=affine.reshape((-1, 2)))

        # ------------------
        # affine not right length

        with pytest.raises(ValueError):
            irep_cls(rep, affine=affine[::2])

        # ------------------
        # when interps not None

        irep = irep_cls(rep, affine=affine, interps=rep)

        assert irep._interps is rep

        # ------------------
        # the standard, need to interpolate

        irep = irep_cls(rep, affine=affine)

        # ------------------
        # differentials already interpolated

        # TODO
        # irep = irep_cls(rep=rep, affine=affine)

    def test_affine(self, irep, affine) -> None:
        """Test method ``affine`."""
        assert np.all(irep.affine == affine)

        # read-only
        with pytest.raises(AttributeError):
            irep.affine = 2

    def test__class_(self, irep_cls, irep) -> None:
        """Test method ``_class_``."""
        assert issubclass(irep._class_, irep_cls)

    def test__realize_class(self, irep, affine) -> None:
        """Test method ``_realize_class``."""
        assert irep._realize_class(irep.data, affine=affine)

    def test_call(self, irep) -> None:
        """Test method ``__call__``."""
        pass  # it's abstract and empty

    def test_derivative_type(self, irep) -> None:
        """Test method ``derivative_type``."""
        assert issubclass(
            irep.derivative_type,
            coord.BaseRepresentationOrDifferential,
        )

        # ----------------
        # test setting
        old_derivative_type = irep.derivative_type
        irep.derivative_type = coord.SphericalDifferential
        # test
        assert issubclass(
            irep.derivative_type,
            coord.SphericalDifferential,
        )
        # reset
        irep.derivative_type = old_derivative_type

    def test_clear_derivatives(self, irep) -> None:
        """Test method ``clear_derivatives``."""
        # calculate derivatives
        irep.derivative(n=1)
        irep.derivative(n=2)

        # will fail until cache in "differentials"
        if hasattr(irep, "_derivatives"):  # skip differentials
            assert not any(["lambda " in irep._derivatives.keys()])

    def test_derivative(self, irep, affine) -> None:
        """Test method ``derivative``.

        .. todo::

            tests on the Generic Coordinates

        """
        # --------------------

        ideriv = irep.derivative(n=1)  # a straight line

        assert all(ideriv.affine == affine)
        assert np.allclose(ideriv._values.view(float), 0.1)

        # --------------------

        ideriv = irep.derivative(n=2)  # no 2nd deriv

        assert all(ideriv.affine == affine)
        assert np.allclose(ideriv._values.view(float), 0.0)

        # --------------------

        ideriv = irep.derivative(n=3)  # no 3rd deriv

        assert all(ideriv.affine == affine)
        assert np.allclose(ideriv._values.view(float), 0.0)

    def test_antiderivative(self, irep) -> None:
        """Test method ``antiderivative``."""
        # not yet implemented!
        assert not hasattr(irep, "antiderivative")

    def test___class__(self, irep) -> None:
        """
        Test method ``__class__``, which is overridden to match the
        non-intperpolated data.
        """
        assert irep.__class__ is irep.data.__class__

    def test___getattr__(self, irep) -> None:
        """Test method ``__getattr__``."""
        key = "shape"
        assert irep.__getattr__(key) == getattr(irep.data, key)

    def test___getitem__(self, irep_cls, irep) -> None:
        """Test method ``__getitem__``."""
        assert isinstance(irep[::2], irep_cls)

    def test_len(self, irep, num) -> None:
        """Test method ``__len__``."""
        assert len(irep) == len(irep.data)
        assert len(irep) == num

    def test___repr__(self, irep) -> None:
        """Test method ``__repr__``."""
        s = repr(irep)

        assert isinstance(s, str)
        assert "lambda" in s

        # Also need to test a dimensionless case
        # This is done in InterpolatedCartesianRepresentation

    # ===========================================
    # Math Operations

    def test__scale_operation(self, irep) -> None:
        """Test method ``_scale_operation``."""
        newrep = irep._scale_operation(operator.mul, 1.1)

        # comparisons are wonky when differentials are attached
        assert np.all(newrep.data.without_differentials() == 1.1 * irep.data.without_differentials())
        assert np.all(newrep.data.differentials["s"].data == 1.1 * irep.data.differentials["s"].data)

    def test___add__(self, irep) -> None:
        """Test method ``__add__``."""
        # -----------
        # fails

        with pytest.raises(ValueError, match="Can only add"):
            irep.__add__(irep[::2])

        # -----------
        # succeeds

        # TODO test in subclass

    def test___sub__(self, irep) -> None:
        """Test method ``__sub__``."""
        # -----------
        # fails

        with pytest.raises(ValueError, match="Can only subtract"):
            irep.__sub__(irep[::2])

        # -----------
        # succeeds

        # TODO test in subclass

    def test___mul__(self, irep) -> None:
        """Test method ``__mul__``."""
        # -----------
        # fails

        with pytest.raises(ValueError, match="Can only multiply"):
            irep.__mul__(irep[::2])

        # -----------
        # succeeds

        # TODO test in subclass

    def test___truediv__(self, irep) -> None:
        """Test method ``__truediv__``."""
        # -----------
        # fails

        with pytest.raises(ValueError, match="Can only divide"):
            irep.__truediv__(irep[::2])

        # -----------
        # succeeds

        # TODO test in subclass

    def test_from_cartesian(self, irep, rep) -> None:
        """Test method ``from_cartesian``."""
        # -------------------
        # works

        newrep = irep.from_cartesian(rep)

        assert isinstance(newrep, irep.__class__)
        assert isinstance(newrep, irep._class_)  # interpolated class

        # -------------------
        # fails

        with pytest.raises(ValueError):
            irep.from_cartesian(rep[::2])

    def test_to_cartesian(self, irep) -> None:
        """Test method ``to_cartesian``."""
        # -------------------
        # works

        newrep = irep.to_cartesian()

        assert isinstance(newrep, coord.CartesianRepresentation)
        assert isinstance(newrep, icoord.InterpolatedRepresentationOrDifferential)

    def test_copy(self, irep) -> None:
        """Test method ``copy``."""
        newrep = irep.copy()

        assert newrep is not irep  # not the same object
        assert isinstance(newrep, icoord.InterpolatedRepresentationOrDifferential)

        # TODO more tests

    # ===============================================================
    # Usage tests


#####################################################################


class Test_InterpolatedRepresentation(Test_InterpolatedRepresentationOrDifferential):
    """Test InterpolatedRepresentation."""

    @pytest.fixture
    def irep_cls(self):
        return icoord.InterpolatedRepresentation

    @pytest.fixture
    def irep(self, irep_cls, rep, affine):
        return irep_cls(rep, affine=affine)

    #######################################################
    # Method tests

    def test_new(self, irep_cls, rep, affine) -> None:
        """Test method ``__init__``."""
        # super().test_new(irep_cls=irep_cls, rep=rep, affine=affine)  # not abstract

        # test it redirects
        irep = irep_cls(
            rep.represent_as(coord.CartesianRepresentation),
            affine=affine,
        )
        assert isinstance(irep, irep_cls)
        assert isinstance(irep, icoord.InterpolatedCartesianRepresentation)

    def test_init(self, irep_cls, rep, affine) -> None:
        """Test method ``__init__``."""
        super().test_init(irep_cls=irep_cls, rep=rep, affine=affine)

        # Test not instantiated
        with pytest.raises(ValueError, match="Must instantiate `rep`"):
            irep_cls(rep.__class__, affine=affine)

        # Test wrong type
        with pytest.raises(TypeError, match="`rep` must be"):
            irep_cls(object(), affine=affine)

    def test_call(self, irep) -> None:
        """Test method ``__call__``."""
        super().test_call(irep)

        got = irep()
        assert isinstance(got, coord.BaseRepresentation)
        assert not isinstance(got, icoord.InterpolatedRepresentation)
        assert all(got._values == irep.data._values)

    def test_represent_as(self, irep) -> None:
        """Test method ``represent_as``.

        Tested in astropy. Here only need to test it stays interpolated

        """
        # super().test_represent_as()
        got = irep.represent_as(coord.PhysicsSphericalRepresentation)

        assert isinstance(got, coord.PhysicsSphericalRepresentation)
        assert isinstance(got, icoord.InterpolatedRepresentation)

    def test_with_differentials(self, irep, rep) -> None:
        """Test method ``with_differentials``."""
        # super().test_with_differentials()

        got = irep.with_differentials(
            rep.differentials["s"].represent_as(
                coord.CartesianDifferential,
                base=rep.represent_as(coord.CartesianRepresentation),
            ),
        )

        assert isinstance(got, irep.__class__)
        assert isinstance(got, irep._class_)

        # --------------
        # bad differential length caught by astropy!

    def test_without_differentials(self, irep) -> None:
        """Test method ``without_differentials``."""
        # super().test_without_differentials()

        got = irep.without_differentials()

        assert isinstance(got, irep.__class__)
        assert isinstance(got, irep._class_)
        assert not got.differentials  # it's empty

    def test_clear_derivatives(self, irep) -> None:
        """Test method ``clear_derivatives``."""
        # calculate derivatives
        irep.derivative(n=1)
        irep.derivative(n=2)

        assert "lambda 1" in irep._derivatives.keys()
        assert "lambda 2" in irep._derivatives.keys()

        irep.clear_derivatives()

        assert "lambda 1" not in irep._derivatives.keys()
        assert "lambda 2" not in irep._derivatives.keys()
        assert not any(["lambda " in irep._derivatives.keys()])

    def test_derivative(self, irep, affine) -> None:
        """Test method ``derivative``."""
        super().test_derivative(irep, affine)

        # Testing cache, it's the only thing different between
        # InterpolatedRepresentationOrDifferential and
        # InterpolatedRepresentation
        assert "lambda 1" in irep._derivatives.keys()
        assert "lambda 2" in irep._derivatives.keys()

        assert irep.derivative(n=1) is irep._derivatives["lambda 1"]
        assert irep.derivative(n=2) is irep._derivatives["lambda 2"]

    def test_headless_tangent_vector(self, irep) -> None:
        """Test method ``headless_tangent_vector."""
        htv = irep.headless_tangent_vectors()

        assert isinstance(htv, icoord.InterpolatedRepresentation)
        assert all(htv.affine == irep.affine)

        # given the straight lines...
        for c in htv.components:
            assert np.allclose(getattr(htv, c), 0.1 * u.kpc)

    def test_tangent_vector(self, irep) -> None:
        """Test method ``headless_tangent_vector."""
        # BaseRepresentationOrDifferential derivative is not interpolated
        tv = irep.tangent_vectors()

        assert isinstance(tv, icoord.InterpolatedRepresentation)
        assert all(tv.affine == irep.affine)

        # given the straight lines...
        for c in tv.components:
            assert np.allclose(
                getattr(tv, c) - getattr(irep, c),
                0.1 * u.kpc,
            )

    def test___add__(self, irep, affine) -> None:
        """Test method ``__add__``."""
        super().test___add__(irep)

        # -----------
        # succeeds
        # requires stripping the differentials

        inst = irep.without_differentials()

        got = inst + inst
        expected = inst.data + inst.data

        # affine is the same
        assert all(got.affine == affine)
        # and components are the same
        for c in expected.components:
            assert all(getattr(got, c) == getattr(expected, c))

        # and a sanity check
        assert all(got.x == 2 * inst.x)

    def test___sub__(self, irep, affine) -> None:
        """Test method ``__sub__``."""
        super().test___sub__(irep)

        # -----------
        # succeeds
        # requires stripping the differentials

        inst = irep.without_differentials()

        got = inst - inst
        expected = inst.data - inst.data

        # affine is the same
        assert all(got.affine == affine)
        # and components are the same
        for c in expected.components:
            assert all(getattr(got, c) == getattr(expected, c))

    def test___mul__(self, irep, affine) -> None:
        """Test method ``__mul__``."""
        super().test___mul__(irep)

        # -----------
        # succeeds
        # requires stripping the differentials

        inst = irep.without_differentials()

        got = inst * 2
        expected = inst.data * 2

        # affine is the same
        assert all(got.affine == affine)
        # and components are the same
        for c in expected.components:
            assert all(getattr(got, c) == getattr(expected, c))

    def test___truediv__(self, irep, affine) -> None:
        """Test method ``__truediv__``."""
        super().test___truediv__(irep)

        # -----------
        # succeeds
        # requires stripping the differentials

        inst = irep.without_differentials()

        got = inst / 2
        expected = inst.data / 2

        # affine is the same
        assert all(got.affine == affine)
        # and components are the same
        for c in expected.components:
            assert all(getattr(got, c) == getattr(expected, c))


#####################################################################


class Test_InterpolatedCartesianRepresentation(Test_InterpolatedRepresentation):
    """Test InterpolatedCartesianRepresentation."""

    @pytest.fixture
    def irep_cls(self):
        return icoord.InterpolatedCartesianRepresentation

    @pytest.fixture
    def irep_dimensionless(self, irep_cls):
        return irep_cls(
            coord.CartesianRepresentation(
                x=[1, 2, 3, 4],
                y=[5, 6, 7, 8],
                z=[9, 10, 11, 12],
            ),
            affine=[0, 2, 4, 6],
        )

    #######################################################
    # Method tests

    def test_init(self, irep_cls, rep, affine) -> None:
        """Test method ``__init__``."""
        super().test_init(irep_cls=irep_cls, rep=rep, affine=affine)

        # TODO!

    def test_transform(self, irep_cls, irep) -> None:
        """Test method ``transform``.

        Astropy tests the underlying method. Only need to test that
        it is interpolated.

        """
        got = irep.transform(np.eye(3))

        assert isinstance(got, irep.__class__)
        assert isinstance(got, irep._class_)
        assert isinstance(got, irep_cls)

    def test___repr__(self, irep_cls, irep, irep_dimensionless) -> None:
        """Test method ``__repr__``."""
        super().test___repr__(irep)

        # Also need to test a dimensionless case
        s = repr(irep_dimensionless)
        assert "[dimensionless]" in s


#####################################################################


class Test_InterpolatedDifferential(Test_InterpolatedRepresentationOrDifferential):
    """Test InterpolatedDifferential."""

    @pytest.fixture
    def idif_cls(self):
        return icoord.InterpolatedDifferential

    @pytest.fixture
    def idif(self, idif_cls, dif, affine):
        return idif_cls(dif, affine=affine)

    #######################################################
    # Method tests

    def test_new(self, idif_cls, dif, affine) -> None:
        """Test method ``__new__``."""
        # super().test_new(irep_cls=idif_cls, rep=dif, affine=affine)  # not abstract

        # test wrong type
        with pytest.raises(TypeError, match="`rep` must be a differential type."):
            idif_cls(object())

    def test_call(self, idif_cls, idif) -> None:
        """Test method ``__call__``."""
        super().test_call(irep=idif)

        # Test it evaluates to the correct class type
        got = idif()

        assert isinstance(got, idif.__class__)
        assert not isinstance(got, (idif._class_, idif_cls))

    def test_represent_as(self, idif_cls, idif) -> None:
        """Test method ``represent_as``.

        Astropy tests the underlying method. Only need to test that
        it is interpolated.

        """
        # super().test_represent_as()

        got = idif.represent_as(
            coord.PhysicsSphericalDifferential,
            base=coord.PhysicsSphericalRepresentation(0 * u.rad, 0 * u.rad, 0 * u.km)
        )

        assert isinstance(got, coord.PhysicsSphericalDifferential)
        assert isinstance(got, icoord.InterpolatedDifferential)
        assert isinstance(got, idif_cls)

    def test__scale_operation(self, idif) -> None:
        """Test method ``_scale_operation``."""
        newinst = idif._scale_operation(operator.mul, 1.1)

        for c in idif.components:
            assert all(getattr(newinst, c) == 1.1 * getattr(idif, c))

        # TODO one that works

    def test_to_cartesian(self, idif) -> None:
        """Test method ``to_cartesian``.

        On Differentials, ``to_cartesian`` returns a Representation
        https://github.com/astropy/astropy/issues/6215

        """
        # -------------------
        # works

        newrep = idif.to_cartesian()

        assert isinstance(newrep, coord.CartesianRepresentation)
        assert isinstance(newrep, icoord.InterpolatedCartesianRepresentation)
        assert not isinstance(newrep, icoord.InterpolatedDifferential)


#####################################################################


class Test_InterpolatedCoordinateFrame(InterpolatedCoordinatesBase):
    """Test :class:`~{package}.{klass}`."""

    @pytest.fixture
    def irep(self, rep, affine):
        return icoord.InterpolatedRepresentation(rep, affine=affine)

    @pytest.fixture
    def frame(self):
        return coord.Galactocentric

    @pytest.fixture
    def icrd_cls(self):
        return icoord.InterpolatedCoordinateFrame

    @pytest.fixture
    def icrd(self, icrd_cls, frame, irep):
        return icrd_cls(frame(irep))

    #######################################################
    # Method Tests

    def test_init(self, icrd_cls, frame, irep, rep, affine) -> None:
        """Test method ``__init__``."""
        # -------------------
        # rep is interpolated

        c = icrd_cls(frame(irep))

        assert isinstance(c, icrd_cls)
        assert isinstance(c.data, icoord.InterpolatedRepresentation)

        # -------------------
        # rep is base astropy

        # doesn't work b/c no affine
        with pytest.raises(ValueError, match="`data` is not already interpolated"):
            icrd_cls(frame(rep), affine=None)

        # ----

        # works with affine
        c = icrd_cls(frame(rep), affine=affine)

        assert isinstance(c, icrd_cls)
        assert isinstance(c.data, icoord.InterpolatedRepresentation)

        # -------------------
        # rep is wrong type

        class Obj:
            data = object()

        with pytest.raises(TypeError, match="`data` must be type "):
            icrd_cls(Obj())

    def test__interp_kwargs(self, icrd) -> None:
        """Test method ``_interp_kwargs``."""
        # property get
        assert icrd._interp_kwargs is icrd.data._interp_kwargs

        # setter
        icrd._interp_kwargs = {"a": 1}
        assert icrd.data._interp_kwargs["a"] == 1
        icrd._interp_kwargs = {}  # reset

    def test_call(self, icrd, frame, num) -> None:
        """Test method ``__call__``.

        Since it returns a BaseCoordinateFrame, and does the evaluation
        through the InterpolatedRepresentation, all we need to test here
        is that it's the right type.

        """
        data = icrd()

        assert isinstance(data, frame)
        assert len(data) == num

    def test__class_(self, icrd_cls, icrd) -> None:
        """Test method ``_class_``."""
        assert issubclass(icrd._class_, icrd_cls)

    @pytest.mark.skip("TODO")
    def test__realize_class(self, icrd) -> None:
        """Test method ``_realize_class``."""
        assert False

    @pytest.mark.skip("TODO")
    def test_realize_frame(self, icrd) -> None:
        """Test method ``realize_frame``."""
        assert False

    def test_derivative(self, icrd, affine) -> None:
        """Test method ``derivative``.

        Just passes to the Representation.

        """
        # --------------------

        ideriv = icrd.derivative(n=1)  # a straight line

        assert all(ideriv.affine == affine)
        assert np.allclose(ideriv._values.view(float), 0.1)

        # --------------------

        ideriv = icrd.derivative(n=2)  # no 2nd deriv

        assert all(ideriv.affine == affine)
        assert np.allclose(ideriv._values.view(float), 0.0)

        # --------------------

        ideriv = icrd.derivative(n=3)  # no 3rd deriv

        assert all(ideriv.affine == affine)
        assert np.allclose(ideriv._values.view(float), 0.0)

    def test_affine(self, icrd, affine) -> None:
        """Test method ``affine``.

        Just passes to the Representation.

        """
        assert all(icrd.affine == affine)
        assert all(icrd.frame.data.affine == affine)

    def test_headless_tangent_vectors(self, icrd_cls, icrd) -> None:
        """Test method ``headless_tangent_vectors``.

        Wraps Representation in InterpolatedCoordinateFrame

        """
        htv = icrd.headless_tangent_vectors()

        assert isinstance(htv, icrd_cls)  # interp
        assert isinstance(htv, coord.BaseCoordinateFrame)

        for c in htv.data.components:
            assert np.allclose(getattr(htv.data, c), 0.1 * u.kpc)

    def test_tangent_vectors(self, icrd_cls, icrd) -> None:
        """Test method ``tangent_vectors``.

        Wraps Representation in InterpolatedCoordinateFrame

        """
        tv = icrd.tangent_vectors()

        assert isinstance(tv, icrd_cls)  # interp
        assert isinstance(tv, coord.BaseCoordinateFrame)

        for c in tv.data.components:
            assert np.allclose(
                getattr(tv.data, c) - getattr(icrd.data, c),
                0.1 * u.kpc,
            )

    def test___class__(self, icrd_cls, icrd) -> None:
        """Test method ``__class__``.

        Just passes to the CoordinateFrame.

        """
        assert icrd.__class__ is icrd.frame.__class__
        assert issubclass(icrd.__class__, coord.BaseCoordinateFrame)
        assert isinstance(icrd, icrd_cls)

    def test___getattr__(self, icrd, num) -> None:
        """Test method ``__getattr__``.

        Routes everything to underlying CoordinateFrame.
        Lets just test the ``shape``.

        """
        assert icrd.shape == icrd.frame.shape
        assert icrd.shape == (num,)

        assert icrd.ndim == icrd.frame.ndim
        assert icrd.ndim == 1

    def test_len(self, icrd, num) -> None:
        """Test method ``__len__``."""
        assert len(icrd) == num

    def test___getitem__(self, icrd_cls, icrd) -> None:
        """Test method ``__getitem__``."""
        # Test has problem when slicing with <3 elements
        # TODO? fix?
        with pytest.raises(Exception):
            icrd[:3]

        # works otherwise
        inst = icrd[:4]

        assert isinstance(inst, coord.BaseCoordinateFrame)
        assert isinstance(inst, icrd_cls)
        assert isinstance(inst, icrd.__class__)
        assert isinstance(inst, icrd._class_)

        assert inst.representation_type == icrd.representation_type

        assert len(inst) == 4

    def test_transform_to(self, icrd) -> None:
        """Test method ``transform_to``.

        All the transformation is handled in the frame. Only need to
        test that it's still interpolated.

        """
        newinst = icrd.transform_to(coord.HeliocentricTrueEcliptic())

        assert isinstance(newinst, coord.HeliocentricTrueEcliptic)
        assert isinstance(newinst, icoord.InterpolatedCoordinateFrame)

        assert isinstance(newinst.frame, coord.HeliocentricTrueEcliptic)

        assert isinstance(newinst.frame.data, coord.CartesianRepresentation)
        assert isinstance(newinst.frame.data, icoord.InterpolatedRepresentation)

        assert isinstance(newinst.frame.data.data, coord.CartesianRepresentation)

    def test_copy(self, icrd) -> None:
        """Test method ``copy``."""
        newrep = icrd.copy()

        assert newrep is not icrd  # not the same object
        assert isinstance(newrep, icoord.InterpolatedCoordinateFrame)

        # TODO more tests

    def test__frame_attrs_repr(self, icrd) -> None:
        """Test method ``_frame_attrs_repr``."""
        assert icrd._frame_attrs_repr() == icrd.frame._frame_attrs_repr()
        # TODO more tests

    def test__data_repr(self, icrd) -> None:
        """Test method ``_data_repr``."""
        data_repr = icrd._data_repr()
        assert isinstance(data_repr, str)
        # TODO more tests

    def test___repr__(self, icrd) -> None:
        """Test method ``__repr__``."""
        s = icrd.__repr__()
        assert isinstance(s, str)

        # a test for unit dif types
        icrd.representation_type = coord.UnitSphericalRepresentation
        s = icrd.__repr__()
        assert isinstance(s, str)

        # TODO more tests


#####################################################################


class Test_InterpolatedSkyCoord(InterpolatedCoordinatesBase):
    """Test InterpolatedSkyCoord."""

    @pytest.fixture
    def irep(self, rep, affine):
        return icoord.InterpolatedRepresentation(rep, affine=affine)

    @pytest.fixture
    def frame(self):
        return coord.Galactocentric

    @pytest.fixture
    def icrd_cls(self):
        return icoord.InterpolatedCoordinateFrame

    @pytest.fixture
    def icrd(self, icrd_cls, frame, irep):
        return icrd_cls(frame(irep))

    @pytest.fixture
    def iscrd_cls(self):
        return icoord.InterpolatedSkyCoord

    @pytest.fixture
    def iscrd(self, iscrd_cls, icrd):
        return iscrd_cls(icrd)

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

    def test_init(self, iscrd_cls, num, affine) -> None:
        """Test method ``__init__``.

        Copying from astropy docs

        """
        # -----------
        c = iscrd_cls(
            [10] * num,
            [20] * num,
            unit="deg",
            affine=affine,
        )  # defaults to ICRS frame
        self._test_isc(c)

        # -----------
        c = iscrd_cls(
            [1, 2, 3, 4],
            [-30, 45, 8, 16],
            frame="icrs",
            unit="deg",
            affine=affine[:4],
        )  # 4 coords
        self._test_isc(c)

        # -----------
        coords = [
            "1:12:43.2 +31:12:43",
            "1:12:43.2 +31:12:43",
            "1:12:43.2 +31:12:43",
            "1 12 43.2 +31 12 43",
        ]
        c = iscrd_cls(
            coords,
            frame=coord.FK4,
            unit=(u.hourangle, u.deg),
            obstime="J1992.21",
            affine=affine[:4],
        )
        self._test_isc(c)

        # -----------
        c = iscrd_cls(
            ["1h12m43.2s +1d12m43s"] * num,
            frame=coord.Galactic,
            affine=affine,
        )  # Units from string
        self._test_isc(c)

        # # -----------
        c = iscrd_cls(
            frame="galactic",
            l=["1h12m43.2s"] * num,
            b="+1d12m43s",  # NOT BROADCASTING THIS ONE
            affine=affine,
        )
        self._test_isc(c)

        # -----------
        ra = coord.Longitude([1, 2, 3, 4], unit=u.deg)  # Could also use Angle
        dec = np.array([4.5, 5.2, 6.3, 7.4]) * u.deg  # Astropy Quantity
        c = iscrd_cls(
            ra,
            dec,
            frame="icrs",
            affine=affine[:4],
        )
        self._test_isc(c)

        # -----------
        c = iscrd_cls(
            frame=coord.ICRS,
            ra=ra,
            dec=dec,
            obstime="2001-01-02T12:34:56",
            affine=affine[:4],
        )
        self._test_isc(c)

        # -----------
        c = coord.FK4(
            [1] * num * u.deg,
            2 * u.deg,
        )  # Uses defaults for obstime, equinox
        c = iscrd_cls(
            c,
            obstime="J2010.11",
            equinox="B1965",
            affine=affine,
        )  # Override defaults
        self._test_isc(c)

        # -----------
        c = iscrd_cls(
            w=[0] * num,
            u=1,
            v=2,
            unit="kpc",
            frame="galactic",
            representation_type="cartesian",
            affine=affine,
        )
        self._test_isc(c, representation_type=coord.CartesianRepresentation)

        # -----------
        c = iscrd_cls(
            [
                coord.ICRS(ra=1 * u.deg, dec=2 * u.deg),
                coord.ICRS(ra=3 * u.deg, dec=4 * u.deg),
            ]
            * (num // 2),
            affine=affine,
        )
        self._test_isc(c)

    def test_call(self, iscrd, num) -> None:
        """Test method ``__call__``."""
        inst = iscrd()

        assert isinstance(inst, coord.SkyCoord)
        assert len(inst) == num

    def test_transform_to(self, iscrd_cls, iscrd, affine) -> None:
        """Test method ``transform_to``."""
        for frame in (coord.ICRS,):
            inst = iscrd.transform_to(frame())

            assert isinstance(inst, coord.SkyCoord)
            assert isinstance(inst, iscrd_cls)

            assert isinstance(inst.frame, frame)

            assert all(inst.affine == affine)

    def test_separation(self) -> None:
        """Test method ``separation``."""
        pass  # it just calls super b/c docstring issues

    def test_separation_3d(self) -> None:
        """Test method ``separation_3d``."""
        pass  # it just calls super b/c docstring issues

    def test_match_to_catalog_sky(self) -> None:
        """Test method ``match_to_catalog_sky``."""
        pass  # it just calls super b/c docstring issues

    def test_match_to_catalog_3d(self) -> None:
        """Test method ``match_to_catalog_3d``."""
        pass  # it just calls super b/c docstring issues

    def test_search_around_sky(self) -> None:
        """Test method ``search_around_sky``."""
        pass  # it just calls super b/c docstring issues

    def test_search_around_3d(self) -> None:
        """Test method ``search_around_3d``."""
        pass  # it just calls super b/c docstring issues


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
