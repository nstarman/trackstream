# type: ignore

"""Testing :mod:`~trackstream.stream.core`."""

##############################################################################
# IMPORTS

# STDLIB
import copy
from copyreg import pickle
from typing import Any, Dict, Iterator, Optional, Type, TypeVar

# THIRD PARTY
import pytest
from astropy.coordinates import BaseCoordinateFrame, RadialDifferential, SkyCoord
from astropy.table import QTable
from attr import evolve

# LOCAL
from .test_base import StreamBaseTest
from trackstream.stream.arm import StreamArm
from trackstream.stream.base import StreamBase
from trackstream.stream.core import Stream
from trackstream.stream.tests.test_arm import StreamArmTestMixin
from trackstream.tests.helper import IbataEtAl2017  # noqa: F401
from trackstream.track.fitter import TrackStreamArm
from trackstream.utils import resolve_framelike

S = TypeVar("S", bound=StreamBase)

##############################################################################
# TESTS
##############################################################################


class Test_Stream(StreamBaseTest, StreamArmTestMixin):
    @pytest.fixture(scope="class")
    def stream_cls(self) -> Type[S]:
        """Stream class."""
        return Stream

    @pytest.fixture(scope="class")
    def DATA(self, IbataEtAl2017) -> Iterator[Dict[str, Any]]:  # noqa: F811
        """Fixture yielding all stream data sets."""
        yield from (IbataEtAl2017,)

    @pytest.fixture(scope="class")
    def data_table(self, DATA) -> QTable:
        return DATA["data"]

    @pytest.fixture(scope="class")
    def data_error_table(self, DATA) -> Optional[QTable]:
        return DATA["data_error"]

    @pytest.fixture(scope="class")
    def origin(self, DATA) -> SkyCoord:
        return DATA["origin"]

    @pytest.fixture(scope="class", params=[None, True])
    def name(self, request) -> Optional[str]:
        return self.__class__.__name__ if request.param else None

    @pytest.fixture(scope="function", params=[None, (None, None)])
    def fitter(self, request) -> Optional[TrackStreamArm]:
        if request.param is None:
            return None
        else:
            onsky, kinematics = request.param
            return TrackStreamArm(onsky=onsky, kinematics=kinematics)

    @pytest.fixture(scope="class")
    def stream(self, stream_cls, data_table, data_error_table, origin, frame, name) -> S:
        """Stream instance."""
        return stream_cls(data_table, origin, data_err=data_error_table, frame=frame, name=name)

    @pytest.fixture(scope="function")
    def tempstream(self, stream: S) -> S:
        """function-scoped stream"""
        return evolve(stream)

    # -----------------------------------------------------

    @pytest.fixture(scope="class")
    def stream_f(self, stream: S) -> S:
        """Stream with a fit frame"""
        strm = evolve(stream)
        strm.fit_frame(fitter=None)
        return strm

    @pytest.fixture(scope="function")
    def tempstream_f(self, stream_f: S) -> S:
        """function-scoped stream"""
        return evolve(stream_f)

    # -----------------------------------------------------

    @pytest.fixture(scope="class", params=[False, True])
    def fit_track_force(self, request) -> bool:
        return request.param

    @pytest.fixture(scope="class")
    def stream_t(self, stream_f):
        """Fixture returning a fit stream."""
        strm = copy.deepcopy(stream_f)  # decouple from stream
        strm.fit_track(force=False, **{})  # TODO! test kwargs
        return strm

    @pytest.fixture(scope="function")
    def tempstream_t(self, stream_t: S):
        return evolve(stream_t)

    # -----------------------------------------------------

    @pytest.fixture(params=["arm1", "arm2"])
    def arm(self, stream: Stream, request) -> StreamArm:
        return getattr(stream, request.param)

    # ===============================================================
    # StreamBase

    def test_data(self, stream: S) -> None:
        """Test property ``data``."""
        super().test_data(stream)

        # todo!

        # See Data Normalization tests

    def test_data_frame(self, stream: S) -> None:
        """Test property ``data_frame``."""
        super().test_data_frame(stream)

        # the frame is easy
        assert isinstance(stream.data_frame, type(stream.data_coords.frame))
        # the representation type depends on whether there are distances and if
        # the representation type has a "unit" version of itself, like
        # (Unit)SphericalRepresentation.
        reptype = stream.data_coords.frame.representation_type
        if not stream.has_distances:
            reptype = getattr(reptype, "_unit_representation", reptype)
        assert stream.data_frame.representation_type is reptype

    @pytest.mark.skip("TODO!")
    def test_coords(self, stream: S) -> None:
        """Test property ``coords``."""
        super().test_coords(stream)

    @pytest.mark.skip("TODO!")
    def test_coords_ord(self, stream: S) -> None:
        """Test property ``coords_ord``."""
        super().test_coords_ord(stream)

    def test_frame(self, stream: S) -> None:
        """Test property ``frame``."""
        super().test_frame(stream)

        assert stream.frame is stream.system_frame

    def test_name(self, stream, name) -> None:
        """Test property ``name``."""
        super().test_name(stream)
        assert stream.name is name

    def test_origin(self, stream, origin) -> None:
        """Test property ``origin``."""
        super().test_origin(stream)

        # stream origin's frame depends on frame at init
        assert stream.origin.transform_to(origin.frame) == origin
        # so also test the frame matches expectations
        assert isinstance(stream.origin.frame, type(stream._best_frame))

        # the origin should not lose information, e.g. distance or kinematics.
        if origin.spherical.distance.unit.physical_type == "length":
            assert stream.origin.spherical.distance.unit.physical_type == "length"
        if "s" in origin.data.differentials:
            assert "s" in stream.origin.data.differentials

    def test_has_distances(self, stream: S) -> None:
        """Test property ``has_distances``."""
        super().test_has_distances(stream)

        data_onsky = stream.data_coords.spherical.distance.unit.physical_type == "dimensionless"
        origin_onsky = stream.origin.spherical.distance.unit.physical_type == "dimensionless"
        expected = not data_onsky and not origin_onsky

        assert stream.has_distances is expected

    def test_has_kinematics(self, stream: S) -> None:
        """Test property ``has_kinematics``."""
        expected = "s" in stream.data_coords.data.differentials
        if expected is True:
            expected &= not isinstance(stream.data_coords.data.differentials["s"], RadialDifferential)
        assert stream.has_kinematics is expected

    def test_full_name(self, stream: S) -> None:
        super().test_full_name(stream)

    # ===============================================================

    def test_init_fail_numargs(self, stream_cls, data_table) -> None:
        """Test with wrong number of arguments. A reminder to include ``origin``."""
        with pytest.raises(TypeError, match="origin"):
            stream_cls(data_table)

    def test_init(self, stream_cls, data_table, data_error_table, origin, frame, name) -> None:
        """Test initialization."""
        # Full initialization
        stream = stream_cls(data_table, origin, data_err=data_error_table, frame=frame, name=name)

        # origin
        assert isinstance(stream.origin, SkyCoord)
        assert stream.origin == origin

        # frame
        assert stream._init_system_frame is frame

        # data
        assert isinstance(stream.data, QTable)

    # -----------------------------------------------------

    def test_data_coords(self, stream: S) -> None:
        assert all(stream.data_coords == stream.data["coord"])
        # See Data Normalization tests

    def test_system_frame(self, stream, frame) -> None:
        # Without fitting a frame this is the initializing frame
        if frame is None:
            assert stream.system_frame is None

        else:
            expected = resolve_framelike(frame)
            assert stream.system_frame == expected

    def test_system_frame_fit(self, stream_f, frame) -> None:
        """"""
        assert stream_f.system_frame is not None

        if frame is None:
            assert stream_f._init_system_frame is None
            assert stream_f._cache["system_frame"] is not None

        else:
            assert stream_f._init_system_frame is not None
            assert stream_f._cache["system_frame"] is None

    def test_best_frame(self, stream, stream_f, frame) -> None:
        # Without fitting a frame this is the initializing frame
        if frame is not None:  # given at initialization
            expected = stream._init_system_frame
            assert stream._best_frame == expected
            assert stream_f._best_frame == expected

        else:  # fit or notfit
            assert stream._best_frame == stream.data_frame
            assert stream_f._best_frame == stream_f.system_frame

    def test_number_of_tails_with_data(self, stream: S) -> None:
        expected = any(stream.data["tail"] == "arm1") + any(stream.data["tail"] == "arm2")
        assert stream.number_of_tails_with_data == expected

    # -----------------------------------------------------

    def test_fit_frame_but_init_frame(self, tempstream, frame) -> None:
        if frame is None:
            pytest.skip("there's an init system frame.")

        with pytest.raises(TypeError, match="a system frame"):
            tempstream.fit_frame()

    def test_fit_frame_already_fit(self, tempstream_f, frame) -> None:
        if frame is None:
            pytest.skip("there's an init system frame.")

        # fails if already fit and not reforced.
        with pytest.raises(ValueError, match="already fit"):
            tempstream_f.fit_frame(force=False)

        # refits if forced
        fitframe = tempstream_f.fit_frame(force=True)

        assert isinstance(fitframe, BaseCoordinateFrame)
        assert False  # TODO! tests

    @pytest.mark.skip("TODO!")
    def test_fit_frame(self, tempstream: S) -> None:

        frame = tempstream.fit_frame()

        assert isinstance(frame, BaseCoordinateFrame)

        assert False

    # -----------------------------------------------------

    def test_track_notfit(self, stream: S) -> None:
        with pytest.raises(ValueError, match="need to fit"):
            stream.track

    @pytest.mark.skip("TODO!")
    def test_track_fit(self, tempstream_f: S) -> None:
        tempstream_f.track

    # -----------------------------------------------------

    def test_fit_track_already_fit(self, stream_t: S) -> None:
        with pytest.raises(ValueError, match="already fit"):
            stream_t.fit_track(force=False)

    @pytest.mark.skip("TODO!")
    def test_fit_track(tempstream_f: S, fitter) -> None:
        pass

    # -----------------------------------------------------

    @pytest.mark.skip("TODO!")
    def test_predict_track(self, stream_t: S) -> None:
        pass

    # ===============================================================
    # Usage Tests

    @pytest.mark.skip("TODO!")
    def test_track_probability(self, stream_t: S) -> None:
        stream_t.track

    @pytest.mark.xfail(reason="can't pickle GenericDifferential")
    def test_pickle_fit(self, stream_t: S) -> None:
        pickle.dumps(stream_t)
