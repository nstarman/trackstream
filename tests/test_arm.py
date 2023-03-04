"""Testing :mod:`~trackstream.stream.core`."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, TypeVar

import pytest

from trackstream.stream.base import StreamBase

if TYPE_CHECKING:
    from trackstream.stream.arm import StreamArm
    from trackstream.stream.core import Stream

S = TypeVar("S", bound=StreamBase)

##############################################################################
# TESTS
##############################################################################


class StreamArmTestMixin:
    @pytest.fixture(params=["arm1"])
    def arm(self, stream: Stream, request) -> StreamArm:
        return getattr(stream, request.param)

    @pytest.fixture()
    def arm_attr_name(self, arm: StreamArm) -> str:
        return arm._enclosing_attr

    # ===============================================================

    def test_arm_full_stream(self, arm, stream: S) -> None:
        assert arm.full_stream is stream

    def test_arm_name(self, arm, arm_attr_name) -> None:
        # split the name on the digit
        expected = " ".join(list(filter(None, re.split(r"(\d+)", arm._enclosing_attr))))
        assert arm.name == expected

        # that test is a little abstract, so hitting the most common
        if arm_attr_name == "arm1":
            assert arm.name == "arm 1"
        elif arm_attr_name == "arm2":
            assert arm.name == "arm 2"

    def test_arm_full_name(self, arm, stream, name) -> None:
        expected = f"Stream, {arm.name}" if name is None else f"{name}, {arm.name}"

        assert arm.full_name == expected

    # def test_arm_index(self, arm, arm_attr_name, stream: S) -> None:

    def test_arm_has_data(self, arm, arm_attr_name, stream: S) -> None:
        expected = any(stream.data["tail"] == arm_attr_name)
        assert arm.has_data == expected

    def test_arm_data(self, arm, arm_attr_name, stream: S) -> None:
        if not arm.has_data:
            with pytest.raises(Exception, match=f"no {arm.name}"):
                arm.data
        else:
            assert all(arm.data == stream.data.loc[arm_attr_name])

    def test_arm_data_frame(self, arm, stream: S) -> None:
        assert arm.data_frame is stream.data_frame

    def test_arm_system_frame(self, arm, stream: S) -> None:
        assert arm.system_frame is stream.system_frame

    def test_arm_origin(self, arm, stream: S) -> None:
        assert arm.origin == stream.origin

    def test_arm_coords(self, arm, arm_attr_name, stream: S) -> None:
        index = stream.data["tail"] == arm_attr_name
        expected = stream.coords[index]
        assert all(arm.coords == expected)

    def test_arm_data_max_lines(self, arm, stream: S) -> None:
        assert arm._data_max_lines is stream._data_max_lines


# class TestStreamArm(StreamArmTestMixin, StreamBaseTest):
#     """Test `trackstream.stream.arm.StreamArm`.

#     Most of the relevant tests are covered in `streamtrack.stream.tests.test_core.Test_Stream`.
#     """

#     pass
