# -*- coding: utf-8 -*-
# type: ignore

"""Testing :mod:`~trackstream.stream.core`."""

##############################################################################
# IMPORTS

# STDLIB
import re
from typing import TypeVar

# THIRD PARTY
import pytest

# LOCAL
from trackstream.stream.arm import StreamArmDescriptor
from trackstream.stream.base import StreamBase
from trackstream.stream.core import Stream

S = TypeVar("S", bound=StreamBase)

##############################################################################
# TESTS
##############################################################################


class StreamArmTestMixin:
    @pytest.fixture(params=["arm1"])
    def arm(self, stream: Stream, request) -> StreamArmDescriptor:
        return getattr(stream, request.param)

    @pytest.fixture
    def arm_attr_name(self, arm: StreamArmDescriptor) -> str:
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
        if name is None:
            expected = f"Stream, {arm.name}"
        else:
            expected = f"{name}, {arm.name}"

        assert arm.full_name == expected

    def test_arm_index(self, arm, arm_attr_name, stream: S) -> None:
        expected = stream.data["tail"] == arm_attr_name
        assert all(arm.index == expected)

    def test_arm_has_data(self, arm, arm_attr_name, stream: S) -> None:
        expected = any(stream.data["tail"] == arm_attr_name)
        assert arm.has_data == expected

    def test_arm_data(self, arm, arm_attr_name, stream: S) -> None:
        if not arm.has_data:
            with pytest.raises(Exception, match=f"no {arm.name}"):
                arm.data
        else:
            index = stream.data["tail"] == arm_attr_name
            expected = stream.data[index]
            assert all(arm.data == expected)

    def test_arm_data_frame(self, arm, stream: S) -> None:
        assert arm.data_frame is stream.data_frame

    def test_arm_frame(self, arm, stream: S) -> None:
        assert arm.frame is stream.frame

    def test_arm_origin(self, arm, stream: S) -> None:
        assert arm.origin == stream.origin

    def test_arm_coords(self, arm, arm_attr_name, stream: S) -> None:
        index = stream.data["tail"] == arm_attr_name
        expected = stream.coords[index]
        assert all(arm.coords == expected)

    def test_arm_data_max_lines(self, arm, stream: S) -> None:
        assert arm._data_max_lines is stream._data_max_lines


# class TestStreamArmDescriptor(StreamArmTestMixin, StreamBaseTest):
#     """Test `trackstream.stream.arm.StreamArmDescriptor`.

#     Most of the relevant tests are covered in `streamtrack.stream.tests.test_core.Test_Stream`.
#     """

#     pass
