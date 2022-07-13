# see LICENSE.rst

"""Exceptions."""

EXCEPT_3D_NO_DISTANCES = ValueError(
    "this stream does not have distance information; cannot compute track with distances.",
)


EXCEPT_NO_KINEMATICS = ValueError(
    "this stream does not have kinematic information; " "cannot compute track with velocities.",
)
