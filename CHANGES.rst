==================
0.1.0 (2020-09-01)
==================

New Features
------------

trackstream
^^^^^^^^^^^

- basic version test [#2]

- TrackStream framework [#13]
  The TrackStream class is the main class for creating a StreamTrack
  given data. The TrackStream will fit the track to the data.

    + In Sci-Kit Learn style with methods "fit", "predict", and "fit-predict".

- StreamTrack framework [#13]
  The StreamTrack is analogous to an interpolated CoordinateFrame
  where the data is is a univariate function of the track arc-length.
  The call method returns a CoordinateFrame by evaluating the interpolations.


trackstream.tests.helper
^^^^^^^^^^^^^^^^^^^^^^^^

- Baseclass-dependent test helper class [#23]


trackstream.utils
^^^^^^^^^^^^^^^^^

- Scipy splines with units. [#24]
  Currently only InterpolatedUnivariateSpline.

- Full implementation of interpolated splines [#30]

- Interpolated Coordinates [#30]


API Changes
-----------

N/A


Bug Fixes
---------

List them


Other Changes and Additions
---------------------------

- renamed project ``trackstream`` from ``streamtrack`` [#2]

- Reapply Project Template [#21]

    + Add PR template [#7]

- Add Issue Templates [#22]

- Adopt pre-commit [#9]

    + `isort <https://pypi.org/project/isort/>`_
    + `black <https://pypi.org/project/black/>`_
    + `flake8 <https://pypi.org/project/flake8/>`_

- Modified BSD-3 Template [#29]

Actions
^^^^^^^

- Add action greeting [#10]

- Add PR/Issue labeler action [#12]

- Label stale PRs / issues [#14]
