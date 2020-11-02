.. trackstream-streamtrack

************
Stream Track
************

The TrackStream class is the main class for creating a StreamTrack given data.
The TrackStream will fit the track to the data, in Sci-Kit Learn style with methods "fit", "predict", and "fit-predict".

The StreamTrack is analogous to an interpolated CoordinateFrame where the data is is a univariate function of the track arc-length. The call method returns a |CoordinateFrame| by evaluating the interpolations.

Examples
========

TODO

- from Test_TrackStream
- from my Poster


Reference/API
=============

.. automodapi:: trackstream.core
   :no-heading:
   :no-main-docstr:
   :include-all-objects:


.. automodapi:: trackstream.tests.test_TrackStream
   :include-all-objects:

.. automodapi:: trackstream.tests.test_StreamTrack
   :include-all-objects:
