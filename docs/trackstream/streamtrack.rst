.. trackstream-streamtrack

************
Stream Track
************

The TrackStream class is the main class for creating a StreamTrack given data.
The TrackStream will fit the track to the data, in Sci-Kit Learn style with
methods "fit", "predict", and "fit-predict".

The StreamTrack is analogous to an interpolated CoordinateFrame where the data
is is a univariate function of the track arc-length. The call method returns a
|Frame| by evaluating the interpolations.


Reference/API
=============

.. automodapi:: trackstream.track
   :no-heading:
   :no-main-docstr:
   :include-all-objects:
