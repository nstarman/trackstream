# -*- coding: utf-8 -*-

"""Core Functions."""

__all__ = ["Stream"]


##############################################################################
# IMPORTS

# STDLIB
import typing as T
import weakref

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy.table import Column, QTable, Table
from astropy.utils.decorators import lazyproperty

# LOCAL
from trackstream._type_hints import CoordinateType, FrameType
from trackstream.preprocess.som import SelfOrganizingMap1D
from trackstream.utils.path import path_moments

##############################################################################
# CODE
##############################################################################


class StreamArmDescriptor:
    def __init__(self) -> None:
        # references to parent class and instance
        self._parent_attr = None  # set in __set_name__
        self._parent_cls = None
        self._parent_ref = None

    @property
    def _parent(self):
        """Parent instance Cosmology."""
        return self._parent_ref() if self._parent_ref is not None else self._parent_cls

    # ------------------------------------

    def __set_name__(self, objcls, name):
        self._parent_attr = name

    def __get__(self, obj, objcls):
        # accessed from a class
        if obj is None:
            self._parent_cls = objcls
            return self

        # accessed from an obj
        equivs = obj.__dict__.get(self._parent_attr)  # get from obj
        if equivs is None:  # hasn't been created on the obj
            descriptor = self.__class__()
            descriptor._parent_cls = obj.__class__
            descriptor._parent_attr = self._parent_attr
            obj.__dict__[self._parent_attr] = descriptor

        # We set `_parent_ref` on every call, since if one makes copies of objs,
        # 'descriptor' will be copied as well, which will lose the reference.
        descriptor._parent_ref = weakref.ref(obj)
        return descriptor

    # ------------------------------------

    @property
    def index(self) -> Column:
        return self._parent.data["tail"] == self._parent_attr

    @property
    def has_data(self):
        return any(self.index)

    @property
    def data(self) -> Column:
        if not self.has_data:
            raise Exception("no arm 1")  # TODO! specific exception
        return self._parent.data[self.index]

    @property
    def coords(self) -> coord.SkyCoord:
        """The coordinates of the arm."""
        arm: coord.SkyCoord
        arm = self._parent.coords[self.index]
        return arm


# /class


class Stream:
    """A Stellar Stream.

    Parameters
    ----------
    data : `~astropy.table.Table`

    origin : `~astropy.coordinates.ICRS`
        The origin point of the rotated reference frame.

    data_err : `~astropy.table.QTable` (optional)
        The data_err must have (at least) column names
        ["x_err", "y_err", "z_err"]

    frame : `~astropy.coordinates.BaseCoordinateFrame` or None (optional, keyword-only)
        The stream frame. Locally linearizes the data.
        If not None, need to fit for the frame (default).

    """

    arm1 = StreamArmDescriptor()
    arm2 = StreamArmDescriptor()

    # ===============================================================

    def __init__(
        self,
        data: QTable,
        origin: FrameType,
        data_err: T.Optional[Table] = None,
        *,
        frame: T.Optional[CoordinateType] = None,
    ):
        # system attributes
        self.origin: coord.SkyCoord = coord.SkyCoord(origin, copy=False)
        self._system_frame: T.Optional[FrameType] = frame

        self._cache = dict()  # TODO! improve

        # ----------
        # process the data

        # seed values set in functions
        self._original_data: coord.SkyCoord = None

        # processed data -> QTable
        self.data: QTable = self._normalize_data(data)
        self._data_max_lines = 10

    # -----------------------------------------------------

    @property
    def system_frame(self) -> T.Optional[coord.BaseCoordinateFrame]:
        """A system-centric frame.

        Determined from the argument ``frame`` at initialization.
        If None (default) and the method ``fit`` has been called,
        then a system frame has been found and cached.
        """
        frame: T.Optional[coord.BaseCoordinateFrame]
        if self._system_frame is not None:
            frame = self._system_frame
        else:
            frame = self._cache.get("frame")

        return frame

    @property
    def frame(self) -> coord.BaseCoordinateFrame:
        """Alias for ``system_frame``."""
        return self.system_frame

    @lazyproperty
    def number_of_tails(self) -> int:
        """Number of tidal tails.

        Returns
        -------
        number_of_tails : int
            There can only be 1, or 2 tidal tails.
        """
        return 2 if (self.arm1.has_data and self.arm2.has_data) else 1

    @property  # TODO! make lazy
    def coords(self) -> coord.SkyCoord:
        """Coordinates."""
        frame: coord.SkyCoord
        if self.system_frame is not None:
            frame = self.system_frame
        else:
            frame = self.data_frame
        return self.data_coords.transform_to(frame)

    # ===============================================================

    @property
    def data_coords(self) -> coord.SkyCoord:
        """Get ``coord`` from data table."""
        return self.data["coord"]

    @property
    def data_frame(self) -> FrameType:
        """The frame of the data."""
        return self.data_coords.frame.replicate_without_data()

    # ===============================================================
    # Data normalization

    def _normalize_data(self, original: Table) -> QTable:
        """Normalize data table.

        Just calls other functions.

        Parameters
        ----------
        original : :class:`~astropy.table.Table`

        Returns
        -------
        data : :class:`~astropy.table.QTable`

        """
        data = QTable()  # going to be assigned in-place

        # 1) data probability
        self._normalize_data_probability(original, data, default_weight=1)

        # 2) coordinates. `data` modded in-place
        self._normalize_data_coordinates(original, data)

        # 3) SOM ordering
        self._normalize_data_arm_index(original, data)

        # Metadata
        # TODO? selective, or just copy over?
        data.meta = original.meta.copy()  # TODO? deepcopy?

        return data

    def _normalize_data_probability(
        self,
        original: Table,
        data: QTable,
        default_weight: T.Union[float, u.Quantity] = 1.0,
    ) -> None:
        """Data probability. Units of percent. Default is 100%.

        Parameters
        ----------
        original : |Table|
            The original data.
        data : |QTable|
            The normalized data.
        default_weight : float
            The default membership probability.
            If float, then range 0-1 maps to 0-100%.
            If has unit of percent, then unchanged
        """
        colns = [n.lower() for n in original.colnames]

        if "pmemb" in colns:
            Pmemb = original[original.colnames[colns.index("pmemb")]]
        else:
            Pmemb = np.ones(len(original)) * default_weight  # non-scalar

        data["Pmemb"] = u.Quantity(Pmemb).to(u.percent)  # in %

    def _normalize_data_coordinates(
        self,
        original: Table,
        data: QTable,
    ) -> None:
        """Parse the data table.

        - the frame is stored in ``_data_frame``
        - the representation is stored in ``_data_rep``
        - the original data representation  is in ``_data``

        Parameters
        ----------
        data : |Table|
            The stream data.

        data_err : |Table| or |Frame| or instance
            It must be in Cartesian coordinates!

        Raises
        ------
        TypeError
            if `data` is not |Table| or |Frame|
        """
        # TODO!!! better

        data["tail"] = original["tail"]

        # ----------
        # 1) the data

        # First look for a column "coord"
        if "coord" in original.colnames:
            sc = osc = coord.SkyCoord(original["coord"], copy=False)
        else:
            sc = osc = coord.SkyCoord.guess_from_table(original)

        self._original_data = osc

        # Convert frame and representation type
        frame = (
            self.system_frame
            if self.system_frame is not None
            else osc.frame.replicate_without_data()
        )
        sc = sc.transform_to(frame)
        sc.representation_type = coord.CartesianRepresentation

        # it's now clean and can be added
        data["coord"] = sc

        # also want to store the components, for plotting
        compnames = sc.representation_component_names.keys()
        for n in compnames:
            data[n] = getattr(sc, n)

        # ----------
        # 2) the error
        # TODO! want errors in frame of the data

        err_cols = ["x_err", "y_err", "z_err"]
        for n in err_cols:
            data[n] = original[n]  # transfer

        # ----------

        data = data.group_by("tail")
        data.add_index("tail")

    def _normalize_data_arm_index(
        self,
        original: Table,
        data: QTable,
    ) -> None:
        """Data probability. Units of percent. Default is 100%.

        Parameters
        ----------
        original : |Table|
            The original data.
        data : |QTable|
            The normalized data.
        """
        if "SOM" in original.colnames:
            data["SOM"] = original["SOM"]
        else:
            data["SOM"] = None

    # ===============================================================
    # Fitting

    @property
    def track(self) -> "StreamTrack":  # noqa: F821
        """Stream track.

        Raises
        ------
        ValueError
            If track is not fit.
        """
        track = self._cache.get("track")
        if track is None:
            raise ValueError("need to fit track.")
        return track

    def fit_track(
        self,
        arm1SOM: T.Optional[SelfOrganizingMap1D] = None,
        arm2SOM: T.Optional[SelfOrganizingMap1D] = None,
        *,
        force: bool = False,
        **kwargs,
    ) -> "StreamTrack":  # noqa: F821
        """Make a stream track.

        Parameters
        ----------
        arm1SOM, arm2SOM : `~trackstream.preprocess.SelfOrganizingMap` (optional, keyword-only)
            Fiducial SOMs for stream arms 1 and 2, respectively.
        force : bool
            Whether to force a fit, even if already fit.
        **kwargs
            Passed to :meth:`trackstream.TrackStream.fit`.

        Returns
        -------
        `trackstream.StreamTrack`
        """
        if not force and "tracker" in self._cache:
            raise Exception("already fit. use ``force`` to re-fit.")

        # LOCAL
        from trackstream.core import StreamTrack, TrackStream

        self._cache["tracker"] = tracker = TrackStream(arm1SOM=arm1SOM, arm2SOM=arm2SOM)

        track: StreamTrack = tracker.fit(self, **kwargs)
        self._cache["track"] = track

        # Add SOM ordering to data
        self.data["SOM"] = np.empty(len(self.data), dtype=int)
        self.data["SOM"][self.arm1.index] = tracker._cache["arm1_visit_order"]
        self.data["SOM"][self.arm2.index] = tracker._cache["arm2_visit_order"]

        return track

    # ===============================================================
    # Math on the Track (requires fitting track)

    def predict_track(
        self,
        affine: T.Optional[u.Quantity] = None,
        angular: bool = False,
    ) -> path_moments:
        return self.track()

    # ===============================================================
    # Misc

    def _base_repr_(self, max_lines=None):
        """mirroring implementation in astropy Table."""
        header: str = super().__repr__()
        frame: str = repr(self.frame)

        datarep: str = self.data._base_repr_(
            html=False,
            max_width=None,
            max_lines=self._data_max_lines,
        )
        table: str = "\n\t".join(datarep.split("\n")[1:])

        return header + "\n  Frame:\n\t" + frame + "\n  Data:\n\t" + table

    def __repr__(self) -> str:
        return self._base_repr_(max_lines=self._data_max_lines)


##############################################################################
# END
