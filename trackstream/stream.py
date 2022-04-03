# -*- coding: utf-8 -*-

"""Core Functions."""

__all__ = ["Stream"]


##############################################################################
# IMPORTS

# STDLIB
import itertools
import re
import weakref
from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Optional, Union

# THIRD PARTY
import astropy.units as u
import numpy as np
from astropy.coordinates import BaseCoordinateFrame, SkyCoord
from astropy.table import Column, QTable, Table
from astropy.utils.decorators import lazyproperty

# LOCAL
from trackstream._type_hints import CoordinateType, FrameLikeType
from trackstream.core import StreamTrack, TrackStream
from trackstream.utils.coord_utils import resolve_framelike
from trackstream.utils.descriptors import InstanceDescriptor
from trackstream.utils.path import path_moments

##############################################################################
# PARAMETERS

_ABC_MSG = "Can't instantiate abstract class {} with abstract method {}"

##############################################################################
# CODE
##############################################################################


class StreamBase(metaclass=ABCMeta):
    """Abstract base class for streams."""

    @abstractmethod
    @property
    def _data_max_lines(self) -> int:
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "_data_max_lines"))

    @abstractmethod
    @property
    def data(self) -> BaseCoordinateFrame:
        """The stream data."""
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "data"))

    @abstractmethod
    @property
    def frame(self) -> BaseCoordinateFrame:
        """The coordinate frame of the data."""
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "frame"))

    @abstractmethod
    @property
    def name(self) -> BaseCoordinateFrame:
        """The name of the stream."""
        raise TypeError(_ABC_MSG.format(self.__class__.__qualname__, "name"))

    # ===============================================================

    def _base_repr_(self, max_lines: Optional[int] = None) -> str:
        """Mirroring implementation in astropy Table."""
        header: str = super().__repr__()
        frame: str = repr(self.frame)
        name: str = getattr(self, "full_name", self.name)  # name, with fallback

        datarep: str = self.data._base_repr_(
            html=False,
            max_width=None,
            max_lines=max_lines,
        )
        table: str = "\n\t".join(datarep.split("\n")[1:])

        return header + "\n  name: " + name + "\n  Frame:\n\t" + frame + "\n  Data:\n\t" + table

    def __repr__(self) -> str:
        return self._base_repr_(max_lines=self._data_max_lines)


# ===================================================================


class StreamArmDescriptor(InstanceDescriptor[Stream], StreamBase):
    @lazyproperty
    def name(self) -> str:
        attr_name = list(filter(None, re.split("(\d+)", self._parent_attr)))
        # e.g. arm1 -> ["arm", "1"]
        return " ".join(attr_name)

    @lazyproperty
    def full_name(self) -> str:
        parent_name = pn if isinstance(pn := self._parent.name, str) else "Stream"
        return " ".join((parent_name + ",", self.name))

    @property
    def index(self) -> Column:
        """Boolean array of which stars are in this arm."""
        tailcolumn: Column = self._parent.data["tail"]
        return tailcolumn == self._parent_attr

    @lazyproperty
    def has_data(self) -> bool:
        """Boolean of whether this arm has data."""
        return any(self.index)

    @property
    def data(self) -> QTable:
        """Return subset of full stream table that is for this arm."""
        if not self.has_data:
            raise Exception(f"{self._parent_attr} has no data")
        return self._parent.data[self.index]

    @property
    def coords(self) -> SkyCoord:
        """The coordinates of the arm."""
        arm: SkyCoord
        arm = self._parent.coords[self.index]
        return arm

    @property
    def frame(self) -> BaseCoordinateFrame:
        return self._parent.frame

    @lazyproperty
    def _data_max_lines(self) -> int:
        """Maximum number of lines in the Table to print."""
        data_max_lines = self._parent._data_max_lines
        return data_max_lines


# ===================================================================


class Stream:
    """A Stellar Stream.

    Parameters
    ----------
    data : `~astropy.table.Table`

    origin : `~astropy.coordinates.ICRS`
        The origin point of the stream (and rotated reference frame).

    data_err : `~astropy.table.QTable` (optional)
        The data_err must have (at least) column names
        ["x_err", "y_err", "z_err"]

    frame : `~astropy.coordinates.BaseCoordinateFrame` or None (optional, keyword-only)
        The stream frame. Locally linearizes the data.
        If None (default), need to fit for the frame.
    """

    arm1 = StreamArmDescriptor()
    arm2 = StreamArmDescriptor()

    # ===============================================================

    _name: Optional[str]
    _origin: SkyCoord
    _system_frame: Optional[BaseCoordinateFrame]
    _cache: Dict[str, Any]
    _original_coord: Optional[SkyCoord]
    _data: QTable
    _data_max_lines: int
    _tracker: TrackStream

    def __init__(
        self,
        data: QTable,
        origin: BaseCoordinateFrame,
        data_err: Optional[Table] = None,
        *,
        frame: Optional[FrameLikeType] = None,
        name: Optional[str] = None,
    ) -> None:
        self._name = name

        # system attributes
        self._origin = SkyCoord(origin, copy=False)
        self._system_frame = resolve_framelike(frame) if frame is not None else frame

        self._cache = dict()  # TODO! improve
        self._original_coord = None  # set _normalize_data

        # ---- Process the data ----
        # processed data
        self._data: QTable = self._normalize_data(data)
        self._data_max_lines: int = 10

        # ---- fitting the data ----
        self._tracker = TrackStream()

    # -----------------------------------------------------

    @property
    def name(self) -> Optional[str]:
        """The name of the stream."""
        return self._name

    @property
    def origin(self) -> SkyCoord:
        """Origin in stream frame."""
        frame = self.frame if self.frame is not None else self.data_frame
        return self._origin.transform_to(frame)

    @property
    def system_frame(self) -> Optional[BaseCoordinateFrame]:
        """A system-centric frame.

        Determined from the argument ``frame`` at initialization.
        If None (default) and the method ``fit`` has been called,
        then a system frame has been found and cached.
        """
        frame: Optional[BaseCoordinateFrame]
        if self._system_frame is not None:
            frame = self._system_frame
        else:
            frame = self._cache.get("frame")  # Can be `None`

        return frame

    @property
    def frame(self) -> BaseCoordinateFrame:
        """Alias for ``system_frame``."""
        return self.system_frame

    @property
    def data(self) -> QTable:
        """Data `astropy.table.QTable`."""
        return self._data

    @lazyproperty
    def number_of_tails(self) -> int:
        """Number of tidal tails.

        Returns
        -------
        number_of_tails : int
            There can only be 1, or 2 tidal tails.
        """
        n: int = 2 if (self.arm1.has_data and self.arm2.has_data) else 1
        return n

    @property
    def coords(self) -> SkyCoord:
        """Coordinates."""
        frame = self.system_frame if self.system_frame is not None else self.data_frame
        return self.data_coords.transform_to(frame)

    # ===============================================================

    @property
    def data_coords(self) -> SkyCoord:
        """Get ``coord`` from data table."""
        return self.data["coord"]

    @property
    def data_frame(self) -> BaseCoordinateFrame:
        """The frame of the data."""
        frame: BaseCoordinateFrame = self.data_coords.frame.replicate_without_data()
        return frame

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
        # TODO? selective, or just copy over? also, deepcopy?
        data.meta = original.meta.copy()

        return data

    def _normalize_data_probability(
        self,
        original: Table,
        data: QTable,
        default_weight: Union[float, u.Quantity] = 1.0,
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
        # ----------
        # tail label
        # TODO!!! better

        data["tail"] = original["tail"]

        # ----------
        # 1) the data

        # First look for a column "coord"
        if "coord" in original.colnames:
            osc = SkyCoord(original["coord"], copy=False)
        else:
            osc = SkyCoord.guess_from_table(original)

        self._original_coord = osc

        # Convert frame and representation type
        frame = (
            self.system_frame
            if self.system_frame is not None
            else osc.frame.replicate_without_data()
        )
        sc = osc.transform_to(frame)
        sc.representation_type = frame.representation_type

        # it's now clean and can be added
        data["coord"] = sc

        # Also store the components
        component_names = list(sc.get_representation_component_names("base").keys())
        #         self._component_names = component_names
        #         if "s" in sc.data.differentials:
        #             self._component_names.extend(sc.get_representation_component_names("s").keys())
        #
        #         for n in self._component_names:
        #             data[n] = getattr(sc, n)

        # ----------
        # 2) the error
        # TODO! want errors in frame of the data
        # import gala.coordinates as gc
        # cov = np.array([[1, 0],
        #                 [0, 1]])
        # gc.transform_pm_cov(sc.icrs, np.repeat(cov[None, :], len(sc), axis=0),
        #                     coord.Galactic())

        for n in component_names:  # transfer
            dn = n + "_err"
            if dn in original.colnames:
                data[dn] = original[dn]
            else:
                data[dn] = 0 * getattr(sc, n)  # assume error is 0

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
            data["SOM"] = -1  # sentinel value

    # ===============================================================
    # Fitting

    def fit_frame(self, *, force: bool = False, **kw: Any) -> BaseCoordinateFrame:
        """Fit a frame to the data."""
        if self._system_frame is not None:
            raise Exception("a system frame was given at initialization.")
        elif self.system_frame is not None and not force:
            raise Exception("already fit. use ``force`` to re-fit.")

        frame, frame_fit = self._tracker._fit_rotated_frame(self, **kw)

        self._cache["frame"] = frame
        return frame

    @property
    def track(self) -> StreamTrack:
        """Stream track.

        Raises
        ------
        ValueError
            If track is not fit.
        """
        track: Optional[StreamTrack] = self._cache.get("track")
        if track is None:
            raise ValueError("need to fit track.")
        return track

    def fit_track(self, *, force: bool = False, **kwargs: Any) -> StreamTrack:
        """Make a stream track.

        Parameters
        ----------
        force : bool, optional keyword-only
            Whether to force a fit, even if already fit.
        **kwargs
            Passed to :meth:`trackstream.TrackStream.fit`.

        Returns
        -------
        `trackstream.StreamTrack`
        """
        if not force and "track" in self._cache:
            raise Exception("already fit. use ``force`` to re-fit.")

        track: StreamTrack = self._tracker.fit(self, **kwargs)
        self._cache["track"] = track

        # Add SOM ordering to data
        self.data["SOM"] = np.empty(len(self.data), dtype=int)
        self.data["SOM"][self.arm1.index] = self._tracker._cache["arm1_visit_order"]
        if self.arm2.has_data:
            self.data["SOM"][self.arm2.index] = self._tracker._cache["arm2_visit_order"]

        return track

    # ===============================================================
    # Math on the Track (requires fitting track)

    def predict_track(
        self,
        affine: Optional[u.Quantity] = None,
        *,
        angular: bool = False,
    ) -> path_moments:
        """
        Parameters
        ----------
        affine : |Quantity| or None, optional
        angular : bool, optional keyword-only

        Returns
        -------
        `trackstream.utils.path.path_moments`
        """
        return self.track()

    # ===============================================================
    # Misc

    def _base_repr_(self, max_lines: Optional[int] = None) -> str:
        """Mirroring implementation in astropy Table."""
        header: str = super().__repr__()
        frame: str = repr(self.frame)

        datarep: str = self.data._base_repr_(
            html=False,
            max_width=None,
            max_lines=max_lines,
        )
        table: str = "\n\t".join(datarep.split("\n")[1:])

        return header + "\n  Frame:\n\t" + frame + "\n  Data:\n\t" + table

    def __repr__(self) -> str:
        s: str = self._base_repr_(max_lines=self._data_max_lines)
        return s
