# -*- coding: utf-8 -*-

"""Core Functions."""

__all__ = [
    "Stream",
]


##############################################################################
# IMPORTS

# STDLIB
import typing as T
import warnings

# THIRD PARTY
import astropy.coordinates as coord
import astropy.units as u
import numpy as np
from astropy import table
from astropy.table import QTable, Table
from astropy.utils.decorators import lazyproperty
from astropy.utils.metadata import MetaAttribute, MetaData
from astropy.utils.misc import indent
from scipy.linalg import block_diag

# LOCAL
from . import _type_hints as TH
from trackstream.utils._framelike import resolve_framelike
from trackstream.utils.path import Path
from trackstream.utils.utils import intermix_arrays

##############################################################################
# CODE
##############################################################################


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

    # -----------------------------------------------------

    def __init__(
        self,
        data: table.QTable,
        origin: TH.FrameType,
        data_err: T.Optional[table.Table] = None,
        *,
        frame: T.Optional[TH.CoordinateType] = None,
    ):
        super().__init__()
        # system attributes
        self.origin: TH.SkyCoordType = coord.SkyCoord(origin, copy=False)
        self._system_frame: T.Optional[TH.FrameType] = frame

        self._cache = dict()  # TODO! improve

        # ----------
        # process the data

        # seed values set in functions
        self._original_data: TH.SkyCoordType = None

        # processed data -> QTable
        self.data = self._normalize_data(data)

    # /def

    @classmethod
    def from_arms(
        cls,
        arm1: table.Table,
        arm2: table.Table,
        origin: TH.FrameType,
        data_err: table.QTable = None,
        *,
        frame: T.Optional[TH.FrameType] = None,
    ) -> "Stream":
        """Data from arms.

        .. todo:: TODO!

        Parameters
        ----------
        arm1, arm2 : `~astropy.table.Table`
            The stream data.

        origin : `~astropy.coordinates.ICRS`
            The origin point of the rotated reference frame.

        data_err : `~astropy.table.QTable` (optional)
            The data_err must have (at least) column names
            ["x_err", "y_err", "z_err"]

        frame : frame-like or None (optional, keyword-only)
            The stream frame. Locally linearizes the data.
            If not None, need to fit for the frame (default).

        Returns
        -------
        stream : `~trackstream.core.Stream`

        """
        raise NotImplementedError("TODO!")  # TODO!

        self = super().__new__(cls)
        data = None

        # Step 1) Convert to Representation, storing the Frame type
        if isinstance(data, (coord.BaseCoordinateFrame, coord.SkyCoord)):
            self._data_frame = data.replicate_without_data()
            self._data_rep = data.representation_type
            self._data = data.data

        elif isinstance(data, table.Table):
            _data = coord.SkyCoord.guess_from_table(data)
            self._data_frame = _data.replicate_without_data()
            self._data_rep = _data.representation_type
            self._data = _data.data

        else:
            raise TypeError(f"`data` type {type(data)} is wrong.")

        err_cols = ["x_err", "y_err", "z_err"]

        # first munge the data
        if data_err is not None:
            pass
        # try to extract from data table
        elif isinstance(data, Table) and set(err_cols).issubset(data.colnames):
            data_err = data[err_cols]
        else:
            warnings.warn(
                "No data_err and cannot extract from `data`." "Assuming errors of 0.",
            )
            data_err = table.QTable(np.zeros((len(data), 3)), names=err_cols)

        if isinstance(data_err, Table):
            data_err = data_err[err_cols]  # extract columns
        elif isinstance(data_err, coord.BaseCoordinateFrame):
            if not isinstance(data_err.data, coord.CartesianRepresentation):
                raise TypeError("data_err must be Cartesian.")
            rep = data_err.data
            data_err = table.QTable(
                dict(x_err=rep.x, y_err=rep.y, z_err=rep.z),
            )
        elif isinstance(data_err, coord.BaseRepresentation):
            if not isinstance(data_err, coord.CartesianRepresentation):
                raise TypeError("data_err must be Cartesian.")
            data_err = table.QTable(
                dict(x_err=rep.x, y_err=rep.y, z_err=rep.z),
            )
        else:
            raise TypeError(f"`data_err` type <{type(data_err)}> is wrong.")

    # /def

    # -----------------------------------------------------

    @property
    def system_frame(self) -> coord.BaseCoordinateFrame:
        """A system-centric frame.

        Determined from the argument ``frame`` at initialization.
        If None (default) and the method ``fit`` has been called,
        than a system frame has been found and cached.

        """
        frame: coord.BaseCoordinateFrame = (
            self._system_frame if self._system_frame is not None else self._cache.get("frame", None)
        )
        return frame

    # /def

    @property
    def frame(self) -> coord.BaseCoordinateFrame:
        """Alias for ``system_frame``."""
        return self.system_frame

    # /def

    @lazyproperty
    def number_of_tails(self) -> int:
        """Number of tidal tails.

        Returns
        -------
        number_of_tails : int
            There can only be 1, or 2 tidal tails.

        """
        return 2 if (self.has_arm1 and self.has_arm2) else 1

    # /def

    @lazyproperty
    def has_arm1(self) -> bool:
        # get flags from data
        flags: table.Column = np.unique(self.data["tail"])
        has_arm: bool = "arm_1" in flags
        return has_arm

    # /def

    @lazyproperty
    def has_arm2(self) -> bool:
        # get flags from data
        flags: table.Column = np.unique(self.data["tail"])
        has_arm: bool = "arm_2" in flags
        return has_arm

    # /def

    @lazyproperty
    def coords(self) -> coord.SkyCoord:
        """Coordinates."""
        frame = self.system_frame if self.system_frame is not None else self.data_frame
        return self.data_coords.transform_to(frame)

    # /def

    @property
    def coords_arm1(self) -> coord.SkyCoord:
        """The coordinates of the first arm.

        If the data isn't labelled, everything is assumed to be in this arm.

        """
        arm1: coord.SkyCoord
        if self.number_of_tails == 2:
            arm1 = self.coords[self.data["tail"] == "arm_1"]
        elif not self.has_arm2:
            arm1 = self.coords

        return arm1

    # /def

    @property
    def coords_arm2(self) -> T.Optional[coord.SkyCoord]:
        arm2: T.Optional[coord.SkyCoord] = None
        if self.has_arm2:
            arm2: coord.SkyCoord = self.coords[self.data["tail"] == "arm_2"]

        return arm2

    # /def

    # ===============================================================

    @property
    def original_data(self) -> TH.FrameType:
        """Original data as passed."""
        # TODO! don't make new one each time
        return self._data_frame.realize_frame(self._data)

    # /def

    @property
    def data_coords(self) -> TH.SkyCoordType:
        """Get ``coord`` from data table."""
        return self.data["coord"]

    # /def

    @property
    def data_frame(self) -> TH.FrameType:
        """The frame of the data."""
        return self.data["coord"].frame.replicate_without_data()

    # /def

    @property
    def data_representation_type(self) -> TH.RepresentationType:
        """The representation type of the data."""
        return self.data["coord"].representation_type

    # /def

    # -----------------------------------------------------
    # Data normalization

    def _normalize_data(self, original: table.Table) -> table.QTable:
        """Normalize data table.

        Just calls other functions.

        .. todo::

            allow argument specifying column names.

        Parameters
        ----------
        original : :class:`~astropy.table.Table`

        Returns
        -------
        data : :class:`~astropy.table.QTable`

        """
        data = QTable()

        # 1) data probability
        self._normalize_data_probability(original, data, default_weight=1)

        # 2) coordinates. `data` modded in-place
        self._normalize_data_coordinates(original, data)

        # --------------
        # Metadata

        # TODO? selective, or just copy over?
        data.meta = original.meta.copy()  # TODO? deepcopy?

        return data

    # /def

    def _normalize_data_probability(
        self,
        original: table.Table,
        data: TH.QTableType,
        default_weight: T.Union[float, TH.QuantityType] = 1.0,
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

    # /def

    def _normalize_data_coordinates(
        self,
        original: table.Table,
        data: TH.QTableType,
    ):
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

    # /def


# /class


##############################################################################
# END
