# -*- coding: utf-8 -*-

"""Core Functions."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import copy
from typing import TYPE_CHECKING, List, Mapping, Optional, Type, TypeVar, Union, cast

# THIRD PARTY
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable, Table
from numpy import arange, ones

# LOCAL
from trackstream.utils.descriptors import InstanceDescriptor

if TYPE_CHECKING:
    # LOCAL
    from trackstream.stream.core import Stream  # noqa: F401

__all__: List[str] = []

##############################################################################
# PARAMETERS

StreamT = TypeVar("StreamT", bound="Stream")

##############################################################################
# CODE
##############################################################################


class StreamDataNormalizer(InstanceDescriptor[StreamT]):
    """Instance-level descriptor to normalize stream data tables.

    Methods
    -------
    run(original, original_err)
        Run the normalizer.
    """

    def __get__(self, obj: StreamT, _: Optional[Type[StreamT]]) -> StreamDataNormalizer:
        if obj is None:
            raise AttributeError(f"{type(self).__name__} can only be called from an instance")

        return super().__get__(obj, _)

    # ===============================================================

    def run(self, original: Table, original_err: Optional[Table]) -> QTable:
        """Normalize data table.

        Parameters
        ----------
        original : |Table|
        original_err : |Table| or None

        Returns
        -------
        data : :class:`~astropy.table.QTable`
        """
        data = QTable()  # going to be assigned in-place

        # 1) data probability. `data` modded in-place
        self._data_probability(original, out=data, default_weight=1)

        # 2) stream arm labels. `data` modded in-place
        self._data_arm(original, out=data)

        # 3) coordinates. `data` modded in-place
        self._data_coordinates(original, original_err, out=data)

        # 4) SOM ordering
        self._data_arm_index(original, out=data)

        # Metadata
        meta = copy.deepcopy(original.meta)
        data.meta = {**meta, **data.meta}

        return data

    def _data_probability(
        self, original: Table, *, out: QTable, default_weight: Union[float, u.Quantity] = 1.0
    ) -> None:
        """Data probability. Units of percent. Default is 100%.

        Parameters
        ----------
        original : |Table|
            The original data.

        out : |QTable|, keyword-only
            The normalized data.
        default_weight : float, optional keyword-only
            The default membership probability.
            If float, then range 0-1 maps to 0-100%.
            If has unit of percent, then unchanged

        Returns
        -------
        None
        """
        colns = [n.lower() for n in original.colnames]  # lower-case columns

        # Case insensitive match for column names
        if "pmemb" in colns:
            index = colns.index("pmemb")
            oname = original.colnames[index]
            Pmemb = original[oname]

            meta = cast(Mapping, original.meta).get(oname, None)
        else:
            Pmemb = ones(len(original)) * default_weight  # non-scalar
            meta = None

        # Add membership probabiliy column
        out["Pmemb"] = u.Quantity(Pmemb).to(u.percent)

        # Add metadata
        out.meta["Pmemb"] = meta or "Probability of stream membership."  # type: ignore

    def _data_arm(self, original: Table, *, out: QTable) -> None:
        """Parse the tail labels.

        Parameters
        ----------
        original : |Table|
            The original data.

        out : |QTable|
            The stream data.

        Returns
        -------
        None
        """
        # TODO!!! better
        out["tail"] = original["tail"]

        # group and add index
        out = out.group_by("tail")
        out.add_index("tail")

        # add metadata
        # out.meta["tail"] =

    def _data_coordinates(
        self, original: Table, original_err: Optional[Table] = None, *, out: QTable
    ) -> None:
        """Parse the data table.

        - the frame is stored in ``_data_frame``
        - the representation is stored in ``_data_rep``
        - the original data representation  is in ``_data``

        Parameters
        ----------
        original : |Table|
            The original data.
        original_err : |Table| or None, optional
            The error in the original data.

        out : |QTable|
            The stream data.

        Returns
        -------
        None
        """
        stream = self._enclosing

        # ----------
        # 1) the data

        # First look for a column "coord"
        if "coord" in original.colnames:
            osc = SkyCoord(original["coord"], copy=False)
        else:
            osc = SkyCoord.guess_from_table(original)

        # add coordinates to stream
        stream._original_coord = osc

        # get the backup frame & representation type
        osc_frame = osc.frame.replicate_without_data()
        osc_frame.representation_type = osc.representation_type

        # Convert frame and representation type
        frame = stream.system_frame if stream.system_frame is not None else osc_frame
        sc = osc.transform_to(frame)
        sc.representation_type = frame.representation_type

        # it's now clean and can be added
        out["coord"] = sc

        # Also store the components
        component_names = list(sc.get_representation_component_names("base").keys())

        # ----------
        # 2) the error
        # TODO! want the ability to convert errors into the frame of the data.
        # import gala.coordinates as gc
        # cov = array([[1, 0], [0, 1]])
        # gc.transform_pm_cov(sc.icrs, repeat(cov[None, :], len(sc), axis=0),
        #                     coord.Galactic())

        # the error is stored on either the original data table, or in a separate table.
        orig_err = original if original_err is None else original_err
        # Iterate over the components, getting the error
        n: str
        for n in component_names:
            dn: str = n + "_err"  # error component name
            # either get the error, or set it to zero.
            out[dn] = orig_err[dn] if dn in orig_err.colnames else 0 * getattr(sc, n)

    def _data_arm_index(self, original: Table, *, out: QTable) -> None:
        """Data probability. Units of percent. Default is 100%.

        Parameters
        ----------
        original : |Table|
            The original data.
        out : |QTable|, optional keyword-only
            The normalized data.

        Returns
        -------
        None
        """
        if "order" in original.colnames:
            out["order"] = original["order"]
        else:
            out["order"] = arange(len(original))  # read order
