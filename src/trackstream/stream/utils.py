# -*- coding: utf-8 -*-

"""Core Functions."""

##############################################################################
# IMPORTS

from __future__ import annotations

# STDLIB
import copy
from typing import TYPE_CHECKING, List, Mapping, Optional, Union, cast

# THIRD PARTY
import astropy.units as u
from astropy.coordinates import SkyCoord
from astropy.table import QTable, Table
from attrs import define, field
from numpy import arange, ones

# LOCAL
from trackstream.utils.coord_utils import deep_transform_to

if TYPE_CHECKING:
    # LOCAL
    from trackstream.stream.core import StreamArm  # noqa: F401

__all__: List[str] = []

##############################################################################
# CODE
##############################################################################


@define(frozen=True)
class StreamArmDataNormalizer:
    """Instance-level descriptor to normalize stream data tables.

    Methods
    -------
    run(original, original_err)
        Run the normalizer.
    """

    original_coord: Optional[SkyCoord] = field(init=False, default=None, repr=False)

    # ===============================================================

    def __call__(
        self, stream: "StreamArm", original: Table, original_err: Optional[Table]
    ) -> QTable:
        """Normalize data table.

        Parameters
        ----------
        original : |Table|
            The table of data. It will be modified if not alrady grouped and
            labeled by the stream arm index.
        original_err : |Table| or None
            A table of the errors.

        Returns
        -------
        data : :class:`~astropy.table.QTable`
        """
        data = QTable()  # going to be assigned in-place

        # 2) data probability. `data` modded in-place
        self._data_probability(stream, original, out=data, default_weight=1)

        # 3) coordinates. `data` modded in-place
        self._data_coordinates(stream, original, original_err, out=data)

        # 4) ordering
        self._data_index(stream, original, out=data)

        # Metadata
        meta = copy.deepcopy(original.meta)
        data.meta = {**meta, **data.meta}

        return data

    def _data_probability(
        self,
        _: "StreamArm",
        original: Table,
        *,
        out: QTable,
        default_weight: Union[float, u.Quantity] = 1.0,
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

        # # modify printing  # TODO!
        # out.formatter

    def _data_coordinates(
        self,
        stream: "StreamArm",
        original: Table,
        original_err: Optional[Table] = None,
        *,
        out: QTable,
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
        # ----------
        # 1) the data

        # First look for a column "coord"
        if "coord" in original.colnames:
            osc = SkyCoord(original["coord"], copy=False)
        else:
            osc = SkyCoord.guess_from_table(original)

        # add coordinates to stream
        object.__setattr__(self, "original_coord", osc)

        if stream.system_frame is None:  # no new frame
            sc = osc
        elif stream.cache["system_frame"] is not None:  # previously fit
            sc = osc
        else:
            sc = deep_transform_to(
                osc,
                stream.system_frame,
                stream.system_frame.representation_type,
                stream.system_frame.differential_type if "s" in osc.data.differentials else None,
            )

        # it's now clean and can be added
        out["coord"] = sc

        # ----------
        # 2) the error
        # TODO! want the ability to convert errors into the frame of the data.
        # import gala.coordinates as gc
        # cov = array([[1, 0], [0, 1]])
        # gc.transform_pm_cov(sc.icrs, repeat(cov[None, :], len(sc), axis=0),
        #                     coord.Galactic())

        # Also store the components
        component_names = list(sc.get_representation_component_names("base").keys())
        if "s" in sc.data.differentials:  # detect kinematics
            component_names += list(sc.get_representation_component_names("s").keys())

        # the error is stored on either the original data table, or in a separate table.
        orig_err = original if original_err is None else original_err
        # Iterate over the components, getting the error
        n: str
        for n in component_names:
            dn: str = n + "_err"  # error component name
            # either transfer the error, or set to zero.
            if dn in orig_err.colnames:
                out[dn] = orig_err[dn]
            else:
                out[dn] = 0 * getattr(sc, n)  # (get correct units)

    def _data_index(self, _: "StreamArm", original: Table, *, out: QTable) -> None:
        """Data ordering.

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
        # Intra-arm ordering.
        if "order" in original.colnames:  # transfer if available.
            out["order"] = original["order"]
        else:  # Make, if not.
            out["order"] = -1  # sentinel value
            # pairwise iterate, making per-arm ordering
            for i, j in zip(out.groups.indices[:-1], out.groups.indices[1:]):
                out["order"][i:j] = arange(j - i)

        # # Total ordering.
        # if "order" in original.colnames:
        #     out["order"] = original["order"]
        # else:
        #     out["order"] = arange(len(original))  # read order
