"""Normalize data."""

from __future__ import annotations

from collections.abc import Mapping
import copy
from dataclasses import dataclass

from astropy.coordinates import BaseCoordinateFrame, SkyCoord
from astropy.table import MaskedColumn, QTable, Table
import astropy.units as u
from numpy import arange, ones

from trackstream.utils.coord_utils import deep_transform_to

__all__: list[str] = []

##############################################################################
# CODE
##############################################################################


@dataclass(frozen=True)
class StreamArmDataNormalizer:
    """Instance-level descriptor to normalize stream data tables.

    Attributes
    ----------
    frame : BaseCoordinateFrame or None
        The frame of the stream.

    Methods
    -------
    __call__(original, original_err)
        Run the normalizer.
    """

    frame: BaseCoordinateFrame | None

    # ===============================================================

    def __call__(self, original: Table, original_err: Table | None) -> QTable:
        """Normalize data table.

        Parameters
        ----------
        original : |Table|
            The table of data. It will be modified if not already grouped and
            labeled by the stream arm index.
        original_err : |Table| or None
            A table of the errors.

        Returns
        -------
        data : `~astropy.table.QTable`
        """
        data = QTable()  # going to be assigned in-place
        data.meta = {}

        # 0) stream arm index. `data` modded in-place.
        self._data_arm_index(original, out=data)

        # 1) data probability. `data` modded in-place.
        self._data_probability(original, out=data, default_weight=1)

        # 2) coordinates. `data` modded in-place.
        self._data_coordinates(original, original_err, out=data)

        # 3) ordering.
        self._data_order(original, out=data)

        # Metadata
        meta = copy.deepcopy(original.meta) if isinstance(original.meta, Mapping) else {}
        data.meta = {**meta, **data.meta}

        # Finally, merge original_err into data
        # TODO!
        # https://docs.astropy.org/en/stable/table/operations.html#merging-details
        # TODO! make sure `arm` matches
        if original_err is not None:
            for n in set(original_err.colnames) - {"arm"}:
                nn = f"err_{n}" if n in original.colnames else n
                if nn in data.colnames:
                    continue

                data[nn] = original_err[n]

        return data

    def _data_arm_index(self, original: Table, *, out: QTable) -> None:
        """Set stream arm index.

        Parameters
        ----------
        original : |Table|
            The original data.
        out : |QTable|, keyword-only
            The normalized data.
        """
        out["arm"] = original["arm"]

    def _data_probability(
        self,
        original: Table,
        *,
        out: QTable,
        default_weight: float | u.Quantity = 1.0,
    ) -> None:
        """Normalize data probability.

        Units of percent. Default is 100%.

        Parameters
        ----------
        original : |Table|
            The original data.
        out : |QTable|, keyword-only
            The normalized data.
        default_weight : float, optional keyword-only
            The default membership probability.
            If float, then range 0-1 maps to 0-100%.
            If has unit of percent, then unchanged.
        """
        colns = [n.lower() for n in original.colnames]  # lower-case columns

        # Case insensitive match for column names
        if "pmemb" in colns:
            index = colns.index("pmemb")
            oname = original.colnames[index]
            Pmemb = original[oname]

            meta = original.meta.get(oname, None)
        else:
            Pmemb = ones(len(original)) * default_weight  # non-scalar
            meta = None

        # Add membership probabiliy column
        # MC(Q(...)) preserves units and masks, if present.
        out["Pmemb"] = MaskedColumn(u.Quantity(Pmemb).to(u.percent))

        # Add metadata
        out.meta["Pmemb"] = meta or "Probability of stream membership."

    def _data_coordinates(
        self,
        original: Table,
        original_err: Table | None = None,
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
        out : |QTable|, keyword-only
            The normalized data.
        """
        # ----------
        # 1) the data

        # First look for a column "coords"
        osc = (
            SkyCoord(original["coords"], copy=False)
            if "coords" in original.colnames
            else SkyCoord.guess_from_table(original)
        )

        # Add coordinates to stream
        if self.frame is None:  # no new frame
            to_frame = osc.frame
            to_rep = type(osc.data)
            to_dif = type(osc.data.differentials["s"]) if "s" in osc.data.differentials else None
        else:
            to_frame = self.frame
            to_rep, to_dif = to_frame.representation_type, to_frame.differential_type

        sc: SkyCoord
        sc = deep_transform_to(osc, to_frame, to_rep, to_dif)

        # it's now clean and can be added
        out["coords"] = sc

        # ----------
        # 2) the error
        # TODO! want the ability to convert errors into the frame of the data.
        # gc.transform_pm_cov(sc.icrs, repeat(cov[None, :], len(sc), axis=0),
        #                     coord.Galactic())

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

    def _data_order(self, original: Table, *, out: QTable) -> None:
        """Normalize data ordering.

        Parameters
        ----------
        original : |Table|
            The original data.
        out : |QTable|, keyword-only
            The normalized data.
        """
        # Intra-arm ordering.
        if "order" in original.colnames:  # transfer if available.
            out["order"] = MaskedColumn(original["order"], dtype=int)
        else:  # Make, if not.
            out["order"] = MaskedColumn(-1 * ones(len(original)), dtype=int)  # sentinel value
            # pairwise iterate, making per-arm ordering
            for i, j in zip(out.groups.indices[:-1], out.groups.indices[1:], strict=True):
                out["order"][i:j] = MaskedColumn(arange(j - i), dtype=int)
