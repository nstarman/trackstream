"""SOM."""

from __future__ import annotations

from dataclasses import dataclass
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, final

import astropy.coordinates as coords
import numpy as np

from trackstream.stream.core import StreamArm
from trackstream.stream.plural import StreamArmsBase
from trackstream.track.fit.som.cartesian import CartesianSOM
from trackstream.track.fit.som.sphere import USphereSOM
from trackstream.track.fit.som.utils import _get_info_for_projection
from trackstream.track.fit.utils import _c2v, _v2c
from trackstream.track.width.core import LENGTH, SPEED, BaseWidth
from trackstream.track.width.oned.core import AngularDiffWidth, AngularWidth, Cartesian1DiffWidth, Cartesian1DWidth
from trackstream.track.width.plural import Widths

__all__: list[str] = []

if TYPE_CHECKING:
    from astropy.coordinates import SkyCoord
    from astropy.units import PhysicalType
    from numpy.random import Generator

    from trackstream._typing import NDFloating
    from trackstream.track.fit.som.base import SOM1DBase, SOMInfo
    from trackstream.track.width.base import WidthBase


#####################################################################


@final
@dataclass(frozen=True)
class SelfOrganizingMap:
    """Self-Organizing Map.

    Parameters
    ----------
    som : `~trackstream.track.fit.som.base.SOM1DBase`
        Low-level SOM implementation, operating on `~numpy.ndarray`.
    frame : `~astropy.coordinates.BaseCoordinateFrame`
        The frame in which to build the SOM. Data is transformed into this frame
        before the SOM is fit.
    origin : `~astropy.coordinates.SkyCoord` or None
        The origin.
    """

    som: SOM1DBase
    frame: coords.BaseCoordinateFrame
    origin: coords.SkyCoord | None

    @property
    def info(self) -> SOMInfo:
        """Return `trackstream.track.fit.som.base.SOMInfo`.

        Desscribes how an ``.som`` interacts with ``.frame``.
        """
        return self.som.info

    # ===============================================================

    @singledispatchmethod
    @classmethod
    def from_format(  # noqa: PLR0913
        cls,
        arm: object,  # noqa: ARG003
        /,
        *,
        onsky: bool,  # noqa: ARG003
        nlattice: int | None = None,  # noqa: ARG003
        sigma: float = 0.1,  # noqa: ARG003
        learning_rate: float = 0.3,  # noqa: ARG003
        rng: Generator | int | None = None,  # noqa: ARG003
        prototype_kw: dict[str, Any] | None = None,  # noqa: ARG003
    ) -> Any:  # https://github.com/python/mypy/issues/11727
        """Make Self-Organiizing Map from data.

        Parameters
        ----------
        arm : object
            The object from which the SOM is built. Single-dispatched on this
            object.
        onsky : bool
            Whether the SOM should be run on-sky or 3D.
        nlattice : int | None, optional
            Number of lattice points / prototypes, by default `None`.
        sigma : float, optional
            Spread of the neighborhood function, needs to be adequate to the
            dimensions of the map. (at the iteration `t` we have ``sigma(t) =
            sigma / (1 + t/T)``)
        learning_rate : float, optional
            At the iteration ``t`` we have ``learning_rate(t) = learning_rate /
            (1 + t/T)``.
        rng : Generator | int | None, optional
            Random generator, or seed thereof.
        prototype_kw : dict[str, Any], optional
            Keyword arguments into initial prototypes algorithm. The exact implementation depends on ``onsky``.

        Returns
        -------
        Any
            Output type depends on the dispatch implementation.

        Raises
        ------
        NotImplementedError
            If there is no dispatched method.
        """
        msg = "not dispatched"
        raise NotImplementedError(msg)

    @from_format.register(StreamArm)
    @classmethod
    def _from_format_streamarm(  # noqa: PLR0913
        cls,
        arm: StreamArm,
        /,
        *,
        onsky: bool,
        kinematics: bool | None = None,
        nlattice: int | None = None,
        sigma: float = 0.1,
        learning_rate: float = 0.3,
        rng: Generator | int | None = None,
        prototype_kw: dict[str, Any] | None = None,
    ) -> SelfOrganizingMap:
        if arm.frame is None:
            # LOCAL
            from trackstream.stream.base import FRAME_NONE_ERR

            raise FRAME_NONE_ERR

        SOM = USphereSOM if onsky else CartesianSOM

        som = SOM.from_format(
            arm,
            nlattice=nlattice,
            sigma=sigma,
            learning_rate=learning_rate,
            rng=rng,
            prototype_kw=prototype_kw,
            kinematics=kinematics,
        )

        return cls(som=som, frame=arm.frame, origin=arm.origin)

    @from_format.register(StreamArmsBase)
    @classmethod
    def _from_format_streamarmsbase(  # noqa: PLR0913
        cls,
        arms: StreamArmsBase,
        /,
        *,
        onsky: bool,
        kinematics: bool | None = None,
        nlattice: int | None = None,
        sigma: float = 0.1,
        learning_rate: float = 0.3,
        rng: Generator | int | None = None,
        prototype_kw: dict[str, Any] | None = None,
    ) -> dict[str, SOM1DBase]:
        out: dict[str, SOM1DBase] = {}
        for k, arm in arms.items():
            out[k] = cls.from_format(
                arm,
                onsky=onsky,
                kinematics=kinematics,
                nlattice=nlattice,
                sigma=sigma,
                learning_rate=learning_rate,
                rng=rng,
                prototype_kw=prototype_kw,
            )
        return out

    # ===============================================================

    @property
    def nlattice(self) -> int:
        """Number of lattice points."""
        return self.som.nlattice

    @property
    def nfeature(self) -> int:
        """Number of features, e.g. dimensions."""
        return self.som.nfeature

    @property
    def prototypes(self) -> coords.BaseCoordinateFrame:
        """Read-only view of prototypes vectors as a |Frame|."""
        p = self.som.prototypes.view()
        p.flags.writeable = False
        return _v2c(self, p)

    @property
    def init_prototypes(self) -> coords.BaseCoordinateFrame:
        """Read-only view of prototypes vectors as a |Frame|."""
        p = self.som.init_prototypes.view()
        p.flags.writeable = False
        return _v2c(self, p)

    @property
    def onsky(self) -> bool:
        """Whether the SOM is run on-sky or in 3D space."""
        return bool(isinstance(self.som, USphereSOM))

    @property
    def kinematics(self) -> bool:
        """Whether the SOM is run with kinematic information."""
        # two options for number of features
        soms = (CartesianSOM, USphereSOM)
        nf = (6, 4)  # number of features
        i = soms.index(type(self.som))
        return bool(self.som.nfeature == nf[i])

    # ===============================================================

    def fit(
        self,
        data: SkyCoord,
        num_iteration: int = 100_000,
        *,
        random_order: bool = False,
        progress: bool = True,
    ) -> None:
        """Fit the SOM to the data.

        Parameters
        ----------
        data : SkyCoord
            Fit the data.
        num_iteration : int, optional
            Number of iterations when fitting, by default 10^5.
        random_order : bool, optional
            Whether to introduce data in order (`False`, default), or randomly
            (`True`).
        progress : bool, optional
            Whether to show a progress bar (`True`, default).

        Returns
        -------
        None
        """
        v = _c2v(self, data)
        return self.som.fit(v, num_iteration=num_iteration, random_order=random_order, progress=progress)

    def predict(self, data: SkyCoord, /) -> tuple[SkyCoord, NDFloating]:
        """Predict projection and ordering.

        Parameters
        ----------
        data : SkyCoord
            Data.

        Returns
        -------
        SkyCoord
            Projection (ordered).
        ndarray[int]
            ordering.
        """
        v = _c2v(self, data)
        projv, ordering = self.som.predict(v)

        projdata = coords.SkyCoord(_v2c(self, projv), copy=False)

        if self.origin is not None:
            # the visit order can be backward so need to detect proximity to origin
            armep = data[ordering[[0, -1]]]  # end points

            sep = armep.separation(self.origin) if self.onsky else armep.separation_3d(self.origin)

            if np.argmin(sep) == 1:  # End point is closer, flip order
                ordering = ordering[::-1]
                projdata = projdata[::-1]

        return projdata, ordering

    def fit_predict(
        self,
        data: SkyCoord,
        /,
        num_iteration: int = 100_000,
        *,
        random_order: bool = False,
        progress: bool = True,
    ) -> tuple[coords.SkyCoord, NDFloating]:
        """Fit and predict.

        Parameters
        ----------
        self : SOM1DBase
            Self.
        data : SkyCoord
            Data.
        num_iteration : int, optional
            Number of iterations when fitting, by default 10^5.

        random_order : bool, optional keyword-only
            Whether to introduce data in order (`False`, default), or randomly
            (`True`).
        progress : bool, optional keyword-only
            Whether to show a progress bar (`True`, default).

        Returns
        -------
        SkyCoord, ndarray[int]
            Projection (ordered) and ordering.
        """
        self.fit(data, num_iteration=num_iteration, random_order=random_order, progress=progress)
        projdata, order = self.predict(data)
        return projdata, order

    # ===============================================================

    def separation(self, data: SkyCoord, /) -> Widths[WidthBase]:
        """Compute orthogonal distances to SOM.

        Parameters
        ----------
        data : |SkyCoord|, positional-only
            Data.

        Returns
        -------
        Widths
            With keys ``length`` and ``speed``.
        """
        # Prep
        qp = _c2v(self, data)
        D = qp.shape[1]
        units = tuple(self.info.units.values())  # from _c2v
        ws: dict[PhysicalType, BaseWidth] = {}

        # Positions:
        iq = slice(0, D) if not self.kinematics else slice(0, D // 2)
        _, _, _, distances = _get_info_for_projection(qp[:, iq], self.som.prototypes[:, iq])
        ind_best_distance = np.argmin(distances, axis=1)
        orth_distance = distances[(np.arange(len(distances))), (ind_best_distance)]
        wcls = AngularWidth if self.onsky else Cartesian1DWidth
        ws[LENGTH] = wcls(orth_distance * units[0][0])

        # Velocities:
        if self.kinematics:
            wdcls = AngularDiffWidth if self.onsky else Cartesian1DiffWidth
            ip = slice(D // 2, None)

            _, _, _, distances = _get_info_for_projection(qp[:, ip], self.som.prototypes[:, ip])

            ind_best_distance = np.argmin(distances, axis=1)
            orth_distance = distances[(np.arange(len(distances))), (ind_best_distance)]
            ws[SPEED] = wdcls(orth_distance * units[1][0])

        return Widths(ws)
