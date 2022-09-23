from __future__ import annotations

# STDLIB
from dataclasses import dataclass
from functools import singledispatchmethod
from typing import TYPE_CHECKING, Any, Literal, final

# THIRD PARTY
import astropy.coordinates as coords
import astropy.units as u
import numpy as np

# LOCAL
from trackstream.stream.core import StreamArm
from trackstream.stream.plural import StreamArmsBase
from trackstream.track.fit.som.cartesian import CartesianSOM
from trackstream.track.fit.som.sphere import USphereSOM
from trackstream.track.fit.som.utils import _get_info_for_projection
from trackstream.track.fit.utils import _c2v, _v2c
from trackstream.track.width.core import LENGTH, SPEED, BaseWidth
from trackstream.track.width.oned.core import (
    AngularDiffWidth,
    AngularWidth,
    Cartesian1DiffWidth,
    Cartesian1DWidth,
)
from trackstream.track.width.plural import Widths
from trackstream.utils.visualization import CommonPlotDescriptorBase, DKindT

if TYPE_CHECKING:
    # THIRD PARTY
    from matplotlib.pyplot import Axes
    from numpy.random import Generator

    # LOCAL
    from trackstream.track.fit.som.base import SOM1DBase, SOMInfo

__all__: list[str] = []


#####################################################################


@dataclass(frozen=True)
class SOMPlotDescriptor(CommonPlotDescriptorBase["SelfOrganizingMap"]):
    def current(
        self,
        kind: DKindT = "positions",
        *,
        origin: bool = True,
        connect: bool = True,
        x_offset: u.Quantity | Literal[0] = 0,
        y_offset: u.Quantity | Literal[0] = 0,
        ax: Axes | None = None,
        format_ax: bool = False,
    ) -> Axes:
        som, _ax, _ = self._setup(ax=ax)
        (x, xn), (y, yn) = self._get_xy(som.prototypes, kind=kind)

        if connect:
            _ax.plot(x + x_offset, y + y_offset, c="k")
        _ax.scatter(x + x_offset, y + y_offset, marker="P", edgecolors="black", facecolor="none")
        if origin:
            self._origin(frame=som.frame, kind=kind, ax=_ax)

        if format_ax:  # Axes settings
            self._format_ax(_ax, frame=som.frame.name, x=xn, y=yn)
        return _ax

    def initial(
        self,
        kind: DKindT = "positions",
        *,
        origin: bool = True,
        connect: bool = True,
        x_offset: u.Quantity | Literal[0] = 0,
        y_offset: u.Quantity | Literal[0] = 0,
        ax: Axes | None = None,
        format_ax: bool = False,
    ) -> Axes:
        som, _ax, _ = self._setup(ax=ax)
        (x, xn), (y, yn) = self._get_xy(som.init_prototypes, kind=kind)

        if connect:
            _ax.plot(x + x_offset, y + y_offset, c="k")
        _ax.scatter(x + x_offset, y + y_offset, marker="P", edgecolors="black", facecolor="none")
        if origin:
            self._origin(frame=som.frame, kind=kind, ax=_ax)

        if format_ax:  # Axes settings
            self._format_ax(_ax, frame=som.frame.name, x=xn, y=yn)
        return _ax

    def __call__(
        self,
        kind: DKindT = "positions",
        *,
        origin: bool = True,
        connect: bool = True,
        initial_prototypes: bool = False,
        x_offset: u.Quantity | Literal[0] = 0,
        y_offset: u.Quantity | Literal[0] = 0,
        ax: Axes | None = None,
        format_ax: bool = False,
    ) -> Axes:
        if initial_prototypes:
            _ax = self.initial(kind=kind, origin=False, connect=False, x_offset=0, y_offset=0, ax=ax, format_ax=False)
        else:
            _ax = ax
        axes = self.current(
            kind=kind, origin=origin, connect=connect, x_offset=x_offset, y_offset=y_offset, ax=_ax, format_ax=format_ax
        )
        return axes


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

    plot = SOMPlotDescriptor()

    # ===============================================================

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
    def from_format(
        cls,
        arm: object,
        /,
        onsky: bool,
        *,
        nlattice: int | None = None,
        sigma: float = 0.1,
        learning_rate: float = 0.3,
        rng: Generator | int | None = None,
        prototype_kw: dict[str, Any] = {},
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
        raise NotImplementedError("not dispatched")

    @from_format.register(StreamArm)
    @classmethod
    def _from_format_streamarm(
        cls,
        arm: StreamArm,
        /,
        onsky: bool,
        kinematics: bool | None = None,
        *,
        nlattice: int | None = None,
        sigma: float = 0.1,
        learning_rate: float = 0.3,
        rng: Generator | int | None = None,
        prototype_kw: dict[str, Any] = {},
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
    def _from_format_streamarmsbase(
        cls,
        arms: StreamArmsBase,
        /,
        onsky: bool,
        kinematics: bool | None = None,
        *,
        nlattice: int | None = None,
        sigma: float = 0.1,
        learning_rate: float = 0.3,
        rng: Generator | int | None = None,
        prototype_kw: dict[str, Any] = {},
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
        return True if isinstance(self.som, USphereSOM) else False

    @property
    def kinematics(self) -> bool:
        """Whether the SOM is run with kinematic information."""
        # two options for number of features
        soms = (CartesianSOM, USphereSOM)
        nf = (6, 4)  # number of features
        i = soms.index(type(self.som))
        return True if self.som.nfeature == nf[i] else False

    # ===============================================================

    def fit(
        self,
        data: coords.SkyCoord,
        num_iteration: int = int(1e5),
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

    def predict(self, data: coords.SkyCoord, /) -> tuple[coords.SkyCoord, np.ndarray]:
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

            # import pdb
            # import matplotlib.pyplot as plt

            # line = plt.scatter(data.x[ordering], data.y[ordering], c=np.arange(len(data)))
            # plt.scatter(armep.x[0], armep.y[0], c="red")
            # plt.scatter(armep.x[1], armep.y[1], c="green")
            # plt.scatter(self.prototypes.x, self.prototypes.y, c="k", s=2)
            # plt.colorbar(line)
            # plt.show()

            # pdb.set_trace()

            if np.argmin(sep) == 1:  # End point is closer, flip order
                ordering = ordering[::-1]
                projdata = projdata[::-1]

        return projdata, ordering

    def fit_predict(
        self,
        data: coords.SkyCoord,
        /,
        num_iteration: int = int(1e5),
        random_order: bool = False,
        progress: bool = True,
        split: int | None = None,
    ) -> tuple[coords.SkyCoord, np.ndarray]:
        self.fit(data, num_iteration=num_iteration, random_order=random_order, progress=progress)
        projdata, order = self.predict(data)
        return projdata, order

    # ===============================================================

    def separation(self, data: coords.SkyCoord, /) -> Widths:
        """Compute orthogonal distances to SOM.

        Parameters
        ----------
        data : |SkyCoord|, positional-only

        Returns
        -------
        Widths
            With keys ``length`` and ``speed``.
        """
        # Prep
        qp = _c2v(self, data)
        D = qp.shape[1]
        units = tuple(self.info.units.values())  # from _c2v
        ws: dict[u.PhysicalType, BaseWidth] = {}

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
