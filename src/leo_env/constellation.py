"""
Constellation – define and manage a LEO satellite constellation.

The default configuration mirrors the first three Starlink deployment shells
(simplified numbers suitable for simulation).  A full-scale replica is also
available via STARLINK_SHELLS.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

from .satellite import Satellite
from .utils import EARTH_RADIUS_KM, deg2rad


@dataclass
class StarLinkShellConfig:
    """
    Configuration for one orbital shell.

    Parameters
    ----------
    shell_id    : integer identifier
    altitude_km : orbital altitude above the Earth's surface
    inc_deg     : orbital inclination (degrees)
    n_planes    : number of orbital planes in this shell
    n_sats      : number of satellites per orbital plane
    """

    shell_id: int
    altitude_km: float
    inc_deg: float
    n_planes: int
    n_sats: int

    @property
    def total_sats(self) -> int:
        return self.n_planes * self.n_sats

    @property
    def semi_major_axis_km(self) -> float:
        return EARTH_RADIUS_KM + self.altitude_km


# ---------------------------------------------------------------------------
# Reference Starlink shell configurations (as filed with the FCC)
# Shell indices and numbers are simplified for simulation purposes.
# ---------------------------------------------------------------------------

STARLINK_SHELLS: List[StarLinkShellConfig] = [
    # Shell 0: 550 km, 53.0° – first operational shell
    StarLinkShellConfig(shell_id=0, altitude_km=550.0, inc_deg=53.0,  n_planes=72, n_sats=22),
    # Shell 1: 540 km, 53.2° – second shell
    StarLinkShellConfig(shell_id=1, altitude_km=540.0, inc_deg=53.2,  n_planes=72, n_sats=22),
    # Shell 2: 570 km, 70.0° – high-inclination shell for polar coverage
    StarLinkShellConfig(shell_id=2, altitude_km=570.0, inc_deg=70.0,  n_planes=36, n_sats=20),
    # Shell 3: 560 km, 97.6° – sun-synchronous polar shell
    StarLinkShellConfig(shell_id=3, altitude_km=560.0, inc_deg=97.6,  n_planes=6,  n_sats=58),
    # Shell 4: 560 km, 97.6° – additional sun-synchronous shell
    StarLinkShellConfig(shell_id=4, altitude_km=560.0, inc_deg=97.6,  n_planes=4,  n_sats=43),
]

# A reduced constellation for fast simulation / testing
STARLINK_SHELLS_MINI: List[StarLinkShellConfig] = [
    StarLinkShellConfig(shell_id=0, altitude_km=550.0, inc_deg=53.0, n_planes=6, n_sats=11),
    StarLinkShellConfig(shell_id=1, altitude_km=570.0, inc_deg=70.0, n_planes=3, n_sats=10),
]


class Constellation:
    """
    A collection of Satellite objects organised into orbital shells and planes.

    The Walker-Delta pattern is used to distribute satellites evenly:
        - RAAN spacing  = 360° / n_planes  per plane
        - Phase offset  = 360° / (n_planes * n_sats) * plane_index  (Walker-Star variant)
        - In-plane spacing = 360° / n_sats

    Parameters
    ----------
    shells : list of StarLinkShellConfig objects describing each shell
    """

    def __init__(self, shells: Optional[List[StarLinkShellConfig]] = None):
        self.shells: List[StarLinkShellConfig] = shells or STARLINK_SHELLS_MINI
        self.satellites: List[Satellite] = []
        self._build()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def _build(self) -> None:
        """Populate self.satellites from the shell configurations."""
        self.satellites.clear()
        for cfg in self.shells:
            a = cfg.semi_major_axis_km
            raan_step = 360.0 / cfg.n_planes
            for p in range(cfg.n_planes):
                raan = p * raan_step
                # Walker-Delta phase offset so adjacent planes are staggered
                phase_offset = (360.0 / (cfg.n_planes * cfg.n_sats)) * p
                in_plane_step = 360.0 / cfg.n_sats
                for s in range(cfg.n_sats):
                    M0 = (s * in_plane_step + phase_offset) % 360.0
                    sat_id = f"sh{cfg.shell_id}-p{p:03d}-s{s:03d}"
                    sat = Satellite(
                        sat_id=sat_id,
                        a=a,
                        e=0.0,
                        inc_deg=cfg.inc_deg,
                        raan_deg=raan,
                        argp_deg=0.0,
                        M0_deg=M0,
                        shell_id=cfg.shell_id,
                        plane_id=p,
                        sat_index=s,
                    )
                    self.satellites.append(sat)

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate(self, t: float) -> "Constellation":
        """
        Propagate all satellites to time *t* (seconds from epoch).
        Returns self for chaining.
        """
        for sat in self.satellites:
            sat.propagate(t)
        return self

    # ------------------------------------------------------------------
    # Query helpers
    # ------------------------------------------------------------------

    def get_shell(self, shell_id: int) -> List[Satellite]:
        """Return all satellites belonging to a given shell."""
        return [s for s in self.satellites if s.shell_id == shell_id]

    def get_plane(self, shell_id: int, plane_id: int) -> List[Satellite]:
        """Return all satellites in a specific shell/plane pair."""
        return [
            s for s in self.satellites
            if s.shell_id == shell_id and s.plane_id == plane_id
        ]

    @property
    def total_satellites(self) -> int:
        return len(self.satellites)

    def __repr__(self) -> str:  # pragma: no cover
        n_shells = len(self.shells)
        return (
            f"Constellation(shells={n_shells}, "
            f"total_satellites={self.total_satellites})"
        )
