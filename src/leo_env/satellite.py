"""
Satellite – orbital element representation and state propagation.

A Satellite is described by its Keplerian elements and propagated forward in
time using a simple Kepler (two-body) model.  For short simulations (≤ a few
hours) this is accurate to within a few km.
"""

from __future__ import annotations

import math
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .utils import (
    mean_motion,
    orbital_period,
    true_anomaly_from_mean,
    orbit_to_inertial,
    eci_to_ecef,
    ecef_to_lla,
    deg2rad,
)


@dataclass
class Satellite:
    """
    Represents a single LEO satellite.

    Orbital elements follow the standard Keplerian conventions:
        a    – semi-major axis (km)
        e    – eccentricity (dimensionless, 0 for circular)
        inc  – inclination (degrees)
        raan – right ascension of the ascending node (degrees)
        argp – argument of perigee (degrees)
        M0   – mean anomaly at epoch (degrees)

    Parameters
    ----------
    sat_id      : unique identifier string  e.g. "shell0-p03-s07"
    a           : semi-major axis (km)
    e           : eccentricity
    inc_deg     : inclination (degrees)
    raan_deg    : RAAN (degrees)
    argp_deg    : argument of perigee (degrees)
    M0_deg      : mean anomaly at t=0 (degrees)
    shell_id    : optional shell / layer identifier
    plane_id    : orbital plane index within the shell
    sat_index   : satellite index within the orbital plane
    """

    sat_id: str
    a: float                        # km
    e: float = 0.0
    inc_deg: float = 53.0
    raan_deg: float = 0.0
    argp_deg: float = 0.0
    M0_deg: float = 0.0
    shell_id: int = 0
    plane_id: int = 0
    sat_index: int = 0

    # Runtime state (populated by propagate)
    _pos_eci: np.ndarray = field(default_factory=lambda: np.zeros(3), repr=False, compare=False)
    _vel_eci: np.ndarray = field(default_factory=lambda: np.zeros(3), repr=False, compare=False)
    _pos_ecef: np.ndarray = field(default_factory=lambda: np.zeros(3), repr=False, compare=False)
    _t: float = field(default=0.0, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Derived orbital properties
    # ------------------------------------------------------------------

    @property
    def altitude_km(self) -> float:
        """Altitude above Earth's surface for a circular orbit (km)."""
        from .utils import EARTH_RADIUS_KM
        return self.a - EARTH_RADIUS_KM

    @property
    def period_s(self) -> float:
        """Orbital period in seconds."""
        return orbital_period(self.a)

    @property
    def n(self) -> float:
        """Mean motion (rad/s)."""
        return mean_motion(self.a)

    # ------------------------------------------------------------------
    # Propagation
    # ------------------------------------------------------------------

    def propagate(self, t: float) -> "Satellite":
        """
        Update the satellite's ECI and ECEF position/velocity at time *t*
        (seconds from the simulation epoch, t=0).

        Returns self for chaining.
        """
        # Current mean anomaly
        M = deg2rad(self.M0_deg) + self.n * t
        nu = true_anomaly_from_mean(M % (2 * math.pi), self.e)

        pos_eci, vel_eci = orbit_to_inertial(
            self.a,
            self.e,
            deg2rad(self.inc_deg),
            deg2rad(self.raan_deg),
            deg2rad(self.argp_deg),
            nu,
        )

        self._pos_eci = pos_eci
        self._vel_eci = vel_eci
        self._pos_ecef = eci_to_ecef(pos_eci, t)
        self._t = t
        return self

    # ------------------------------------------------------------------
    # Position accessors (only valid after propagate() has been called)
    # ------------------------------------------------------------------

    @property
    def pos_eci(self) -> np.ndarray:
        """ECI position vector (km)."""
        return self._pos_eci

    @property
    def vel_eci(self) -> np.ndarray:
        """ECI velocity vector (km/s)."""
        return self._vel_eci

    @property
    def pos_ecef(self) -> np.ndarray:
        """ECEF position vector (km)."""
        return self._pos_ecef

    @property
    def lat_lon_alt(self) -> tuple:
        """Geodetic (latitude_deg, longitude_deg, altitude_km)."""
        return ecef_to_lla(self._pos_ecef)

    def __repr__(self) -> str:  # pragma: no cover
        lat, lon, alt = self.lat_lon_alt
        return (
            f"Satellite(id={self.sat_id!r}, alt={alt:.1f} km, "
            f"lat={lat:.2f}°, lon={lon:.2f}°)"
        )
