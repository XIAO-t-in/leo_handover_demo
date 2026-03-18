"""
GroundStation – represents a user terminal or gateway on Earth's surface.

Provides:
- Visibility (elevation-angle) check for any satellite
- Sorted list of visible satellites
- Best-serving satellite selection
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .satellite import Satellite
from .utils import (
    lla_to_ecef,
    elevation_angle,
    distance_km,
)


@dataclass
class GroundStation:
    """
    A ground station (user terminal or gateway) at a fixed geodetic location.

    Parameters
    ----------
    gs_id          : unique identifier
    lat_deg        : geodetic latitude (degrees, -90 to +90)
    lon_deg        : geodetic longitude (degrees, -180 to +180)
    alt_km         : altitude above sea level (km, default 0)
    min_elev_deg   : minimum elevation angle to consider a satellite visible
                     (degrees, Starlink uses ~25° for user terminals)
    """

    gs_id: str
    lat_deg: float
    lon_deg: float
    alt_km: float = 0.0
    min_elev_deg: float = 25.0

    _ecef: np.ndarray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._ecef = lla_to_ecef(self.lat_deg, self.lon_deg, self.alt_km)

    @property
    def ecef(self) -> np.ndarray:
        """ECEF position of the ground station (km)."""
        return self._ecef

    # ------------------------------------------------------------------
    # Visibility
    # ------------------------------------------------------------------

    def elevation_to(self, sat: Satellite) -> float:
        """
        Elevation angle (degrees) from this ground station to *sat*.
        Negative means the satellite is below the horizon.
        """
        return elevation_angle(self._ecef, sat.pos_ecef)

    def is_visible(self, sat: Satellite) -> bool:
        """True if the satellite is above the minimum elevation threshold."""
        return self.elevation_to(sat) >= self.min_elev_deg

    def distance_to(self, sat: Satellite) -> float:
        """Slant-range distance (km) from the ground station to *sat*."""
        return distance_km(self._ecef, sat.pos_ecef)

    # ------------------------------------------------------------------
    # Multi-satellite helpers
    # ------------------------------------------------------------------

    def visible_satellites(self, satellites: List[Satellite]) -> List[Tuple[Satellite, float]]:
        """
        Return all visible satellites with their elevation angles,
        sorted by descending elevation (highest elevation first).

        Returns
        -------
        list of (Satellite, elevation_deg) tuples
        """
        result = []
        for sat in satellites:
            el = self.elevation_to(sat)
            if el >= self.min_elev_deg:
                result.append((sat, el))
        result.sort(key=lambda x: x[1], reverse=True)
        return result

    def best_satellite(self, satellites: List[Satellite]) -> Optional[Satellite]:
        """
        Return the single best (highest elevation) visible satellite,
        or None if no satellite is visible.
        """
        visible = self.visible_satellites(satellites)
        return visible[0][0] if visible else None

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"GroundStation(id={self.gs_id!r}, "
            f"lat={self.lat_deg:.2f}°, lon={self.lon_deg:.2f}°, "
            f"min_elev={self.min_elev_deg}°)"
        )
