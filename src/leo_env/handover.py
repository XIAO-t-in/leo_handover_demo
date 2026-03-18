"""
Handover – event detection and handover management.

The HandoverManager tracks the currently-serving satellite for each ground
station and triggers a HandoverEvent whenever the serving satellite should
change.

Supported handover strategies
------------------------------
- "best_rsrp"      : always connect to the satellite with the highest RSRP
- "hysteresis"     : trigger handover only if the new satellite's RSRP exceeds
                     the current one by at least `hysteresis_db`
- "elevation"      : always connect to the highest-elevation visible satellite
"""

from __future__ import annotations

import time as _time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .satellite import Satellite
from .ground_station import GroundStation
from .channel import LinkChannel, LinkQuality


STRATEGY_BEST_RSRP = "best_rsrp"
STRATEGY_HYSTERESIS = "hysteresis"
STRATEGY_ELEVATION = "elevation"

_VALID_STRATEGIES = {STRATEGY_BEST_RSRP, STRATEGY_HYSTERESIS, STRATEGY_ELEVATION}


@dataclass
class HandoverEvent:
    """
    Represents a single handover event at a ground station.

    Attributes
    ----------
    sim_time_s      : simulation time (seconds from epoch) when handover occurs
    gs_id           : ground-station identifier
    from_sat_id     : satellite being released (empty string if first connection)
    to_sat_id       : satellite being connected
    from_rsrp_dbm   : RSRP of the old satellite (NaN if first connection)
    to_rsrp_dbm     : RSRP of the new satellite
    reason          : textual reason for the handover
    """

    sim_time_s: float
    gs_id: str
    from_sat_id: str
    to_sat_id: str
    from_rsrp_dbm: float
    to_rsrp_dbm: float
    reason: str = ""

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"HandoverEvent(t={self.sim_time_s:.1f}s, gs={self.gs_id!r}, "
            f"{self.from_sat_id!r} → {self.to_sat_id!r}, "
            f"RSRP {self.from_rsrp_dbm:.1f}→{self.to_rsrp_dbm:.1f} dBm)"
        )


class HandoverManager:
    """
    Manages handover decisions for a set of ground stations.

    Parameters
    ----------
    ground_stations  : list of GroundStation objects to manage
    channel          : LinkChannel instance for computing link quality
    strategy         : handover strategy ("best_rsrp", "hysteresis", "elevation")
    hysteresis_db    : RSRP gain threshold required to trigger handover
                       (only used when strategy == "hysteresis")
    """

    def __init__(
        self,
        ground_stations: List[GroundStation],
        channel: Optional[LinkChannel] = None,
        strategy: str = STRATEGY_HYSTERESIS,
        hysteresis_db: float = 3.0,
    ):
        if strategy not in _VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy {strategy!r}. "
                f"Choose from {sorted(_VALID_STRATEGIES)}."
            )
        self.ground_stations = ground_stations
        self.channel = channel or LinkChannel()
        self.strategy = strategy
        self.hysteresis_db = hysteresis_db

        # Current serving satellite per GS (keyed by gs_id)
        self._serving: Dict[str, Optional[Satellite]] = {
            gs.gs_id: None for gs in ground_stations
        }
        # Current serving link quality per GS
        self._serving_lq: Dict[str, Optional[LinkQuality]] = {
            gs.gs_id: None for gs in ground_stations
        }
        # Cumulative event log
        self.events: List[HandoverEvent] = []

    # ------------------------------------------------------------------
    # Per-step update
    # ------------------------------------------------------------------

    def step(self, t: float, satellites: List[Satellite]) -> List[HandoverEvent]:
        """
        Evaluate handover decisions for all ground stations at simulation
        time *t*.  All satellites must already be propagated to time *t*.

        Returns the list of HandoverEvent objects generated in this step.
        """
        new_events: List[HandoverEvent] = []
        for gs in self.ground_stations:
            event = self._evaluate_gs(t, gs, satellites)
            if event is not None:
                new_events.append(event)
                self.events.append(event)
        return new_events

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _evaluate_gs(
        self,
        t: float,
        gs: GroundStation,
        satellites: List[Satellite],
    ) -> Optional[HandoverEvent]:
        """
        Evaluate whether a handover should occur for *gs* at time *t*.
        Returns a HandoverEvent if a handover is triggered, else None.
        """
        if self.strategy == STRATEGY_ELEVATION:
            candidate_sat = gs.best_satellite(satellites)
            candidate_lq = (
                self.channel.evaluate(candidate_sat, gs)
                if candidate_sat is not None
                else None
            )
        else:
            # RSRP-based strategies: evaluate all visible satellites
            candidate_lq = self.channel.best_link(satellites, gs)
            candidate_sat = candidate_lq.satellite if candidate_lq else None

        serving_sat = self._serving[gs.gs_id]
        serving_lq = self._serving_lq[gs.gs_id]

        # Case 1: No satellite visible at all – nothing to connect to
        if candidate_sat is None:
            # If we were connected, the serving satellite is now out of view
            if serving_sat is not None:
                event = HandoverEvent(
                    sim_time_s=t,
                    gs_id=gs.gs_id,
                    from_sat_id=serving_sat.sat_id,
                    to_sat_id="",
                    from_rsrp_dbm=serving_lq.rsrp_dbm if serving_lq else float("nan"),
                    to_rsrp_dbm=float("nan"),
                    reason="serving_satellite_out_of_view",
                )
                self._serving[gs.gs_id] = None
                self._serving_lq[gs.gs_id] = None
                return event
            return None

        # Case 2: Not yet connected – first association
        if serving_sat is None:
            event = HandoverEvent(
                sim_time_s=t,
                gs_id=gs.gs_id,
                from_sat_id="",
                to_sat_id=candidate_sat.sat_id,
                from_rsrp_dbm=float("nan"),
                to_rsrp_dbm=candidate_lq.rsrp_dbm,
                reason="initial_association",
            )
            self._serving[gs.gs_id] = candidate_sat
            self._serving_lq[gs.gs_id] = candidate_lq
            return event

        # Case 3: Already connected – check whether to switch
        if candidate_sat.sat_id == serving_sat.sat_id:
            # Refresh serving link quality
            self._serving_lq[gs.gs_id] = candidate_lq
            return None

        # Re-evaluate current serving satellite's link quality
        serving_lq_now = self.channel.evaluate(serving_sat, gs)
        self._serving_lq[gs.gs_id] = serving_lq_now

        # Check if serving satellite dropped below the elevation threshold
        if not gs.is_visible(serving_sat):
            event = HandoverEvent(
                sim_time_s=t,
                gs_id=gs.gs_id,
                from_sat_id=serving_sat.sat_id,
                to_sat_id=candidate_sat.sat_id,
                from_rsrp_dbm=serving_lq_now.rsrp_dbm,
                to_rsrp_dbm=candidate_lq.rsrp_dbm,
                reason="serving_satellite_below_elevation",
            )
            self._serving[gs.gs_id] = candidate_sat
            self._serving_lq[gs.gs_id] = candidate_lq
            return event

        # Hysteresis check
        if self.strategy == STRATEGY_HYSTERESIS:
            gain = candidate_lq.rsrp_dbm - serving_lq_now.rsrp_dbm
            if gain < self.hysteresis_db:
                return None

        # Strategy: best_rsrp or elevation – switch if candidate is better
        if self.strategy in (STRATEGY_BEST_RSRP, STRATEGY_HYSTERESIS):
            if candidate_lq.rsrp_dbm <= serving_lq_now.rsrp_dbm:
                return None

        event = HandoverEvent(
            sim_time_s=t,
            gs_id=gs.gs_id,
            from_sat_id=serving_sat.sat_id,
            to_sat_id=candidate_sat.sat_id,
            from_rsrp_dbm=serving_lq_now.rsrp_dbm,
            to_rsrp_dbm=candidate_lq.rsrp_dbm,
            reason=f"strategy_{self.strategy}",
        )
        self._serving[gs.gs_id] = candidate_sat
        self._serving_lq[gs.gs_id] = candidate_lq
        return event

    # ------------------------------------------------------------------
    # Reporting
    # ------------------------------------------------------------------

    @property
    def total_handovers(self) -> int:
        """Total number of handover events accumulated across all ground stations."""
        return len(self.events)

    def handover_rate(self, duration_s: float) -> float:
        """
        Average handover rate (handovers per second) over the given duration.
        Only counts non-initial-association and non-out-of-view events.
        """
        n = sum(
            1 for e in self.events
            if e.reason not in ("initial_association", "serving_satellite_out_of_view")
        )
        return n / duration_s if duration_s > 0 else 0.0

    def serving_satellite(self, gs_id: str) -> Optional[Satellite]:
        """Return the currently-serving satellite for a ground station."""
        return self._serving.get(gs_id)
