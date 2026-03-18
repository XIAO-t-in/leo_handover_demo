"""
Traditional LEO Satellite Handover Baseline Algorithms

Provides rule-based handover policies that are used as baselines when
comparing against the DQN agent.  All policies implement a common
``select_satellite(visible_sats)`` interface that returns the index of the
chosen satellite in the *visible_sats* list.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


class TraditionalHandoverPolicy:
    """Base class for traditional handover algorithms.

    Subclasses must implement :meth:`select_satellite`.
    """

    def __init__(self, name: str) -> None:
        self.name = name
        self.connected_sat_id: Optional[int] = None
        self.n_handovers: int = 0

    def select_satellite(self, visible_sats: List[Dict[str, Any]]) -> int:
        """Return the index (into *visible_sats*) of the satellite to connect to."""
        raise NotImplementedError

    def reset(self) -> None:
        """Reset internal state at the start of a new episode."""
        self.connected_sat_id = None
        self.n_handovers = 0

    def _register_choice(self, visible_sats: List[Dict[str, Any]], idx: int) -> None:
        """Update handover counter and connected satellite."""
        chosen_id = visible_sats[idx]["sat_id"]
        if self.connected_sat_id is not None and chosen_id != self.connected_sat_id:
            self.n_handovers += 1
        self.connected_sat_id = chosen_id


class BestRSRPPolicy(TraditionalHandoverPolicy):
    """Always connect to the satellite with the highest RSRP.

    Parameters
    ----------
    hysteresis_db : float
        Handover margin (dB).  The new satellite must exceed the current
        satellite's RSRP by at least this amount before a handover is
        triggered.  When 0 the agent always moves to the best satellite.
    """

    def __init__(self, hysteresis_db: float = 0.0) -> None:
        super().__init__("Best RSRP")
        self.hysteresis_db = hysteresis_db

    def select_satellite(self, visible_sats: List[Dict[str, Any]]) -> int:
        if not visible_sats:
            return 0

        if self.connected_sat_id is None:
            chosen_idx = 0
        else:
            current_idx = next(
                (i for i, s in enumerate(visible_sats)
                 if s["sat_id"] == self.connected_sat_id),
                None,
            )
            if current_idx is None:
                # Previous satellite has gone below horizon
                chosen_idx = 0
            elif visible_sats[0]["rsrp"] > visible_sats[current_idx]["rsrp"] + self.hysteresis_db:
                chosen_idx = 0
            else:
                chosen_idx = current_idx

        self._register_choice(visible_sats, chosen_idx)
        return chosen_idx


class HysteresisRSRPPolicy(BestRSRPPolicy):
    """Best RSRP with a configurable hysteresis margin to reduce ping-pong
    handovers (analogous to the 3GPP A3 event with offset)."""

    def __init__(self, hysteresis_db: float = 3.0) -> None:
        super().__init__(hysteresis_db=hysteresis_db)
        self.name = f"RSRP+Hysteresis({hysteresis_db:.0f}dB)"


class MaxElevationPolicy(TraditionalHandoverPolicy):
    """Connect to the satellite with the highest elevation angle.

    Higher elevation ≈ shorter slant range ≈ lower path loss, while also
    minimising atmospheric attenuation and Doppler shift.
    """

    def __init__(self) -> None:
        super().__init__("Max Elevation")

    def select_satellite(self, visible_sats: List[Dict[str, Any]]) -> int:
        if not visible_sats:
            return 0
        idx = max(range(len(visible_sats)), key=lambda i: visible_sats[i]["elevation"])
        self._register_choice(visible_sats, idx)
        return idx


class LongestRemainingTimePolicy(TraditionalHandoverPolicy):
    """Connect to the satellite with the longest remaining visible time.

    This strategy defers handovers as long as possible, minimising handover
    frequency at the cost of potentially suboptimal signal quality.
    """

    def __init__(self) -> None:
        super().__init__("Longest Remaining Time")

    def select_satellite(self, visible_sats: List[Dict[str, Any]]) -> int:
        if not visible_sats:
            return 0
        idx = max(
            range(len(visible_sats)), key=lambda i: visible_sats[i]["remaining_time"]
        )
        self._register_choice(visible_sats, idx)
        return idx
