"""
低轨卫星切换仿真环境 (LEO Satellite Handover Simulation Environment)

Implements a gymnasium-compatible environment where a ground user terminal
must decide which LEO satellite to connect to as satellites pass overhead.

State  : [rsrp_norm, elevation_norm, remaining_time_norm, load] × max_visible
Action : integer index of the satellite to connect to (0 … max_visible-1)
Reward : signal-quality gain − handover penalty − outage penalty
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class LEOHandoverEnv(gym.Env):
    """LEO satellite handover reinforcement-learning environment.

    Parameters
    ----------
    n_satellites : int
        Total number of satellites in the constellation.
    max_visible : int
        Maximum number of simultaneously visible satellites tracked by the agent.
    altitude_km : float
        Orbital altitude above Earth's surface in kilometres (default: 550 km,
        similar to Starlink shell 1).
    freq_ghz : float
        Carrier frequency in GHz (default: 28 GHz, Ka-band).
    tx_power_dbm : float
        Satellite transmit power in dBm.
    time_step_s : float
        Simulation time step in seconds.
    min_elevation_deg : float
        Minimum elevation angle (degrees) below which a satellite is not
        considered visible.
    handover_penalty : float
        Reward penalty applied every time the agent switches to a different
        satellite.
    outage_penalty : float
        Reward penalty applied every step when no satellite is visible.
    render_mode : str or None
        ``"human"`` enables console output; ``None`` disables rendering.
    """

    metadata = {"render_modes": ["human"]}

    # Physical constants
    EARTH_RADIUS_KM: float = 6371.0
    GM: float = 398_600.4418          # km³ s⁻²
    SPEED_OF_LIGHT: float = 3e8       # m s⁻¹

    def __init__(
        self,
        n_satellites: int = 20,
        max_visible: int = 6,
        altitude_km: float = 550.0,
        freq_ghz: float = 28.0,
        tx_power_dbm: float = 30.0,
        time_step_s: float = 10.0,
        min_elevation_deg: float = 10.0,
        handover_penalty: float = 0.5,
        outage_penalty: float = 10.0,
        render_mode: Optional[str] = None,
    ) -> None:
        super().__init__()

        self.n_satellites = n_satellites
        self.max_visible = max_visible
        self.altitude_km = altitude_km
        self.freq_ghz = freq_ghz
        self.tx_power_dbm = tx_power_dbm
        self.time_step_s = time_step_s
        self.min_elevation_deg = min_elevation_deg
        self.handover_penalty = handover_penalty
        self.outage_penalty = outage_penalty
        self.render_mode = render_mode

        # Derived orbital parameters
        self.orbital_radius_km: float = self.EARTH_RADIUS_KM + self.altitude_km
        self.orbital_period_s: float = (
            2.0 * np.pi * np.sqrt(self.orbital_radius_km ** 3 / self.GM)
        )
        self.orbital_omega: float = 2.0 * np.pi / self.orbital_period_s

        # Maximum central angle for satellite visibility at min_elevation_deg
        self.max_central_angle_rad: float = self._compute_max_central_angle()

        # Episode length: 2 full orbital periods
        self.max_steps: int = int(2.0 * self.orbital_period_s / self.time_step_s)

        # Observation: [rsrp, elevation, remaining_time, load] × max_visible (all in [-1, 1])
        obs_dim = self.max_visible * 4
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32
        )
        # Action: choose which visible satellite to connect to
        self.action_space = spaces.Discrete(self.max_visible)

        # Episode state (initialised in reset)
        self.sat_angles: np.ndarray = np.zeros(n_satellites)
        self.sat_loads: np.ndarray = np.zeros(n_satellites)
        self.current_step: int = 0
        self.connected_sat_id: Optional[int] = None
        self.n_handovers: int = 0
        self.outage_steps: int = 0
        self.total_rsrp_sum: float = 0.0
        self.episode_reward: float = 0.0

    # ------------------------------------------------------------------
    # Internal geometry helpers
    # ------------------------------------------------------------------

    def _compute_max_central_angle(self) -> float:
        """Return the maximum central angle (radians) at which a satellite
        is still above *min_elevation_deg*."""
        Re = self.EARTH_RADIUS_KM
        Rs = self.orbital_radius_km
        el_min = np.radians(self.min_elevation_deg)
        # sin(nadir_angle) = (Re / Rs) * cos(el_min)
        sin_rho = (Re / Rs) * np.cos(el_min)
        rho = np.arcsin(np.clip(sin_rho, -1.0, 1.0))
        return np.pi / 2.0 - el_min - rho

    def _elevation_angle(self, central_angle_rad: float) -> float:
        """Elevation angle (degrees) as a function of the central angle between
        the ground user and the satellite's sub-satellite point.

        Uses the formula derived from the law of cosines in the triangle
        (Earth centre, ground user, satellite):
            sin(el) = (Rs·cos(γ) − Re) / slant_range
        """
        Re = self.EARTH_RADIUS_KM
        Rs = self.orbital_radius_km
        gamma = abs(central_angle_rad)
        d = np.sqrt(Re ** 2 + Rs ** 2 - 2.0 * Re * Rs * np.cos(gamma))
        if d < 1e-6:
            return 90.0
        sin_el = (Rs * np.cos(gamma) - Re) / d
        return float(np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0))))

    def _slant_range_km(self, central_angle_rad: float) -> float:
        """Slant range (km) from ground user to satellite."""
        Re = self.EARTH_RADIUS_KM
        Rs = self.orbital_radius_km
        gamma = abs(central_angle_rad)
        return float(np.sqrt(Re ** 2 + Rs ** 2 - 2.0 * Re * Rs * np.cos(gamma)))

    def _compute_rsrp(self, elevation_deg: float, distance_km: float) -> float:
        """Compute RSRP (dBm) using a simplified link budget.

        RSRP = Tx_Power + Sat_Gain + UE_Gain − FSPL − Atm_Loss
        """
        d_m = distance_km * 1_000.0
        f_hz = self.freq_ghz * 1e9
        fspl = (
            20.0 * np.log10(d_m)
            + 20.0 * np.log10(f_hz)
            + 20.0 * np.log10(4.0 * np.pi / self.SPEED_OF_LIGHT)
        )
        # Atmospheric loss increases towards the horizon
        el_rad = np.radians(max(float(elevation_deg), 1.0))
        atm_loss_db = min(0.5 / np.sin(el_rad), 5.0)

        sat_gain_dbi = 35.0
        ue_gain_dbi = 30.0
        return float(self.tx_power_dbm + sat_gain_dbi + ue_gain_dbi - fspl - atm_loss_db)

    # ------------------------------------------------------------------
    # Core environment logic
    # ------------------------------------------------------------------

    def _get_visible_satellites(self) -> List[Dict[str, Any]]:
        """Return a list of visible satellite dicts, sorted by RSRP (best first).

        Each entry contains:
            sat_id, elevation (deg), rsrp (dBm),
            remaining_time (s), load (0–1).
        """
        visible: List[Dict[str, Any]] = []
        for sat_id in range(self.n_satellites):
            gamma = self.sat_angles[sat_id]
            el_deg = self._elevation_angle(gamma)
            if el_deg >= self.min_elevation_deg:
                dist_km = self._slant_range_km(gamma)
                rsrp = self._compute_rsrp(el_deg, dist_km)
                angle_to_horizon = self.max_central_angle_rad - abs(gamma)
                remaining_s = max(0.0, angle_to_horizon / self.orbital_omega)
                visible.append(
                    {
                        "sat_id": sat_id,
                        "elevation": el_deg,
                        "rsrp": rsrp,
                        "remaining_time": remaining_s,
                        "load": float(self.sat_loads[sat_id]),
                    }
                )
        visible.sort(key=lambda x: x["rsrp"], reverse=True)
        return visible[: self.max_visible]

    def _get_observation(self, visible_sats: List[Dict[str, Any]]) -> np.ndarray:
        """Convert visible satellite data to a normalised observation vector in [-1, 1]."""
        obs = np.zeros(self.max_visible * 4, dtype=np.float32)

        rsrp_lo, rsrp_hi = -120.0, -40.0
        el_lo, el_hi = self.min_elevation_deg, 90.0
        time_lo, time_hi = 0.0, self.orbital_period_s * 0.15

        def _norm(val: float, lo: float, hi: float) -> float:
            return float(np.clip(2.0 * (val - lo) / (hi - lo) - 1.0, -1.0, 1.0))

        for i, sat in enumerate(visible_sats):
            if i >= self.max_visible:
                break
            base = i * 4
            obs[base] = _norm(sat["rsrp"], rsrp_lo, rsrp_hi)
            obs[base + 1] = _norm(sat["elevation"], el_lo, el_hi)
            obs[base + 2] = _norm(sat["remaining_time"], time_lo, time_hi)
            obs[base + 3] = _norm(sat["load"], 0.0, 1.0)
        return obs

    # ------------------------------------------------------------------
    # gymnasium API
    # ------------------------------------------------------------------

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)

        # Evenly distribute satellites around the orbit, with a small random offset
        self.sat_angles = np.linspace(-np.pi, np.pi, self.n_satellites, endpoint=False)
        self.sat_angles += self.np_random.uniform(
            -np.pi / self.n_satellites, np.pi / self.n_satellites, self.n_satellites
        )
        self.sat_loads = self.np_random.uniform(0.1, 0.8, self.n_satellites)

        self.current_step = 0
        self.connected_sat_id = None
        self.n_handovers = 0
        self.outage_steps = 0
        self.total_rsrp_sum = 0.0
        self.episode_reward = 0.0

        visible = self._get_visible_satellites()
        if visible:
            self.connected_sat_id = visible[0]["sat_id"]

        obs = self._get_observation(visible)
        info: Dict[str, Any] = {
            "n_handovers": 0,
            "visible_count": len(visible),
            "connected_sat": self.connected_sat_id,
        }
        return obs, info

    def step(
        self, action: int
    ) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        visible = self._get_visible_satellites()

        reward = 0.0
        current_rsrp = -130.0

        if not visible:
            reward -= self.outage_penalty
            self.outage_steps += 1
            self.connected_sat_id = None
        else:
            action_clamped = min(int(action), len(visible) - 1)
            chosen = visible[action_clamped]

            if (
                self.connected_sat_id is not None
                and chosen["sat_id"] != self.connected_sat_id
            ):
                reward -= self.handover_penalty
                self.n_handovers += 1

            # Signal-quality reward normalised to [0, 1]
            rsrp_norm = (chosen["rsrp"] - (-120.0)) / ((-40.0) - (-120.0))
            reward += float(np.clip(rsrp_norm, 0.0, 1.0))
            # Mild load penalty (prefer less-loaded satellites)
            reward -= 0.1 * chosen["load"]

            self.connected_sat_id = chosen["sat_id"]
            current_rsrp = chosen["rsrp"]

        self.total_rsrp_sum += current_rsrp
        self.episode_reward += reward

        # Advance all satellite positions by one time step
        self.sat_angles += self.orbital_omega * self.time_step_s
        # Wrap angles to (−π, π]
        self.sat_angles = ((self.sat_angles + np.pi) % (2.0 * np.pi)) - np.pi

        # Slowly vary satellite loads
        self.sat_loads += self.np_random.normal(0.0, 0.02, self.n_satellites)
        self.sat_loads = np.clip(self.sat_loads, 0.1, 0.95)

        self.current_step += 1
        terminated = False
        truncated = self.current_step >= self.max_steps

        new_visible = self._get_visible_satellites()
        obs = self._get_observation(new_visible)
        info: Dict[str, Any] = {
            "n_handovers": self.n_handovers,
            "visible_count": len(new_visible),
            "rsrp": current_rsrp,
            "outage_steps": self.outage_steps,
            "connected_sat": self.connected_sat_id,
            "episode_reward": self.episode_reward,
        }
        return obs, reward, terminated, truncated, info

    def render(self) -> None:
        if self.render_mode != "human":
            return
        visible = self._get_visible_satellites()
        print(
            f"\nStep {self.current_step:4d} | "
            f"Handovers: {self.n_handovers:3d} | "
            f"Outages: {self.outage_steps:3d}"
        )
        for sat in visible:
            marker = "► " if sat["sat_id"] == self.connected_sat_id else "  "
            print(
                f"  {marker}Sat {sat['sat_id']:3d}: "
                f"El={sat['elevation']:5.1f}°  "
                f"RSRP={sat['rsrp']:7.2f} dBm  "
                f"Remain={sat['remaining_time']:6.0f} s  "
                f"Load={sat['load']:.2f}"
            )
