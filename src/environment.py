"""LEO satellite handover environment for cooperative multi-user decision making."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


@dataclass(frozen=True)
class Satellite:
    """Satellite with deterministic angular motion on a circular orbit."""

    sat_id: int
    phase: float
    angular_speed: float


@dataclass(frozen=True)
class User:
    """Aircraft user with deterministic angular motion."""

    user_id: int
    phase: float
    angular_speed: float


class LeoHandoverEnv:
    """Dynamic multi-user LEO access environment in discrete time slots.

    The environment models:
    - per-slot dynamic geometry between users and satellites
    - visible satellite candidate sets
    - communication rate for each candidate link
    - admission failure risk from resource occupancy
    - per-user utility from effective access reward, handover cost and disconnect loss
    """

    EARTH_RADIUS_KM = 6371.0

    def __init__(
        self,
        num_users: int = 3,
        num_satellites: int = 6,
        slot_seconds: float = 1.0,
        orbit_altitude_km: float = 550.0,
        aircraft_altitude_km: float = 12.0,
        sat_capacity: int = 2,
        bandwidth_hz: float = 20e6,
        tx_power: float = 60.0,
        noise_power: float = 1e-8,
        pathloss_exp: float = 2.2,
        switch_cost: float = 0.8,
        disconnect_loss: float = 1.5,
        max_steps: int = 200,
        seed: int | None = None,
    ) -> None:
        if num_users <= 0:
            raise ValueError("num_users must be positive")
        if num_satellites <= 0:
            raise ValueError("num_satellites must be positive")
        if sat_capacity <= 0:
            raise ValueError("sat_capacity must be positive")

        self.num_users = num_users
        self.num_satellites = num_satellites
        self.slot_seconds = slot_seconds
        self.orbit_altitude_km = orbit_altitude_km
        self.aircraft_altitude_km = aircraft_altitude_km
        self.sat_capacity = sat_capacity
        self.bandwidth_hz = bandwidth_hz
        self.tx_power = tx_power
        self.noise_power = noise_power
        self.pathloss_exp = pathloss_exp
        self.switch_cost = switch_cost
        self.disconnect_loss = disconnect_loss
        self.max_steps = max_steps

        self.rng = random.Random(seed)

        self.satellites: List[Satellite] = []
        self.users: List[User] = []
        self.satellite_positions: List[Tuple[float, float]] = []
        self.user_positions: List[Tuple[float, float]] = []

        self.current_slot = 0
        self.current_assignment: List[int] = [-1 for _ in range(self.num_users)]
        self.satellite_loads: List[int] = [0 for _ in range(self.num_satellites)]

        self._init_entities()

    @property
    def obs_dim(self) -> int:
        # [current_sat_norm] + visibility_mask + normalized_rates + normalized_loads
        return 1 + self.num_satellites * 3

    @property
    def state_dim(self) -> int:
        # Global state aggregates every user's local observation.
        return self.num_users * self.obs_dim

    @property
    def action_dim(self) -> int:
        # 0 means disconnect, 1..num_satellites map to satellite IDs 0..num_satellites-1
        return self.num_satellites + 1

    def _init_entities(self) -> None:
        self.satellites = []
        self.users = []

        for sat_id in range(self.num_satellites):
            phase = 2.0 * math.pi * sat_id / self.num_satellites
            # Distinct but close angular speeds.
            speed = 0.004 + 0.0005 * (sat_id % 3)
            self.satellites.append(Satellite(sat_id=sat_id, phase=phase, angular_speed=speed))

        for user_id in range(self.num_users):
            phase = 2.0 * math.pi * user_id / self.num_users + 0.13 * (user_id % 2)
            speed = 0.0012 + 0.0003 * (user_id % 2)
            self.users.append(User(user_id=user_id, phase=phase, angular_speed=speed))

    def reset(self) -> Dict[str, object]:
        self.current_slot = 0
        self.current_assignment = [-1 for _ in range(self.num_users)]
        self.satellite_loads = [0 for _ in range(self.num_satellites)]
        self._update_positions()
        return {
            "local_obs": self.get_local_observations(),
            "global_state": self.get_global_state(),
        }

    def _angle_at(self, phase: float, angular_speed: float) -> float:
        return phase + angular_speed * self.current_slot * self.slot_seconds

    def _point(self, radius: float, angle: float) -> Tuple[float, float]:
        return (radius * math.cos(angle), radius * math.sin(angle))

    def _distance(self, a: Tuple[float, float], b: Tuple[float, float]) -> float:
        dx = a[0] - b[0]
        dy = a[1] - b[1]
        return math.sqrt(dx * dx + dy * dy)

    def _visibility_range_km(self) -> float:
        # A practical cutoff for candidate set calculation.
        return 3500.0

    def _update_positions(self) -> None:
        sat_radius = self.EARTH_RADIUS_KM + self.orbit_altitude_km
        user_radius = self.EARTH_RADIUS_KM + self.aircraft_altitude_km

        self.satellite_positions = [
            self._point(sat_radius, self._angle_at(s.phase, s.angular_speed)) for s in self.satellites
        ]
        self.user_positions = [
            self._point(user_radius, self._angle_at(u.phase, u.angular_speed)) for u in self.users
        ]

    def visible_satellites(self, user_id: int) -> List[int]:
        user_pos = self.user_positions[user_id]
        threshold = self._visibility_range_km()
        visible: List[int] = []
        for sat_id, sat_pos in enumerate(self.satellite_positions):
            if self._distance(user_pos, sat_pos) <= threshold:
                visible.append(sat_id)
        return visible

    def link_rate(self, user_id: int, sat_id: int) -> float:
        distance_km = self._distance(self.user_positions[user_id], self.satellite_positions[sat_id])
        distance_km = max(distance_km, 1e-3)
        snr = self.tx_power / (self.noise_power * (distance_km ** self.pathloss_exp))
        # Shannon-inspired achievable rate.
        return self.bandwidth_hz * math.log2(1.0 + snr)

    def _risk(self, sat_id: int, projected_load: int) -> float:
        current_load = self.satellite_loads[sat_id]
        overload_component = 0.0
        if projected_load > self.sat_capacity:
            overload_component = (projected_load - self.sat_capacity) / projected_load
        occupancy_component = current_load / self.sat_capacity
        risk = overload_component + 0.35 * occupancy_component
        return min(1.0, max(0.0, risk))

    def _action_to_satellite(self, action: int) -> int:
        if action <= 0:
            return -1
        sat_id = action - 1
        if sat_id < 0 or sat_id >= self.num_satellites:
            return -1
        return sat_id

    def evaluate_utility(
        self,
        user_id: int,
        target_sat: int,
        projected_loads: Sequence[int],
        prev_sat: int,
    ) -> float:
        if target_sat < 0:
            return -self.disconnect_loss

        if target_sat not in self.visible_satellites(user_id):
            return -self.disconnect_loss

        rate = self.link_rate(user_id, target_sat)
        risk = self._risk(target_sat, projected_loads[target_sat])
        effective_access = (rate / 1e6) * (1.0 - risk)

        handover_cost = 0.0
        if prev_sat >= 0 and target_sat != prev_sat:
            handover_cost = self.switch_cost

        return effective_access - handover_cost

    def _projected_loads(self, requested_sats: Sequence[int]) -> List[int]:
        projected = [0 for _ in range(self.num_satellites)]
        for sat in requested_sats:
            if sat >= 0:
                projected[sat] += 1
        return projected

    def get_local_observation(self, user_id: int) -> List[float]:
        current_sat = self.current_assignment[user_id]
        current_sat_norm = (current_sat + 1) / self.num_satellites

        visible = set(self.visible_satellites(user_id))
        vis_mask = [1.0 if sat_id in visible else 0.0 for sat_id in range(self.num_satellites)]

        rates = []
        max_rate = self.bandwidth_hz * math.log2(1 + self.tx_power / self.noise_power)
        max_rate = max(max_rate, 1.0)
        for sat_id in range(self.num_satellites):
            if sat_id in visible:
                rates.append(min(1.0, self.link_rate(user_id, sat_id) / max_rate))
            else:
                rates.append(0.0)

        loads = [min(1.0, load / self.sat_capacity) for load in self.satellite_loads]

        return [current_sat_norm] + vis_mask + rates + loads

    def get_local_observations(self) -> List[List[float]]:
        return [self.get_local_observation(user_id) for user_id in range(self.num_users)]

    def get_global_state(self) -> List[float]:
        state: List[float] = []
        for user_id in range(self.num_users):
            state.extend(self.get_local_observation(user_id))
        return state

    def step(self, actions: Sequence[int]) -> Dict[str, object]:
        if len(actions) != self.num_users:
            raise ValueError("actions length does not match num_users")

        requested_sats = [self._action_to_satellite(action) for action in actions]
        projected_loads = self._projected_loads(requested_sats)

        prev_assignment = list(self.current_assignment)
        utilities = [
            self.evaluate_utility(
                user_id=user_id,
                target_sat=requested_sats[user_id],
                projected_loads=projected_loads,
                prev_sat=prev_assignment[user_id],
            )
            for user_id in range(self.num_users)
        ]

        # Deterministic admission control with utility-priority under overload.
        admitted_users = set()
        for sat_id in range(self.num_satellites):
            contenders = [u for u in range(self.num_users) if requested_sats[u] == sat_id]
            if len(contenders) <= self.sat_capacity:
                admitted_users.update(contenders)
                continue
            contenders.sort(key=lambda u: utilities[u], reverse=True)
            admitted_users.update(contenders[: self.sat_capacity])

        new_assignment = [-1 for _ in range(self.num_users)]
        for user_id in range(self.num_users):
            sat = requested_sats[user_id]
            if sat >= 0 and user_id in admitted_users:
                new_assignment[user_id] = sat
            else:
                if sat >= 0 and user_id not in admitted_users:
                    utilities[user_id] -= self.disconnect_loss

        self.current_assignment = new_assignment
        self.satellite_loads = [0 for _ in range(self.num_satellites)]
        for sat in self.current_assignment:
            if sat >= 0:
                self.satellite_loads[sat] += 1

        global_reward = sum(utilities)

        self.current_slot += 1
        self._update_positions()

        done = self.current_slot >= self.max_steps
        return {
            "local_obs": self.get_local_observations(),
            "global_state": self.get_global_state(),
            "reward": global_reward,
            "done": done,
            "info": {
                "utilities": utilities,
                "requested_sats": requested_sats,
                "assignment": list(self.current_assignment),
                "loads": list(self.satellite_loads),
            },
        }
