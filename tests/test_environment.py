"""Tests for the LEO satellite handover simulation environment."""

from __future__ import annotations

import numpy as np
import pytest

from src.environment import LEOHandoverEnv


@pytest.fixture
def env() -> LEOHandoverEnv:
    return LEOHandoverEnv(n_satellites=20, max_visible=6)


# ---------------------------------------------------------------------------
# Observation / action spaces
# ---------------------------------------------------------------------------

class TestObservationSpace:
    def test_shape(self, env: LEOHandoverEnv) -> None:
        obs, _ = env.reset(seed=0)
        assert obs.shape == (env.max_visible * 4,)

    def test_range(self, env: LEOHandoverEnv) -> None:
        obs, _ = env.reset(seed=0)
        assert np.all(obs >= -1.0) and np.all(obs <= 1.0)

    def test_dtype(self, env: LEOHandoverEnv) -> None:
        obs, _ = env.reset(seed=0)
        assert obs.dtype == np.float32

    def test_action_space_size(self, env: LEOHandoverEnv) -> None:
        assert env.action_space.n == env.max_visible


# ---------------------------------------------------------------------------
# Step / episode mechanics
# ---------------------------------------------------------------------------

class TestStepMechanics:
    def test_step_return_types(self, env: LEOHandoverEnv) -> None:
        env.reset(seed=1)
        obs, reward, terminated, truncated, info = env.step(0)
        assert isinstance(obs, np.ndarray)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_episode_truncates_at_max_steps(self, env: LEOHandoverEnv) -> None:
        env.reset(seed=2)
        done = False
        steps = 0
        while not done:
            _, _, terminated, truncated, _ = env.step(0)
            done = terminated or truncated
            steps += 1
            if steps > env.max_steps + 10:
                break
        assert done, "Episode should end by truncation"
        assert steps <= env.max_steps + 1

    def test_step_count_increments(self, env: LEOHandoverEnv) -> None:
        env.reset(seed=3)
        for i in range(1, 6):
            env.step(0)
            assert env.current_step == i

    def test_handover_counter(self, env: LEOHandoverEnv) -> None:
        env.reset(seed=4)
        # Alternate between action 0 and 1 to encourage handovers
        for _ in range(30):
            _, _, t, tr, info = env.step(0)
            if t or tr:
                break
            _, _, t, tr, info = env.step(1)
            if t or tr:
                break
        assert info["n_handovers"] >= 0

    def test_outage_steps_tracked(self, env: LEOHandoverEnv) -> None:
        env.reset(seed=5)
        for _ in range(10):
            _, _, t, tr, info = env.step(0)
            if t or tr:
                break
        assert "outage_steps" in info


# ---------------------------------------------------------------------------
# Physics helpers
# ---------------------------------------------------------------------------

class TestPhysics:
    def test_elevation_overhead(self, env: LEOHandoverEnv) -> None:
        """Satellite directly overhead (γ = 0) should have elevation ≈ 90°."""
        env.reset(seed=0)
        el = env._elevation_angle(0.0)
        assert abs(el - 90.0) < 1.0

    def test_elevation_decreases_with_central_angle(self, env: LEOHandoverEnv) -> None:
        env.reset(seed=0)
        el_near = env._elevation_angle(np.radians(5))
        el_far = env._elevation_angle(np.radians(20))
        assert el_near > el_far

    def test_elevation_below_horizon(self, env: LEOHandoverEnv) -> None:
        """A satellite far from the user should be below the horizon."""
        env.reset(seed=0)
        el = env._elevation_angle(np.pi / 2)  # 90° central angle
        assert el < env.min_elevation_deg

    def test_rsrp_decreases_with_distance(self, env: LEOHandoverEnv) -> None:
        env.reset(seed=0)
        rsrp_near = env._compute_rsrp(80.0, 560.0)
        rsrp_far = env._compute_rsrp(15.0, 1500.0)
        assert rsrp_near > rsrp_far

    def test_slant_range_overhead(self, env: LEOHandoverEnv) -> None:
        """Slant range at γ = 0 should equal the orbital altitude."""
        env.reset(seed=0)
        d = env._slant_range_km(0.0)
        assert abs(d - env.altitude_km) < 1.0

    def test_visible_satellites_sorted_by_rsrp(self, env: LEOHandoverEnv) -> None:
        env.reset(seed=0)
        visible = env._get_visible_satellites()
        if len(visible) >= 2:
            rsrps = [s["rsrp"] for s in visible]
            assert rsrps == sorted(rsrps, reverse=True)


# ---------------------------------------------------------------------------
# Reset reproducibility
# ---------------------------------------------------------------------------

class TestReset:
    def test_same_seed_same_obs(self, env: LEOHandoverEnv) -> None:
        obs1, _ = env.reset(seed=99)
        obs2, _ = env.reset(seed=99)
        np.testing.assert_array_equal(obs1, obs2)

    def test_different_seeds_different_obs(self, env: LEOHandoverEnv) -> None:
        obs1, _ = env.reset(seed=1)
        obs2, _ = env.reset(seed=2)
        assert not np.array_equal(obs1, obs2)

    def test_reset_clears_counters(self, env: LEOHandoverEnv) -> None:
        env.reset(seed=0)
        for _ in range(5):
            env.step(1)
        env.reset(seed=0)
        assert env.n_handovers == 0
        assert env.outage_steps == 0
        assert env.current_step == 0
