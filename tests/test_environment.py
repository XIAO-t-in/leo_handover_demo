import unittest

from src.environment import LeoHandoverEnv


class TestLeoHandoverEnv(unittest.TestCase):
    def test_reset_produces_consistent_shapes(self):
        env = LeoHandoverEnv(num_users=2, num_satellites=4, sat_capacity=1, seed=42)
        state = env.reset()

        self.assertEqual(len(state["local_obs"]), 2)
        self.assertEqual(len(state["local_obs"][0]), env.obs_dim)
        self.assertEqual(len(state["global_state"]), env.state_dim)

    def test_visibility_and_rate_positive_for_visible_links(self):
        env = LeoHandoverEnv(num_users=1, num_satellites=5, seed=1)
        env.reset()

        visible = env.visible_satellites(0)
        self.assertGreater(len(visible), 0)
        for sat_id in visible:
            self.assertGreater(env.link_rate(0, sat_id), 0.0)

    def test_overload_creates_admission_failure(self):
        env = LeoHandoverEnv(num_users=2, num_satellites=1, sat_capacity=1, seed=7)
        env.reset()

        # Both users request the only satellite; one must fail admission.
        result = env.step([1, 1])
        assignments = result["info"]["assignment"]
        connected_count = sum(1 for sat in assignments if sat >= 0)
        self.assertEqual(connected_count, 1)

    def test_switch_cost_reduces_utility(self):
        env = LeoHandoverEnv(num_users=1, num_satellites=2, sat_capacity=1, seed=9)
        env.reset()

        visible = env.visible_satellites(0)
        self.assertGreater(len(visible), 0)
        target_sat = visible[0]
        projected = [0 for _ in range(env.num_satellites)]
        projected[target_sat] = 1

        stay_utility = env.evaluate_utility(user_id=0, target_sat=target_sat, projected_loads=projected, prev_sat=target_sat)
        switch_utility = env.evaluate_utility(user_id=0, target_sat=target_sat, projected_loads=projected, prev_sat=(1 - target_sat))
        self.assertAlmostEqual(stay_utility - switch_utility, env.switch_cost, places=6)


if __name__ == "__main__":
    unittest.main()
