import copy
import math
import unittest

from src.environment import LeoHandoverEnv
from src.qmix import QMIXLearner


class TestQMIXLearner(unittest.TestCase):
    def setUp(self):
        self.env = LeoHandoverEnv(num_users=2, num_satellites=3, sat_capacity=1, seed=11)
        self.env.reset()
        self.learner = QMIXLearner(
            num_agents=self.env.num_users,
            obs_dim=self.env.obs_dim,
            action_dim=self.env.action_dim,
            state_dim=self.env.state_dim,
            seed=11,
        )

    def test_decentralized_action_shape(self):
        obs = self.env.get_local_observations()
        actions = self.learner.act(obs, epsilon=0.0)

        self.assertEqual(len(actions), self.env.num_users)
        for action in actions:
            self.assertGreaterEqual(action, 0)
            self.assertLess(action, self.env.action_dim)

    def test_train_step_returns_finite_metrics_and_updates_weights(self):
        state = self.env.reset()
        obs = state["local_obs"]
        actions = [1, 1]
        next_state = self.env.step(actions)

        transition = {
            "obs": obs,
            "actions": actions,
            "reward": next_state["reward"],
            "next_obs": next_state["local_obs"],
            "state": state["global_state"],
            "next_state": next_state["global_state"],
            "done": next_state["done"],
        }

        pre_w = copy.deepcopy(self.learner.agent_nets[0].w2)
        metrics = self.learner.train_step([transition])

        self.assertIn("loss", metrics)
        self.assertTrue(math.isfinite(metrics["loss"]))

        post_w = self.learner.agent_nets[0].w2
        changed = any(
            abs(pre_w[i][j] - post_w[i][j]) > 1e-12
            for i in range(len(pre_w))
            for j in range(len(pre_w[i]))
        )
        self.assertTrue(changed)


if __name__ == "__main__":
    unittest.main()
