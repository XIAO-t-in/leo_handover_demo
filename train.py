"""Simple training entrypoint for QMIX-based LEO handover strategy."""

from __future__ import annotations

from src.environment import LeoHandoverEnv
from src.qmix import QMIXLearner


def train(episodes: int = 5, steps_per_episode: int = 30) -> None:
    env = LeoHandoverEnv(num_users=3, num_satellites=5, sat_capacity=2, max_steps=steps_per_episode)
    learner = QMIXLearner(
        num_agents=env.num_users,
        obs_dim=env.obs_dim,
        action_dim=env.action_dim,
        state_dim=env.state_dim,
        lr=5e-4,
    )

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0.0
        metrics = {}

        for step in range(steps_per_episode):
            local_obs = state["local_obs"]
            global_state = state["global_state"]

            actions = learner.act(local_obs=local_obs, epsilon=0.1)
            next_state = env.step(actions)

            transition = {
                "obs": local_obs,
                "actions": actions,
                "reward": next_state["reward"],
                "next_obs": next_state["local_obs"],
                "state": global_state,
                "next_state": next_state["global_state"],
                "done": next_state["done"],
            }
            metrics = learner.train_step([transition])
            total_reward += next_state["reward"]
            state = next_state

            if next_state["done"]:
                break

        print(
            f"Episode {episode + 1}: total_reward={total_reward:.3f}, "
            f"last_loss={metrics.get('loss', float('nan')):.6f}"
        )


if __name__ == "__main__":
    train()
