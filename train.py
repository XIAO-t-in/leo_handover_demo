"""
Training script for the DQN-based LEO satellite handover agent.

Usage
-----
    python train.py                          # use defaults
    python train.py --n_episodes 1000 --lr 3e-4
    python train.py --help                   # show all options
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from collections import deque
from typing import Any, Dict

import numpy as np
import torch

from src.environment import LEOHandoverEnv
from src.dqn_agent import DQNAgent


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the full training loop and return episode-level metrics."""
    set_seed(args.seed)

    env = LEOHandoverEnv(
        n_satellites=args.n_satellites,
        max_visible=args.max_visible,
        time_step_s=args.time_step,
        handover_penalty=args.handover_penalty,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = int(env.action_space.n)

    print("=" * 60)
    print("LEO Satellite Handover — DQN Training")
    print("=" * 60)
    print(f"  State dim        : {state_dim}")
    print(f"  Action dim       : {action_dim}")
    print(f"  Max steps/episode: {env.max_steps}")
    print(f"  Orbital period   : {env.orbital_period_s:.0f} s "
          f"({env.orbital_period_s / 60:.1f} min)")
    print(f"  Device           : {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    print("=" * 60)

    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        lr=args.lr,
        gamma=args.gamma,
        epsilon_start=args.epsilon_start,
        epsilon_end=args.epsilon_end,
        epsilon_decay=args.epsilon_decay,
        batch_size=args.batch_size,
        buffer_capacity=args.buffer_capacity,
        target_update_freq=args.target_update_freq,
    )

    metrics: Dict[str, list] = {
        "episode_rewards": [],
        "episode_handovers": [],
        "episode_outage_steps": [],
        "episode_avg_rsrp": [],
        "losses": [],
    }

    recent_rewards: deque = deque(maxlen=50)
    best_reward = -float("inf")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"\nStarting training for {args.n_episodes} episodes …\n")
    t0 = time.time()

    for episode in range(1, args.n_episodes + 1):
        state, _ = env.reset(seed=args.seed + episode)
        episode_reward = 0.0
        episode_losses: list[float] = []

        while True:
            action = agent.select_action(state, training=True)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.push_transition(state, action, reward, next_state, float(done))
            loss = agent.update()
            if loss is not None:
                episode_losses.append(loss)

            episode_reward += reward
            state = next_state
            if done:
                break

        agent.decay_epsilon()
        recent_rewards.append(episode_reward)

        avg_rsrp = env.total_rsrp_sum / max(env.current_step, 1)
        metrics["episode_rewards"].append(episode_reward)
        metrics["episode_handovers"].append(info["n_handovers"])
        metrics["episode_outage_steps"].append(info["outage_steps"])
        metrics["episode_avg_rsrp"].append(avg_rsrp)
        metrics["losses"].append(float(np.mean(episode_losses)) if episode_losses else 0.0)

        if episode_reward > best_reward:
            best_reward = episode_reward
            agent.save(os.path.join(args.save_dir, "best_model.pth"))

        if episode % args.log_interval == 0:
            elapsed = time.time() - t0
            avg50 = float(np.mean(recent_rewards))
            print(
                f"Ep {episode:5d}/{args.n_episodes}  "
                f"reward={episode_reward:8.2f}  "
                f"avg50={avg50:8.2f}  "
                f"handovers={info['n_handovers']:4d}  "
                f"ε={agent.epsilon:.3f}  "
                f"t={elapsed:.0f}s"
            )

    agent.save(os.path.join(args.save_dir, "final_model.pth"))
    with open(os.path.join(args.save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nTraining finished.  Best episode reward: {best_reward:.2f}")
    print(f"Checkpoints saved to '{args.save_dir}/'")
    return metrics


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train DQN agent for LEO satellite handover"
    )
    p.add_argument("--n_episodes", type=int, default=500,
                   help="Number of training episodes (default: 500)")
    p.add_argument("--n_satellites", type=int, default=20,
                   help="Total satellites in constellation (default: 20)")
    p.add_argument("--max_visible", type=int, default=6,
                   help="Max simultaneously visible satellites (default: 6)")
    p.add_argument("--time_step", type=float, default=10.0,
                   help="Simulation time step in seconds (default: 10)")
    p.add_argument("--handover_penalty", type=float, default=0.5,
                   help="Reward penalty per handover (default: 0.5)")
    p.add_argument("--lr", type=float, default=1e-4,
                   help="Adam learning rate (default: 1e-4)")
    p.add_argument("--gamma", type=float, default=0.99,
                   help="Discount factor (default: 0.99)")
    p.add_argument("--epsilon_start", type=float, default=1.0)
    p.add_argument("--epsilon_end", type=float, default=0.05)
    p.add_argument("--epsilon_decay", type=float, default=0.995)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--buffer_capacity", type=int, default=100_000)
    p.add_argument("--target_update_freq", type=int, default=10)
    p.add_argument("--log_interval", type=int, default=50,
                   help="Print progress every N episodes (default: 50)")
    p.add_argument("--save_dir", type=str, default="checkpoints",
                   help="Directory to save model checkpoints (default: checkpoints)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
