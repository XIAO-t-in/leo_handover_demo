"""
Evaluation script: compare the trained DQN agent against traditional baselines.

Usage
-----
    python evaluate.py                         # use defaults
    python evaluate.py --model_dir checkpoints --output_dir results
    python evaluate.py --help
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Callable, Dict, List

import numpy as np

from src.environment import LEOHandoverEnv
from src.dqn_agent import DQNAgent
from src.traditional_handover import (
    BestRSRPPolicy,
    HysteresisRSRPPolicy,
    MaxElevationPolicy,
    LongestRemainingTimePolicy,
    TraditionalHandoverPolicy,
)
from src.utils import plot_comparison, plot_training_curves


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _run_episodes(
    env: LEOHandoverEnv,
    action_fn: Callable[[np.ndarray], int],
    n_episodes: int,
    seed_offset: int = 1000,
) -> Dict[str, float]:
    """Run *n_episodes* and return aggregate performance metrics."""
    rewards: List[float] = []
    handovers: List[int] = []
    outages: List[int] = []
    avg_rsrps: List[float] = []

    for ep in range(n_episodes):
        state, _ = env.reset(seed=seed_offset + ep)
        ep_reward = 0.0

        while True:
            action = action_fn(state)
            state, reward, terminated, truncated, info = env.step(action)
            ep_reward += reward
            if terminated or truncated:
                break

        avg_rsrp = env.total_rsrp_sum / max(env.current_step, 1)
        rewards.append(ep_reward)
        handovers.append(info["n_handovers"])
        outages.append(info["outage_steps"])
        avg_rsrps.append(avg_rsrp)

    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_handovers": float(np.mean(handovers)),
        "mean_outage_steps": float(np.mean(outages)),
        "mean_avg_rsrp": float(np.mean(avg_rsrps)),
    }


def evaluate_dqn(
    env: LEOHandoverEnv,
    agent: DQNAgent,
    n_episodes: int = 20,
    seed_offset: int = 1000,
) -> Dict[str, float]:
    """Evaluate the DQN agent in greedy (non-training) mode."""
    return _run_episodes(
        env,
        lambda s: agent.select_action(s, training=False),
        n_episodes,
        seed_offset,
    )


def evaluate_traditional(
    env: LEOHandoverEnv,
    policy: TraditionalHandoverPolicy,
    n_episodes: int = 20,
    seed_offset: int = 1000,
) -> Dict[str, float]:
    """Evaluate a traditional rule-based handover policy."""
    policy.reset()

    def action_fn(state: np.ndarray) -> int:  # noqa: ARG001
        visible = env._get_visible_satellites()
        return policy.select_satellite(visible) if visible else 0

    return _run_episodes(env, action_fn, n_episodes, seed_offset)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args: argparse.Namespace) -> None:
    env = LEOHandoverEnv(
        n_satellites=args.n_satellites,
        max_visible=args.max_visible,
        time_step_s=args.time_step,
        handover_penalty=args.handover_penalty,
    )

    state_dim = env.observation_space.shape[0]
    action_dim = int(env.action_space.n)
    results: Dict[str, Dict[str, float]] = {}

    # ---- DQN ---------------------------------------------------------------
    model_path = os.path.join(args.model_dir, "best_model.pth")
    if os.path.exists(model_path):
        agent = DQNAgent(state_dim=state_dim, action_dim=action_dim)
        agent.load(model_path)
        print("Evaluating DQN agent …")
        results["DQN"] = evaluate_dqn(env, agent, n_episodes=args.n_episodes)
    else:
        print(f"[warn] No trained model found at '{model_path}' — skipping DQN evaluation.")

    # ---- Traditional baselines --------------------------------------------
    policies: List[TraditionalHandoverPolicy] = [
        BestRSRPPolicy(hysteresis_db=0.0),
        HysteresisRSRPPolicy(hysteresis_db=3.0),
        MaxElevationPolicy(),
        LongestRemainingTimePolicy(),
    ]
    for policy in policies:
        print(f"Evaluating {policy.name} …")
        results[policy.name] = evaluate_traditional(env, policy, n_episodes=args.n_episodes)

    # ---- Print summary table -----------------------------------------------
    print("\n" + "=" * 84)
    print(
        f"{'Policy':<32} {'Reward':>10} {'Handovers':>12}"
        f" {'RSRP(dBm)':>12} {'Outages':>10}"
    )
    print("-" * 84)
    for name, res in results.items():
        print(
            f"{name:<32} "
            f"{res['mean_reward']:>10.2f} "
            f"{res['mean_handovers']:>12.1f} "
            f"{res['mean_avg_rsrp']:>12.2f} "
            f"{res['mean_outage_steps']:>10.1f}"
        )
    print("=" * 84)

    # ---- Save results and plots -------------------------------------------
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to '{args.output_dir}/results.json'")

    if results:
        plot_comparison(results, args.output_dir)

    metrics_path = os.path.join(args.model_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            metrics = json.load(f)
        plot_training_curves(metrics, args.output_dir)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Evaluate LEO handover policies (DQN vs baselines)"
    )
    p.add_argument("--model_dir", type=str, default="checkpoints")
    p.add_argument("--output_dir", type=str, default="results")
    p.add_argument("--n_episodes", type=int, default=20,
                   help="Evaluation episodes per policy (default: 20)")
    p.add_argument("--n_satellites", type=int, default=20)
    p.add_argument("--max_visible", type=int, default=6)
    p.add_argument("--time_step", type=float, default=10.0)
    p.add_argument("--handover_penalty", type=float, default=0.5)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
