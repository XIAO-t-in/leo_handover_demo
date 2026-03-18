"""
Utility helpers: visualisation and statistics for the LEO handover demo.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_satellite_positions(
    env: Any,
    step: int = 0,
    output_path: Optional[str] = None,
) -> None:
    """Render a top-down view of the satellite constellation.

    Visible satellites are highlighted in red; the ground user is shown as a
    green triangle on the Earth's surface.

    Parameters
    ----------
    env :
        A :class:`~src.environment.LEOHandoverEnv` instance.
    step : int
        Current simulation step (used for the plot title only).
    output_path : str or None
        File path to save the figure.  When ``None`` the figure is displayed.
    """
    fig, ax = plt.subplots(figsize=(9, 9))

    # Earth
    earth = plt.Circle((0, 0), env.EARTH_RADIUS_KM, color="lightblue", alpha=0.7, label="Earth")
    ax.add_patch(earth)

    # Orbital ring
    orbit = plt.Circle(
        (0, 0), env.orbital_radius_km, fill=False, color="grey", linestyle="--", alpha=0.4
    )
    ax.add_patch(orbit)

    # Ground user at (Re, 0)
    ax.plot(env.EARTH_RADIUS_KM, 0, "g^", markersize=14, zorder=6, label="User Terminal")

    # Visibility cone boundaries
    max_a = env.max_central_angle_rad
    for sign in (+1, -1):
        sx = env.orbital_radius_km * np.cos(sign * max_a)
        sy = env.orbital_radius_km * np.sin(sign * max_a)
        ax.plot(
            [env.EARTH_RADIUS_KM, sx], [0.0, sy],
            "g--", alpha=0.35, linewidth=1,
        )

    visible = env._get_visible_satellites()
    visible_ids = {s["sat_id"] for s in visible}

    for i in range(env.n_satellites):
        angle = env.sat_angles[i]
        sx = env.orbital_radius_km * np.cos(angle)
        sy = env.orbital_radius_km * np.sin(angle)
        if i in visible_ids:
            color = "crimson" if i == env.connected_sat_id else "orangered"
            ax.plot(sx, sy, "o", color=color, markersize=9, zorder=5)
            ax.annotate(
                f"S{i}", (sx, sy), textcoords="offset points",
                xytext=(5, 5), fontsize=7, color=color,
            )
        else:
            ax.plot(sx, sy, "o", color="dimgray", markersize=5, alpha=0.3, zorder=3)

    lim = env.orbital_radius_km * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_title(
        f"LEO Constellation — Step {step}  "
        f"(red = visible, crimson = connected)",
        fontsize=11,
    )
    ax.set_xlabel("Distance (km)")
    ax.set_ylabel("Distance (km)")
    ax.grid(True, alpha=0.25)

    if output_path:
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()
    else:
        plt.show()


def plot_training_curves(metrics: Dict[str, List[float]], output_dir: str) -> None:
    """Plot episode-level training metrics and save to *output_dir*.

    Parameters
    ----------
    metrics : dict
        Dictionary produced by :func:`train.train` with keys
        ``episode_rewards``, ``episode_handovers``, ``episode_outage_steps``,
        ``losses``.
    output_dir : str
        Directory where ``training_curves.png`` is written.
    """
    os.makedirs(output_dir, exist_ok=True)

    def smooth(data: List[float], window: int = 50) -> np.ndarray:
        if len(data) < window:
            return np.array(data)
        return np.convolve(data, np.ones(window) / window, mode="valid")

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("DQN Training Progress — LEO Satellite Handover", fontsize=13)

    specs = [
        ("episode_rewards", "Episode Reward", "steelblue"),
        ("episode_handovers", "Handovers per Episode", "coral"),
        ("episode_outage_steps", "Outage Steps per Episode", "seagreen"),
        ("losses", "Training Loss (log scale)", "mediumpurple"),
    ]

    for ax, (key, title, color) in zip(axes.flat, specs):
        data = metrics.get(key, [])
        if not data:
            ax.set_visible(False)
            continue
        xs = range(len(data))
        ax.plot(xs, data, alpha=0.25, color=color)
        sm = smooth(data)
        ax.plot(range(len(sm)), sm, color=color, linewidth=2, label="smoothed (w=50)")
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        if key == "losses":
            ax.set_yscale("log")

    plt.tight_layout()
    out = os.path.join(output_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Training curves → {out}")


def plot_comparison(results: Dict[str, Dict[str, float]], output_dir: str) -> None:
    """Bar-chart comparison of DQN vs traditional policies.

    Parameters
    ----------
    results : dict
        Mapping of policy name → evaluation metrics dict.
    output_dir : str
        Directory where ``comparison.png`` is written.
    """
    os.makedirs(output_dir, exist_ok=True)

    names = list(results.keys())
    metrics_cfg = [
        ("mean_reward", "Avg Episode Reward", "↑ higher is better"),
        ("mean_handovers", "Avg Handovers", "↓ lower is better"),
        ("mean_avg_rsrp", "Avg RSRP (dBm)", "↑ higher is better"),
        ("mean_outage_steps", "Avg Outage Steps", "↓ lower is better"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("DQN vs Traditional Handover Policies", fontsize=13)
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))  # type: ignore[attr-defined]

    for ax, (mkey, title, note) in zip(axes.flat, metrics_cfg):
        values = [results[n].get(mkey, 0.0) for n in names]
        bars = ax.bar(names, values, color=colors)
        ax.set_title(f"{title}\n({note})", fontsize=10)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, val in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{val:.1f}",
                ha="center", va="bottom", fontsize=8,
            )

    plt.tight_layout()
    out = os.path.join(output_dir, "comparison.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Comparison chart → {out}")


def compute_statistics(episode_data: List[Dict[str, Any]]) -> Dict[str, float]:
    """Summarise a list of per-episode result dicts."""
    rewards = [ep["reward"] for ep in episode_data]
    handovers = [ep["n_handovers"] for ep in episode_data]
    outages = [ep["outage_steps"] for ep in episode_data]
    rsrps = [ep["avg_rsrp"] for ep in episode_data]
    return {
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_handovers": float(np.mean(handovers)),
        "std_handovers": float(np.std(handovers)),
        "mean_outage_steps": float(np.mean(outages)),
        "mean_avg_rsrp": float(np.mean(rsrps)),
    }
