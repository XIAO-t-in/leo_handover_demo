"""
DQN Agent for LEO Satellite Handover

Implements a Double DQN agent with:
  - Experience replay buffer
  - Target network (soft/hard updates)
  - ε-greedy exploration with exponential decay
  - Gradient clipping for training stability
"""

from __future__ import annotations

import random
from collections import deque
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim


class DQNNetwork(nn.Module):
    """Fully-connected Q-network.

    Parameters
    ----------
    state_dim : int
        Dimensionality of the input state vector.
    action_dim : int
        Number of discrete actions.
    hidden_dims : tuple of int
        Sizes of the hidden layers.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        in_dim = state_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_dim, h), nn.ReLU()]
            in_dim = h
        layers.append(nn.Linear(in_dim, action_dim))
        self.network = nn.Sequential(*layers)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ReplayBuffer:
    """Circular experience replay buffer.

    Parameters
    ----------
    capacity : int
        Maximum number of transitions stored.
    """

    def __init__(self, capacity: int = 100_000) -> None:
        self.buffer: deque = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        self.buffer.append((state, action, reward, next_state, done))

    def sample(
        self, batch_size: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class DQNAgent:
    """Double DQN agent for LEO satellite handover decisions.

    Parameters
    ----------
    state_dim : int
        Dimension of the observation vector.
    action_dim : int
        Number of satellite choices (actions).
    lr : float
        Adam learning rate.
    gamma : float
        Discount factor.
    epsilon_start : float
        Initial exploration probability.
    epsilon_end : float
        Minimum exploration probability after decay.
    epsilon_decay : float
        Multiplicative decay factor applied after each episode.
    batch_size : int
        Mini-batch size for gradient updates.
    buffer_capacity : int
        Replay buffer capacity.
    target_update_freq : int
        Number of gradient updates between hard target-network copies.
    hidden_dims : tuple of int
        Hidden layer sizes for the Q-network.
    device : str or None
        PyTorch device string; auto-detected when ``None``.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-4,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.995,
        batch_size: int = 64,
        buffer_capacity: int = 100_000,
        target_update_freq: int = 10,
        hidden_dims: Tuple[int, ...] = (256, 256),
        device: Optional[str] = None,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = torch.device(
            device if device is not None
            else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.policy_net = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net = DQNNetwork(state_dim, action_dim, hidden_dims).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self.update_count = 0
        self.total_steps = 0

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Return an action via ε-greedy policy.

        Parameters
        ----------
        state : np.ndarray
            Current observation vector.
        training : bool
            When ``True`` applies ε-greedy exploration; when ``False`` acts
            greedily (evaluation mode).
        """
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state_t = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q = self.policy_net(state_t)
        return int(q.argmax(dim=1).item())

    # ------------------------------------------------------------------
    # Learning
    # ------------------------------------------------------------------

    def push_transition(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: float,
    ) -> None:
        """Store a (s, a, r, s', done) transition in the replay buffer."""
        self.replay_buffer.push(state, action, reward, next_state, done)
        self.total_steps += 1

    def update(self) -> Optional[float]:
        """Sample a mini-batch and perform one gradient update.

        Returns the scalar loss value, or ``None`` if the buffer is not yet
        large enough to fill a batch.
        """
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            self.batch_size
        )

        states_t = torch.as_tensor(states, dtype=torch.float32).to(self.device)
        actions_t = torch.as_tensor(actions, dtype=torch.long).to(self.device)
        rewards_t = torch.as_tensor(rewards, dtype=torch.float32).to(self.device)
        next_states_t = torch.as_tensor(next_states, dtype=torch.float32).to(self.device)
        dones_t = torch.as_tensor(dones, dtype=torch.float32).to(self.device)

        # Current Q-values for chosen actions
        q_values = (
            self.policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
        )

        # Double DQN target: policy net selects action, target net evaluates it
        with torch.no_grad():
            next_actions = self.policy_net(next_states_t).argmax(dim=1)
            next_q = (
                self.target_net(next_states_t)
                .gather(1, next_actions.unsqueeze(1))
                .squeeze(1)
            )
            targets = rewards_t + self.gamma * next_q * (1.0 - dones_t)

        loss = nn.SmoothL1Loss()(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 10.0)
        self.optimizer.step()

        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return float(loss.item())

    def decay_epsilon(self) -> None:
        """Apply one step of exponential ε-decay."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    # ------------------------------------------------------------------
    # Checkpoint I/O
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save agent state to *path*."""
        torch.save(
            {
                "policy_net": self.policy_net.state_dict(),
                "target_net": self.target_net.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "update_count": self.update_count,
                "total_steps": self.total_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        """Load agent state from *path*."""
        ckpt = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(ckpt["policy_net"])
        self.target_net.load_state_dict(ckpt["target_net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.epsilon = ckpt["epsilon"]
        self.update_count = ckpt["update_count"]
        self.total_steps = ckpt["total_steps"]
