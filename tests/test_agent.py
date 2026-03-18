"""Tests for the DQN agent (network, replay buffer, agent)."""

from __future__ import annotations

import os

import numpy as np
import pytest
import torch

from src.dqn_agent import DQNAgent, DQNNetwork, ReplayBuffer


STATE_DIM = 24   # 6 satellites × 4 features
ACTION_DIM = 6


# ---------------------------------------------------------------------------
# DQNNetwork
# ---------------------------------------------------------------------------

class TestDQNNetwork:
    def test_output_shape(self) -> None:
        net = DQNNetwork(STATE_DIM, ACTION_DIM)
        x = torch.randn(8, STATE_DIM)
        out = net(x)
        assert out.shape == (8, ACTION_DIM)

    def test_single_sample(self) -> None:
        net = DQNNetwork(STATE_DIM, ACTION_DIM, hidden_dims=(128,))
        x = torch.randn(1, STATE_DIM)
        out = net(x)
        assert out.shape == (1, ACTION_DIM)

    def test_custom_hidden_dims(self) -> None:
        net = DQNNetwork(STATE_DIM, ACTION_DIM, hidden_dims=(64, 64, 64))
        x = torch.randn(4, STATE_DIM)
        out = net(x)
        assert out.shape == (4, ACTION_DIM)

    def test_no_nan_in_output(self) -> None:
        net = DQNNetwork(STATE_DIM, ACTION_DIM)
        x = torch.randn(16, STATE_DIM)
        out = net(x)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# ReplayBuffer
# ---------------------------------------------------------------------------

class TestReplayBuffer:
    def _make_transition(self) -> tuple:
        return (
            np.zeros(STATE_DIM, dtype=np.float32),
            0,
            1.0,
            np.ones(STATE_DIM, dtype=np.float32),
            0.0,
        )

    def test_push_and_length(self) -> None:
        buf = ReplayBuffer(100)
        for _ in range(50):
            buf.push(*self._make_transition())
        assert len(buf) == 50

    def test_sample_shapes(self) -> None:
        buf = ReplayBuffer(1000)
        for _ in range(200):
            buf.push(*self._make_transition())
        states, actions, rewards, next_states, dones = buf.sample(32)
        assert states.shape == (32, STATE_DIM)
        assert actions.shape == (32,)
        assert rewards.shape == (32,)
        assert next_states.shape == (32, STATE_DIM)
        assert dones.shape == (32,)

    def test_capacity_limit(self) -> None:
        buf = ReplayBuffer(50)
        for _ in range(100):
            buf.push(*self._make_transition())
        assert len(buf) == 50

    def test_sample_raises_when_too_small(self) -> None:
        buf = ReplayBuffer(10)
        buf.push(*self._make_transition())
        with pytest.raises(ValueError):
            buf.sample(32)


# ---------------------------------------------------------------------------
# DQNAgent
# ---------------------------------------------------------------------------

@pytest.fixture
def agent() -> DQNAgent:
    return DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, device="cpu")


class TestDQNAgent:
    def test_greedy_action_in_range(self, agent: DQNAgent) -> None:
        state = np.zeros(STATE_DIM, dtype=np.float32)
        action = agent.select_action(state, training=False)
        assert 0 <= action < ACTION_DIM

    def test_exploration_covers_multiple_actions(self, agent: DQNAgent) -> None:
        agent.epsilon = 1.0
        state = np.zeros(STATE_DIM, dtype=np.float32)
        actions = {agent.select_action(state, training=True) for _ in range(200)}
        assert len(actions) > 1

    def test_update_returns_none_when_buffer_empty(self, agent: DQNAgent) -> None:
        assert agent.update() is None

    def test_update_returns_loss_after_warmup(self, agent: DQNAgent) -> None:
        rng = np.random.default_rng(0)
        for _ in range(100):
            s = rng.standard_normal(STATE_DIM).astype(np.float32)
            ns = rng.standard_normal(STATE_DIM).astype(np.float32)
            agent.push_transition(s, 0, 1.0, ns, 0.0)
        loss = agent.update()
        assert loss is not None
        assert loss >= 0.0

    def test_epsilon_decay(self, agent: DQNAgent) -> None:
        agent.epsilon = 1.0
        for _ in range(10):
            agent.decay_epsilon()
        assert agent.epsilon < 1.0
        assert agent.epsilon >= agent.epsilon_end

    def test_epsilon_floor(self, agent: DQNAgent) -> None:
        agent.epsilon = agent.epsilon_end + 1e-9
        agent.decay_epsilon()
        assert agent.epsilon >= agent.epsilon_end

    def test_save_and_load(self, agent: DQNAgent, tmp_path: str) -> None:
        path = os.path.join(str(tmp_path), "ckpt.pth")
        original_eps = agent.epsilon
        agent.save(path)

        new_agent = DQNAgent(state_dim=STATE_DIM, action_dim=ACTION_DIM, device="cpu")
        new_agent.load(path)
        assert new_agent.epsilon == original_eps
        assert new_agent.update_count == agent.update_count
        assert new_agent.total_steps == agent.total_steps

    def test_target_net_updated(self, agent: DQNAgent) -> None:
        """After target_update_freq gradient steps, target net weights should
        match the policy net."""
        rng = np.random.default_rng(1)
        # Fill buffer
        for _ in range(200):
            s = rng.standard_normal(STATE_DIM).astype(np.float32)
            ns = rng.standard_normal(STATE_DIM).astype(np.float32)
            agent.push_transition(s, rng.integers(0, ACTION_DIM), 1.0, ns, 0.0)
        # Run target_update_freq updates to trigger a hard update
        for _ in range(agent.target_update_freq):
            agent.update()
        for p, t in zip(agent.policy_net.parameters(), agent.target_net.parameters()):
            assert torch.allclose(p.data, t.data)
