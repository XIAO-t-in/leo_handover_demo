"""Minimal QMIX implementation with centralized training and decentralized execution."""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple


def _rand_matrix(rows: int, cols: int, scale: float = 0.1, rng: random.Random | None = None) -> List[List[float]]:
    rand = rng or random
    return [[rand.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


def _rand_vector(size: int, scale: float = 0.1, rng: random.Random | None = None) -> List[float]:
    rand = rng or random
    return [rand.uniform(-scale, scale) for _ in range(size)]


def _matvec(x: Sequence[float], w: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
    out: List[float] = []
    for col in range(len(b)):
        value = b[col]
        for row in range(len(x)):
            value += x[row] * w[row][col]
        out.append(value)
    return out


def _relu(x: Sequence[float]) -> List[float]:
    return [v if v > 0.0 else 0.0 for v in x]


def _relu_grad(x: Sequence[float]) -> List[float]:
    return [1.0 if v > 0.0 else 0.0 for v in x]


@dataclass
class AgentCache:
    x: List[List[float]]
    z1: List[List[float]]
    h1: List[List[float]]


class AgentQNetwork:
    """Per-agent local value network Q_i(o_i, a_i)."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int = 32, seed: int | None = None) -> None:
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.rng = random.Random(seed)

        self.w1 = _rand_matrix(obs_dim, hidden_dim, rng=self.rng)
        self.b1 = _rand_vector(hidden_dim, rng=self.rng)
        self.w2 = _rand_matrix(hidden_dim, action_dim, rng=self.rng)
        self.b2 = _rand_vector(action_dim, rng=self.rng)

    def forward(self, obs_batch: Sequence[Sequence[float]]) -> Tuple[List[List[float]], AgentCache]:
        z1_batch: List[List[float]] = []
        h1_batch: List[List[float]] = []
        q_batch: List[List[float]] = []

        for obs in obs_batch:
            z1 = _matvec(obs, self.w1, self.b1)
            h1 = _relu(z1)
            q = _matvec(h1, self.w2, self.b2)
            z1_batch.append(z1)
            h1_batch.append(h1)
            q_batch.append(q)

        return q_batch, AgentCache(x=[list(x) for x in obs_batch], z1=z1_batch, h1=h1_batch)

    def backward(self, cache: AgentCache, grad_q: Sequence[Sequence[float]], lr: float) -> None:
        batch_size = max(1, len(cache.x))

        grad_w2 = [[0.0 for _ in range(self.action_dim)] for _ in range(self.hidden_dim)]
        grad_b2 = [0.0 for _ in range(self.action_dim)]
        grad_h1_batch = [[0.0 for _ in range(self.hidden_dim)] for _ in range(batch_size)]

        for i in range(batch_size):
            for h in range(self.hidden_dim):
                for a in range(self.action_dim):
                    grad_w2[h][a] += cache.h1[i][h] * grad_q[i][a]
            for a in range(self.action_dim):
                grad_b2[a] += grad_q[i][a]
            for h in range(self.hidden_dim):
                grad_h1_batch[i][h] = sum(grad_q[i][a] * self.w2[h][a] for a in range(self.action_dim))

        grad_z1_batch = [[0.0 for _ in range(self.hidden_dim)] for _ in range(batch_size)]
        for i in range(batch_size):
            relu_g = _relu_grad(cache.z1[i])
            for h in range(self.hidden_dim):
                grad_z1_batch[i][h] = grad_h1_batch[i][h] * relu_g[h]

        grad_w1 = [[0.0 for _ in range(self.hidden_dim)] for _ in range(self.obs_dim)]
        grad_b1 = [0.0 for _ in range(self.hidden_dim)]
        for i in range(batch_size):
            for o in range(self.obs_dim):
                for h in range(self.hidden_dim):
                    grad_w1[o][h] += cache.x[i][o] * grad_z1_batch[i][h]
            for h in range(self.hidden_dim):
                grad_b1[h] += grad_z1_batch[i][h]

        scale = lr / batch_size
        for o in range(self.obs_dim):
            for h in range(self.hidden_dim):
                self.w1[o][h] -= scale * grad_w1[o][h]
        for h in range(self.hidden_dim):
            self.b1[h] -= scale * grad_b1[h]
        for h in range(self.hidden_dim):
            for a in range(self.action_dim):
                self.w2[h][a] -= scale * grad_w2[h][a]
        for a in range(self.action_dim):
            self.b2[a] -= scale * grad_b2[a]


@dataclass
class MixerSampleCache:
    state: List[float]
    agent_q: List[float]
    w1_raw: List[float]
    w1: List[List[float]]
    b1: List[float]
    hidden_pre: List[float]
    hidden: List[float]
    w2_raw: List[float]
    w2: List[float]


class MixingNetwork:
    """QMIX monotonic mixer with state-conditioned hypernetworks."""

    def __init__(self, num_agents: int, state_dim: int, embed_dim: int = 16, seed: int | None = None) -> None:
        self.num_agents = num_agents
        self.state_dim = state_dim
        self.embed_dim = embed_dim
        self.rng = random.Random(seed)

        self.hw1_w = _rand_matrix(state_dim, num_agents * embed_dim, rng=self.rng)
        self.hw1_b = _rand_vector(num_agents * embed_dim, rng=self.rng)

        self.hb1_w = _rand_matrix(state_dim, embed_dim, rng=self.rng)
        self.hb1_b = _rand_vector(embed_dim, rng=self.rng)

        self.hw2_w = _rand_matrix(state_dim, embed_dim, rng=self.rng)
        self.hw2_b = _rand_vector(embed_dim, rng=self.rng)

        self.v_w = _rand_matrix(state_dim, 1, rng=self.rng)
        self.v_b = _rand_vector(1, rng=self.rng)

    def _hyper(self, state: Sequence[float], w: Sequence[Sequence[float]], b: Sequence[float]) -> List[float]:
        return _matvec(state, w, b)

    def forward(
        self,
        agent_q_batch: Sequence[Sequence[float]],
        state_batch: Sequence[Sequence[float]],
    ) -> Tuple[List[float], List[MixerSampleCache]]:
        if len(agent_q_batch) != len(state_batch):
            raise ValueError("agent_q_batch and state_batch must have same batch size")

        q_totals: List[float] = []
        caches: List[MixerSampleCache] = []

        for agent_q, state in zip(agent_q_batch, state_batch):
            w1_raw = self._hyper(state, self.hw1_w, self.hw1_b)
            w1 = []
            for i in range(self.num_agents):
                row: List[float] = []
                for j in range(self.embed_dim):
                    row.append(abs(w1_raw[i * self.embed_dim + j]))
                w1.append(row)

            b1 = self._hyper(state, self.hb1_w, self.hb1_b)
            hidden_pre = [
                sum(agent_q[i] * w1[i][j] for i in range(self.num_agents)) + b1[j]
                for j in range(self.embed_dim)
            ]
            hidden = [math.tanh(v) for v in hidden_pre]

            w2_raw = self._hyper(state, self.hw2_w, self.hw2_b)
            w2 = [abs(v) for v in w2_raw]

            v = self._hyper(state, self.v_w, self.v_b)[0]
            q_total = sum(hidden[j] * w2[j] for j in range(self.embed_dim)) + v

            q_totals.append(q_total)
            caches.append(
                MixerSampleCache(
                    state=list(state),
                    agent_q=list(agent_q),
                    w1_raw=w1_raw,
                    w1=w1,
                    b1=b1,
                    hidden_pre=hidden_pre,
                    hidden=hidden,
                    w2_raw=w2_raw,
                    w2=w2,
                )
            )

        return q_totals, caches

    def backward(
        self,
        caches: Sequence[MixerSampleCache],
        grad_q_total: Sequence[float],
        lr: float,
    ) -> List[List[float]]:
        batch_size = max(1, len(caches))
        grad_agent_q = [[0.0 for _ in range(self.num_agents)] for _ in range(batch_size)]

        grad_hw1_w = [[0.0 for _ in range(self.num_agents * self.embed_dim)] for _ in range(self.state_dim)]
        grad_hw1_b = [0.0 for _ in range(self.num_agents * self.embed_dim)]

        grad_hb1_w = [[0.0 for _ in range(self.embed_dim)] for _ in range(self.state_dim)]
        grad_hb1_b = [0.0 for _ in range(self.embed_dim)]

        grad_hw2_w = [[0.0 for _ in range(self.embed_dim)] for _ in range(self.state_dim)]
        grad_hw2_b = [0.0 for _ in range(self.embed_dim)]

        grad_v_w = [[0.0] for _ in range(self.state_dim)]
        grad_v_b = [0.0]

        for idx, cache in enumerate(caches):
            grad_out = grad_q_total[idx]

            # q_total = sum(hidden[j]*w2[j]) + v
            grad_hidden = [grad_out * cache.w2[j] for j in range(self.embed_dim)]
            grad_w2 = [grad_out * cache.hidden[j] for j in range(self.embed_dim)]

            # w2 = abs(w2_raw)
            grad_w2_raw = [
                grad_w2[j] * (1.0 if cache.w2_raw[j] >= 0 else -1.0) for j in range(self.embed_dim)
            ]

            for s in range(self.state_dim):
                for j in range(self.embed_dim):
                    grad_hw2_w[s][j] += cache.state[s] * grad_w2_raw[j]
            for j in range(self.embed_dim):
                grad_hw2_b[j] += grad_w2_raw[j]

            # hidden = tanh(hidden_pre)
            grad_hidden_pre = [
                grad_hidden[j] * (1.0 - cache.hidden[j] * cache.hidden[j]) for j in range(self.embed_dim)
            ]

            # hidden_pre = sum_i agent_q[i] * w1[i][j] + b1[j]
            for i in range(self.num_agents):
                grad_agent_q[idx][i] = sum(grad_hidden_pre[j] * cache.w1[i][j] for j in range(self.embed_dim))

            grad_w1 = [[0.0 for _ in range(self.embed_dim)] for _ in range(self.num_agents)]
            for i in range(self.num_agents):
                for j in range(self.embed_dim):
                    grad_w1[i][j] = grad_hidden_pre[j] * cache.agent_q[i]

            grad_w1_raw = [0.0 for _ in range(self.num_agents * self.embed_dim)]
            for i in range(self.num_agents):
                for j in range(self.embed_dim):
                    flat = i * self.embed_dim + j
                    sign = 1.0 if cache.w1_raw[flat] >= 0 else -1.0
                    grad_w1_raw[flat] = grad_w1[i][j] * sign

            for s in range(self.state_dim):
                for flat in range(self.num_agents * self.embed_dim):
                    grad_hw1_w[s][flat] += cache.state[s] * grad_w1_raw[flat]
            for flat in range(self.num_agents * self.embed_dim):
                grad_hw1_b[flat] += grad_w1_raw[flat]

            for s in range(self.state_dim):
                for j in range(self.embed_dim):
                    grad_hb1_w[s][j] += cache.state[s] * grad_hidden_pre[j]
            for j in range(self.embed_dim):
                grad_hb1_b[j] += grad_hidden_pre[j]

            # v(state) contribution
            for s in range(self.state_dim):
                grad_v_w[s][0] += cache.state[s] * grad_out
            grad_v_b[0] += grad_out

        scale = lr / batch_size

        for s in range(self.state_dim):
            for f in range(self.num_agents * self.embed_dim):
                self.hw1_w[s][f] -= scale * grad_hw1_w[s][f]
        for f in range(self.num_agents * self.embed_dim):
            self.hw1_b[f] -= scale * grad_hw1_b[f]

        for s in range(self.state_dim):
            for j in range(self.embed_dim):
                self.hb1_w[s][j] -= scale * grad_hb1_w[s][j]
                self.hw2_w[s][j] -= scale * grad_hw2_w[s][j]
        for j in range(self.embed_dim):
            self.hb1_b[j] -= scale * grad_hb1_b[j]
            self.hw2_b[j] -= scale * grad_hw2_b[j]

        for s in range(self.state_dim):
            self.v_w[s][0] -= scale * grad_v_w[s][0]
        self.v_b[0] -= scale * grad_v_b[0]

        return grad_agent_q


class QMIXLearner:
    """Centralized-training/decentralized-execution QMIX learner."""

    def __init__(
        self,
        num_agents: int,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        hidden_dim: int = 32,
        mixer_embed_dim: int = 16,
        gamma: float = 0.99,
        lr: float = 1e-3,
        tau: float = 0.01,
        seed: int | None = None,
    ) -> None:
        self.num_agents = num_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.gamma = gamma
        self.lr = lr
        self.tau = tau
        self.rng = random.Random(seed)

        self.agent_nets = [
            AgentQNetwork(obs_dim=obs_dim, action_dim=action_dim, hidden_dim=hidden_dim, seed=self.rng.randint(0, 10**9))
            for _ in range(num_agents)
        ]
        self.target_agent_nets = [copy.deepcopy(net) for net in self.agent_nets]

        self.mixer = MixingNetwork(
            num_agents=num_agents,
            state_dim=state_dim,
            embed_dim=mixer_embed_dim,
            seed=self.rng.randint(0, 10**9),
        )
        self.target_mixer = copy.deepcopy(self.mixer)

    def _select_q_by_action(self, q_batch: Sequence[Sequence[float]], actions: Sequence[int]) -> List[float]:
        values: List[float] = []
        for i, q_values in enumerate(q_batch):
            action = actions[i]
            if action < 0 or action >= len(q_values):
                action = 0
            values.append(q_values[action])
        return values

    def _max_q(self, q_batch: Sequence[Sequence[float]]) -> List[float]:
        return [max(row) if row else 0.0 for row in q_batch]

    def act(self, local_obs: Sequence[Sequence[float]], epsilon: float = 0.0) -> List[int]:
        actions: List[int] = []
        for agent_id, obs in enumerate(local_obs):
            if self.rng.random() < epsilon:
                actions.append(self.rng.randrange(self.action_dim))
                continue
            q_values, _ = self.agent_nets[agent_id].forward([obs])
            greedy = max(range(self.action_dim), key=lambda a: q_values[0][a])
            actions.append(greedy)
        return actions

    def train_step(self, batch: Sequence[Dict[str, object]]) -> Dict[str, float]:
        batch_size = len(batch)
        if batch_size == 0:
            return {"loss": 0.0}

        obs = [item["obs"] for item in batch]
        actions = [item["actions"] for item in batch]
        rewards = [float(item["reward"]) for item in batch]
        next_obs = [item["next_obs"] for item in batch]
        states = [item["state"] for item in batch]
        next_states = [item["next_state"] for item in batch]
        dones = [1.0 if bool(item["done"]) else 0.0 for item in batch]

        chosen_agent_q: List[List[float]] = [[0.0 for _ in range(self.num_agents)] for _ in range(batch_size)]
        next_max_agent_q: List[List[float]] = [[0.0 for _ in range(self.num_agents)] for _ in range(batch_size)]

        online_cache_by_agent: List[AgentCache] = []
        for agent_id in range(self.num_agents):
            obs_batch = [obs[b][agent_id] for b in range(batch_size)]
            q_batch, cache = self.agent_nets[agent_id].forward(obs_batch)
            online_cache_by_agent.append(cache)
            selected = self._select_q_by_action(q_batch, [actions[b][agent_id] for b in range(batch_size)])
            for b in range(batch_size):
                chosen_agent_q[b][agent_id] = selected[b]

            next_obs_batch = [next_obs[b][agent_id] for b in range(batch_size)]
            tgt_q_batch, _ = self.target_agent_nets[agent_id].forward(next_obs_batch)
            maxima = self._max_q(tgt_q_batch)
            for b in range(batch_size):
                next_max_agent_q[b][agent_id] = maxima[b]

        q_tot, mixer_cache = self.mixer.forward(chosen_agent_q, states)
        next_q_tot, _ = self.target_mixer.forward(next_max_agent_q, next_states)

        targets = [
            rewards[b] + self.gamma * (1.0 - dones[b]) * next_q_tot[b]
            for b in range(batch_size)
        ]

        td_errors = [q_tot[b] - targets[b] for b in range(batch_size)]
        loss = sum(err * err for err in td_errors) / batch_size
        grad_q_total = [2.0 * err / batch_size for err in td_errors]

        grad_agent_q = self.mixer.backward(mixer_cache, grad_q_total, lr=self.lr)

        for agent_id in range(self.num_agents):
            grad_q = [[0.0 for _ in range(self.action_dim)] for _ in range(batch_size)]
            for b in range(batch_size):
                action = actions[b][agent_id]
                if 0 <= action < self.action_dim:
                    grad_q[b][action] = grad_agent_q[b][agent_id]
            self.agent_nets[agent_id].backward(online_cache_by_agent[agent_id], grad_q, lr=self.lr)

        self._soft_update_targets()

        return {"loss": loss, "mean_q_tot": sum(q_tot) / batch_size, "mean_target": sum(targets) / batch_size}

    def _soft_update_matrix(self, src: List[List[float]], dst: List[List[float]]) -> None:
        for i in range(len(src)):
            for j in range(len(src[i])):
                dst[i][j] = self.tau * src[i][j] + (1.0 - self.tau) * dst[i][j]

    def _soft_update_vector(self, src: List[float], dst: List[float]) -> None:
        for i in range(len(src)):
            dst[i] = self.tau * src[i] + (1.0 - self.tau) * dst[i]

    def _soft_update_targets(self) -> None:
        for src, dst in zip(self.agent_nets, self.target_agent_nets):
            self._soft_update_matrix(src.w1, dst.w1)
            self._soft_update_vector(src.b1, dst.b1)
            self._soft_update_matrix(src.w2, dst.w2)
            self._soft_update_vector(src.b2, dst.b2)

        self._soft_update_matrix(self.mixer.hw1_w, self.target_mixer.hw1_w)
        self._soft_update_vector(self.mixer.hw1_b, self.target_mixer.hw1_b)
        self._soft_update_matrix(self.mixer.hb1_w, self.target_mixer.hb1_w)
        self._soft_update_vector(self.mixer.hb1_b, self.target_mixer.hb1_b)
        self._soft_update_matrix(self.mixer.hw2_w, self.target_mixer.hw2_w)
        self._soft_update_vector(self.mixer.hw2_b, self.target_mixer.hw2_b)
        self._soft_update_matrix(self.mixer.v_w, self.target_mixer.v_w)
        self._soft_update_vector(self.mixer.v_b, self.target_mixer.v_b)
