"""
改进 DDPG 智能体 (PER + TD3)
==============================
在标准 DDPG 基础上叠加两项改进:

改进 1: 优先经验回放 (PER)
  - SumTree O(log n) 按 TD-error 优先采样
  - 重要性采样 (IS) 权重修正偏差
  - 参考: Schaul et al., ICLR 2016

改进 2: 双 Critic + 延迟更新 (TD3)
  - 双 Q 网络取 min 防止 Q 值过估计
  - 目标策略平滑: target action 加噪声
  - Actor 延迟更新: 每 2 步更新 1 次
  - 参考: Fujimoto et al., ICML 2018
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    STATE_DIM, ACTION_DIM, ACTION_BOUND,
    LR_ACTOR, LR_CRITIC, GAMMA, TAU_SOFT,
    MEMORY_CAPACITY, BATCH_SIZE,
    HIDDEN_DIM_1, HIDDEN_DIM_2, GRAD_CLIP,
    TD3_POLICY_NOISE, TD3_NOISE_CLIP, TD3_POLICY_DELAY,
    PER_ALPHA, PER_BETA_START, PER_BETA_FRAMES,
    PER_EPSILON, PER_MAX_IS_WEIGHT,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================================================================
# Actor (与标准 DDPG 相同结构)
# ==================================================================
class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fc1 = nn.Linear(s_dim, HIDDEN_DIM_1)
        self.ln1 = nn.LayerNorm(HIDDEN_DIM_1)
        self.fc2 = nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2)
        self.ln2 = nn.LayerNorm(HIDDEN_DIM_2)
        self.fc3 = nn.Linear(HIDDEN_DIM_2, a_dim)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, state):
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        return torch.tanh(self.fc3(x)) * ACTION_BOUND[1]


# ==================================================================
# 双 Critic (TD3: 两个独立 Q 网络)
# ==================================================================
class TwinCritic(nn.Module):
    """
    TD3 核心: 维护两个独立的 Q 网络, 取 min(Q1, Q2)
    防止 Actor 利用 Q 的过估计来"作弊".
    """
    def __init__(self, s_dim, a_dim):
        super().__init__()
        # Q1
        self.fc1_1 = nn.Linear(s_dim + a_dim, HIDDEN_DIM_1)
        self.ln1_1 = nn.LayerNorm(HIDDEN_DIM_1)
        self.fc1_2 = nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2)
        self.ln1_2 = nn.LayerNorm(HIDDEN_DIM_2)
        self.fc1_3 = nn.Linear(HIDDEN_DIM_2, 1)
        # Q2
        self.fc2_1 = nn.Linear(s_dim + a_dim, HIDDEN_DIM_1)
        self.ln2_1 = nn.LayerNorm(HIDDEN_DIM_1)
        self.fc2_2 = nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2)
        self.ln2_2 = nn.LayerNorm(HIDDEN_DIM_2)
        self.fc2_3 = nn.Linear(HIDDEN_DIM_2, 1)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1_1, self.fc1_2, self.fc2_1, self.fc2_2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in [self.fc1_3, self.fc2_3]:
            nn.init.uniform_(layer.weight, -3e-3, 3e-3)
            nn.init.zeros_(layer.bias)

    def forward(self, state, action):
        """返回两个 Q 值."""
        x = torch.cat([state, action], dim=1)
        q1 = torch.relu(self.ln1_1(self.fc1_1(x)))
        q1 = torch.relu(self.ln1_2(self.fc1_2(q1)))
        q1 = self.fc1_3(q1)
        q2 = torch.relu(self.ln2_1(self.fc2_1(x)))
        q2 = torch.relu(self.ln2_2(self.fc2_2(q2)))
        q2 = self.fc2_3(q2)
        return q1, q2

    def q1_only(self, state, action):
        """Actor 更新时只用 Q1 (避免重复计算)."""
        x = torch.cat([state, action], dim=1)
        q1 = torch.relu(self.ln1_1(self.fc1_1(x)))
        q1 = torch.relu(self.ln1_2(self.fc1_2(q1)))
        return self.fc1_3(q1)


# ==================================================================
# SumTree — PER 的核心数据结构
# ==================================================================
class SumTree:
    """
    完全二叉树: 叶节点存优先级, 内部节点存子树之和.
    支持 O(log n) 的按权重采样和更新.
    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0

    def update(self, tree_idx, priority):
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, value):
        parent = 0
        while True:
            left = 2 * parent + 1
            right = left + 1
            if left >= len(self.tree):
                leaf_idx = parent
                break
            if value <= self.tree[left]:
                parent = left
            else:
                value -= self.tree[left]
                parent = right
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], data_idx

    @property
    def total_priority(self):
        return self.tree[0]

    @property
    def max_priority(self):
        return np.max(self.tree[self.capacity - 1:
                                self.capacity - 1 + self.capacity])


# ==================================================================
# 优先经验回放缓冲区 (PER)
# ==================================================================
class PrioritizedReplayBuffer:
    """
    核心创新: 用 TD-error 作为优先级, 重要经验被采到的概率更高.
    P(i) = p_i^α / Σ p_j^α
    w_i = (N * P(i))^{-β}  (重要性采样修正)
    """
    def __init__(self, capacity, s_dim, a_dim):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.size = 0

        self.states = np.zeros((capacity, s_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, a_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, s_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self._frame = 0

    def store(self, s, a, r, s_, done):
        idx = self.tree.data_pointer
        self.states[idx] = s
        self.actions[idx] = a
        self.rewards[idx] = r
        self.next_states[idx] = s_
        self.dones[idx] = float(done)

        # 新经验赋予最大优先级 (保证至少被采样一次)
        max_p = self.tree.max_priority
        if max_p <= 0:
            max_p = 1.0
        tree_idx = self.tree.data_pointer + self.capacity - 1
        self.tree.update(tree_idx, max_p)

        self.tree.data_pointer = (self.tree.data_pointer + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        self._frame += 1
        beta = min(1.0,
                   PER_BETA_START + self._frame *
                   (1.0 - PER_BETA_START) / PER_BETA_FRAMES)

        indices, tree_indices, priorities = [], [], []
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            value = np.random.uniform(segment * i, segment * (i + 1))
            tree_idx, priority, data_idx = self.tree.get_leaf(value)
            data_idx = np.clip(data_idx, 0, self.size - 1)
            indices.append(data_idx)
            tree_indices.append(tree_idx)
            priorities.append(priority)

        priorities = np.array(priorities, dtype=np.float32) + PER_EPSILON
        sampling_probs = priorities / (self.tree.total_priority + PER_EPSILON)
        is_weights = np.power(self.size * sampling_probs, -beta)
        is_weights /= is_weights.max()
        is_weights = np.clip(is_weights, 0.0, PER_MAX_IS_WEIGHT)

        indices = np.array(indices)
        return (
            torch.FloatTensor(self.states[indices]).to(DEVICE),
            torch.FloatTensor(self.actions[indices]).to(DEVICE),
            torch.FloatTensor(self.rewards[indices]).to(DEVICE),
            torch.FloatTensor(self.next_states[indices]).to(DEVICE),
            torch.FloatTensor(self.dones[indices]).to(DEVICE),
            torch.FloatTensor(is_weights).unsqueeze(1).to(DEVICE),
            tree_indices,
        )

    def update_priorities(self, tree_indices, td_errors):
        """每次 learn 后用新的 TD-error 更新优先级."""
        for tree_idx, td_err in zip(tree_indices, td_errors):
            priority = (abs(td_err) + PER_EPSILON) ** PER_ALPHA
            self.tree.update(tree_idx, priority)

    def ready(self, batch_size):
        return self.size >= batch_size


# ==================================================================
# OU Noise (同标准版)
# ==================================================================
class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.size) * self.mu

    def reset(self):
        self.state = np.ones(self.size) * self.mu

    def sample(self):
        dx = self.theta * (self.mu - self.state)
        dx += self.sigma * np.random.randn(self.size)
        self.state += dx
        return self.state


# ==================================================================
# PER-DDPG Agent (V2: PER + TD3)
# ==================================================================
class PERDDPGAgent:
    """
    改进的 DDPG 智能体.

    与标准 DDPG 的区别:
      1. PER Buffer → 优先学习重要经验
      2. Twin Critic → min(Q1,Q2) 防止 Q 过估计
      3. Target Policy Smoothing → target action 加噪声
      4. Delayed Actor Update → Actor 每 2 步更新 1 次
    """

    POLICY_NOISE = TD3_POLICY_NOISE
    NOISE_CLIP = TD3_NOISE_CLIP
    POLICY_DELAY = TD3_POLICY_DELAY

    def __init__(self, s_dim=STATE_DIM, a_dim=ACTION_DIM):
        self.s_dim = s_dim
        self.a_dim = a_dim

        # Actor (同标准 DDPG)
        self.actor = Actor(s_dim, a_dim).to(DEVICE)
        self.actor_target = Actor(s_dim, a_dim).to(DEVICE)

        # ★ 改进: 双 Critic (TD3)
        self.critic = TwinCritic(s_dim, a_dim).to(DEVICE)
        self.critic_target = TwinCritic(s_dim, a_dim).to(DEVICE)

        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # ★ 改进: PER Buffer (替代均匀采样)
        self.buffer = PrioritizedReplayBuffer(MEMORY_CAPACITY, s_dim, a_dim)
        self.ou_noise = OUNoise(a_dim, sigma=0.2)
        self._learn_step = 0

    def select_action(self, state, noise_std=0.0, use_ou=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        if use_ou:
            action = action + self.ou_noise.sample() * noise_std
        elif noise_std > 0:
            action = action + np.random.normal(0, noise_std, size=self.a_dim)
        return np.clip(action, ACTION_BOUND[0], ACTION_BOUND[1])

    def store(self, s, a, r, s_, done):
        self.buffer.store(s, a, r, s_, done)

    def learn(self):
        if not self.buffer.ready(BATCH_SIZE):
            return None
        self._learn_step += 1

        states, actions, rewards, next_states, dones, \
            is_weights, tree_indices = self.buffer.sample(BATCH_SIZE)

        # ---- Critic 更新 ----
        with torch.no_grad():
            # ★ TD3: 目标策略平滑 (给 target action 加噪声)
            noise = (torch.randn_like(actions) * self.POLICY_NOISE
                     ).clamp(-self.NOISE_CLIP, self.NOISE_CLIP)
            next_actions = (self.actor_target(next_states) + noise
                           ).clamp(ACTION_BOUND[0], ACTION_BOUND[1])

            # ★ TD3: 双 Q 取 min (防止过估计)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            y = rewards + GAMMA * (1 - dones) * target_q

        current_q1, current_q2 = self.critic(states, actions)
        td_errors = (y - current_q1).detach()

        # ★ PER: IS 加权的 Critic Loss
        critic_loss = (is_weights * ((current_q1 - y) ** 2 +
                                     (current_q2 - y) ** 2)).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=GRAD_CLIP)
        self.critic_optimizer.step()

        # ---- Actor 更新 ----
        # ★ TD3: 延迟更新 (每 2 步更新 1 次, 让 Critic 先稳定)
        if self._learn_step % self.POLICY_DELAY == 0:
            pred_actions = self.actor(states)
            actor_loss = -self.critic.q1_only(states, pred_actions).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=GRAD_CLIP)
            self.actor_optimizer.step()

            self._soft_update(self.actor_target, self.actor)
            self._soft_update(self.critic_target, self.critic)

        # ★ PER: 用新 TD-error 更新优先级
        self.buffer.update_priorities(
            tree_indices,
            td_errors.squeeze(1).cpu().numpy()
        )

        # 返回诊断指标
        return {
            'critic_loss': critic_loss.item(),
            'avg_q': current_q1.mean().item(),
            'avg_td_error': td_errors.abs().mean().item(),
        }

    @staticmethod
    def _soft_update(target, source, tau=TAU_SOFT):
        for tp, sp in zip(target.parameters(), source.parameters()):
            tp.data.copy_(tau * sp.data + (1 - tau) * tp.data)

    @staticmethod
    def _hard_update(target, source):
        target.load_state_dict(source.state_dict())

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
