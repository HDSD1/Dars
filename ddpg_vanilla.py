"""
原始 DDPG 智能体 (Vanilla — 无优化技巧)
=========================================
用作论文的 Baseline 对照组.

与 ddpg_agent.py 的区别 (去掉了以下优化):
  1. 去掉 LayerNorm          → 训练更不稳定
  2. 去掉 Xavier 初始化       → 用 PyTorch 默认随机初始化
  3. 去掉梯度裁剪 clip_grad  → 梯度可能爆炸
  4. 网络更小 (256→128)       → 容量更低

保留不变:
  - 均匀随机经验回放
  - 单 Critic (无 TD3)
  - OU 噪声
  - Soft update
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    STATE_DIM, ACTION_DIM, ACTION_BOUND,
    LR_ACTOR, LR_CRITIC, GAMMA, TAU_SOFT,
    MEMORY_CAPACITY, BATCH_SIZE,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ==================================================================
# 原始 Actor (无 LayerNorm, 无 Xavier, 更小网络)
# ==================================================================
class VanillaActor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, a_dim),
            nn.Tanh(),
        )
        # 不做任何特殊初始化, 用 PyTorch 默认的 Kaiming uniform

    def forward(self, state):
        return self.net(state) * ACTION_BOUND[1]


# ==================================================================
# 原始 Critic (无 LayerNorm, 无 Xavier, 更小网络)
# ==================================================================
class VanillaCritic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(s_dim + a_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        return self.net(x)


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
# 经验回放 (同标准版)
# ==================================================================
class ReplayBuffer:
    def __init__(self, capacity, s_dim, a_dim):
        self.capacity = capacity
        self.pointer = 0
        self.size = 0
        self.states = np.zeros((capacity, s_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, a_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, s_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def store(self, s, a, r, s_, done):
        idx = self.pointer % self.capacity
        self.states[idx] = s
        self.actions[idx] = a
        self.rewards[idx] = r
        self.next_states[idx] = s_
        self.dones[idx] = float(done)
        self.pointer += 1
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, size=batch_size, replace=False)
        return (
            torch.FloatTensor(self.states[indices]).to(DEVICE),
            torch.FloatTensor(self.actions[indices]).to(DEVICE),
            torch.FloatTensor(self.rewards[indices]).to(DEVICE),
            torch.FloatTensor(self.next_states[indices]).to(DEVICE),
            torch.FloatTensor(self.dones[indices]).to(DEVICE),
        )

    def ready(self, batch_size):
        return self.size >= batch_size


# ==================================================================
# Vanilla DDPG Agent (原始版, 无任何优化技巧)
# ==================================================================
class VanillaDDPGAgent:
    """
    原始 DDPG — 作为论文 Baseline.

    与改进版 (ddpg_agent.py / per_ddpg_agent.py) 的差异:
      - 无 LayerNorm
      - 无 Xavier 初始化
      - 无梯度裁剪
      - 更小网络 (256-128 vs 400-300)
      - 单 Critic (无 TD3)
      - 均匀采样 (无 PER)
    """

    def __init__(self, s_dim=STATE_DIM, a_dim=ACTION_DIM):
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.actor = VanillaActor(s_dim, a_dim).to(DEVICE)
        self.critic = VanillaCritic(s_dim, a_dim).to(DEVICE)
        self.actor_target = VanillaActor(s_dim, a_dim).to(DEVICE)
        self.critic_target = VanillaCritic(s_dim, a_dim).to(DEVICE)
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.buffer = ReplayBuffer(MEMORY_CAPACITY, s_dim, a_dim)
        self.ou_noise = OUNoise(a_dim, sigma=0.2)

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
            return

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(BATCH_SIZE)

        # Critic
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            y = rewards + GAMMA * (1 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # ★ 无梯度裁剪
        self.critic_optimizer.step()

        # Actor
        pred_actions = self.actor(states)
        actor_loss = -self.critic(states, pred_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # ★ 无梯度裁剪
        self.actor_optimizer.step()

        # Soft update
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

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
