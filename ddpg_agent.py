"""
PyTorch DDPG 智能体 (V2 — 改进版)
===================================
改进点:
  1. 增大网络容量 (400→300)
  2. 加入 LayerNorm 稳定训练
  3. 运行时 reward normalization
  4. Ornstein-Uhlenbeck 噪声 (可选)
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
# 网络定义 (加大 + LayerNorm)
# ==================================================================
class Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fc1 = nn.Linear(s_dim, 400)
        self.ln1 = nn.LayerNorm(400)
        self.fc2 = nn.Linear(400, 300)
        self.ln2 = nn.LayerNorm(300)
        self.fc3 = nn.Linear(300, a_dim)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        # 输出层小权重初始化
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, state):
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        return torch.tanh(self.fc3(x)) * ACTION_BOUND[1]


class Critic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fc1 = nn.Linear(s_dim + a_dim, 400)
        self.ln1 = nn.LayerNorm(400)
        self.fc2 = nn.Linear(400, 300)
        self.ln2 = nn.LayerNorm(300)
        self.fc3 = nn.Linear(300, 1)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


# ==================================================================
# Ornstein-Uhlenbeck 噪声
# ==================================================================
class OUNoise:
    """Ornstein-Uhlenbeck 过程噪声 — 比高斯噪声有更好的探索连续性."""
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
# Reward Normalizer
# ==================================================================
class RunningNormalizer:
    """运行时奖励归一化 (Welford 算法)."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, x):
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    @property
    def std(self):
        if self.n < 2:
            return 1.0
        return max(np.sqrt(self.M2 / (self.n - 1)), 1e-6)

    def normalize(self, x):
        self.update(x)
        return (x - self.mean) / self.std


# ==================================================================
# 经验回放缓冲区
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
# DDPG 智能体
# ==================================================================
class DDPGAgent:
    def __init__(self, s_dim=STATE_DIM, a_dim=ACTION_DIM):
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.actor = Actor(s_dim, a_dim).to(DEVICE)
        self.critic = Critic(s_dim, a_dim).to(DEVICE)
        self.actor_target = Actor(s_dim, a_dim).to(DEVICE)
        self.critic_target = Critic(s_dim, a_dim).to(DEVICE)
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.buffer = ReplayBuffer(MEMORY_CAPACITY, s_dim, a_dim)

        # OU 噪声
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
        # 直接存原始 reward (去掉了 normalizer, 避免过早收敛)
        self.buffer.store(s, a, r, s_, done)

    def learn(self):
        if not self.buffer.ready(BATCH_SIZE):
            return

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(BATCH_SIZE)

        # --- 更新 Critic ---
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            y = rewards + GAMMA * (1 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.MSELoss()(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # --- 更新 Actor ---
        pred_actions = self.actor(states)
        actor_loss = -self.critic(states, pred_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # --- 软更新 ---
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
