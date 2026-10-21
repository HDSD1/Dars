"""
TD3+CVX Baseline (Twin Delayed DDPG)
======================================
确定性策略 + 三项改进:
  ① Twin Critic: 双 Q 网络, 取 min 防过估计
  ② Delayed Policy Update: Actor 延迟更新 (每 2 步)
  ③ Target Policy Smoothing: 目标动作加噪声防过拟合

与本文 PER-SAC+CVX 的差异:
  - 策略类型: TD3 确定性策略 vs SAC 随机策略 (最大熵)
  - 经验回放: TD3 均匀采样 vs PER Dual-Priority 优先采样
  - 探索方式: TD3 高斯噪声 vs SAC 策略熵

参考: Fujimoto et al., ICML 2018

网络结构与 PER-SAC+CVX 一致 (400-300, LayerNorm, Xavier),
确保对比公平, 仅算法层面不同.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from config import (
    STATE_DIM, ACTION_DIM, ACTION_BOUND,
    LR_ACTOR, LR_CRITIC, GAMMA, TAU_SOFT,
    MEMORY_CAPACITY, BATCH_SIZE,
    HIDDEN_DIM_1, HIDDEN_DIM_2, GRAD_CLIP,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TD3 超参数
TD3_POLICY_NOISE = 0.2      # 目标策略平滑噪声 σ
TD3_NOISE_CLIP = 0.5        # 噪声裁剪范围
TD3_POLICY_DELAY = 2        # Actor 延迟更新频率


# ==================================================================
# TD3 Actor (确定性策略, 与 PER-SAC 同架构)
# ==================================================================
class TD3Actor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fc1 = nn.Linear(s_dim, HIDDEN_DIM_1)
        self.ln1 = nn.LayerNorm(HIDDEN_DIM_1)
        self.fc2 = nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2)
        self.ln2 = nn.LayerNorm(HIDDEN_DIM_2)
        self.out = nn.Linear(HIDDEN_DIM_2, a_dim)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.out.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.out.bias)

    def forward(self, state):
        x = torch.relu(self.ln1(self.fc1(state)))
        x = torch.relu(self.ln2(self.fc2(x)))
        return torch.tanh(self.out(x)) * ACTION_BOUND[1]


# ==================================================================
# TD3 Twin Critic (与 PER-SAC 同架构)
# ==================================================================
class TD3Critic(nn.Module):
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
        x = torch.cat([state, action], dim=1)
        q1 = torch.relu(self.ln1_1(self.fc1_1(x)))
        q1 = torch.relu(self.ln1_2(self.fc1_2(q1)))
        q1 = self.fc1_3(q1)
        q2 = torch.relu(self.ln2_1(self.fc2_1(x)))
        q2 = torch.relu(self.ln2_2(self.fc2_2(q2)))
        q2 = self.fc2_3(q2)
        return q1, q2

    def q1_only(self, state, action):
        """只计算 Q1 (Actor 更新时用)."""
        x = torch.cat([state, action], dim=1)
        q1 = torch.relu(self.ln1_1(self.fc1_1(x)))
        q1 = torch.relu(self.ln1_2(self.fc1_2(q1)))
        return self.fc1_3(q1)


# ==================================================================
# Replay Buffer (均匀采样)
# ==================================================================
class ReplayBuffer:
    def __init__(self, capacity, s_dim, a_dim):
        self.capacity = capacity
        self.size = 0
        self.ptr = 0
        self.states = np.zeros((capacity, s_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, a_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, s_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def store(self, s, a, r, s_, done):
        self.states[self.ptr] = s
        self.actions[self.ptr] = a
        self.rewards[self.ptr] = r
        self.next_states[self.ptr] = s_
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size):
        idx = np.random.choice(self.size, batch_size, replace=False)
        return (
            torch.FloatTensor(self.states[idx]).to(DEVICE),
            torch.FloatTensor(self.actions[idx]).to(DEVICE),
            torch.FloatTensor(self.rewards[idx]).to(DEVICE),
            torch.FloatTensor(self.next_states[idx]).to(DEVICE),
            torch.FloatTensor(self.dones[idx]).to(DEVICE),
        )

    def ready(self, batch_size):
        return self.size >= batch_size


# ==================================================================
# TD3 Agent
# ==================================================================
class TD3Agent:
    """
    Twin Delayed DDPG (TD3).
    与 PER-SAC+CVX 同架构, 区别: 确定性策略 + 均匀采样.
    """
    def __init__(self, s_dim=STATE_DIM, a_dim=ACTION_DIM):
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.actor = TD3Actor(s_dim, a_dim).to(DEVICE)
        self.actor_target = TD3Actor(s_dim, a_dim).to(DEVICE)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TD3Critic(s_dim, a_dim).to(DEVICE)
        self.critic_target = TD3Critic(s_dim, a_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.buffer = ReplayBuffer(MEMORY_CAPACITY, s_dim, a_dim)
        self.learn_step = 0

        # OU noise placeholder for compatibility
        self.ou_noise = type('', (), {'reset': lambda s: None})()

    def select_action(self, state, noise_std=0.1, **kw):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            action = self.actor(state_tensor).cpu().numpy()[0]
        # 探索噪声
        noise = np.random.normal(0, noise_std, size=self.a_dim)
        action = np.clip(action + noise, ACTION_BOUND[0], ACTION_BOUND[1])
        return action

    def store(self, s, a, r, s_, done):
        self.buffer.store(s, a, r, s_, done)

    def learn(self):
        if not self.buffer.ready(BATCH_SIZE):
            return {}

        self.learn_step += 1

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(BATCH_SIZE)

        # ---- Critic 更新 ----
        with torch.no_grad():
            # ★ TD3 核心: Target Policy Smoothing
            noise = (torch.randn_like(actions) * TD3_POLICY_NOISE).clamp(
                -TD3_NOISE_CLIP, TD3_NOISE_CLIP)
            next_actions = (self.actor_target(next_states) + noise).clamp(
                ACTION_BOUND[0], ACTION_BOUND[1])

            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            y = rewards + GAMMA * (1 - dones) * target_q

        current_q1, current_q2 = self.critic(states, actions)

        # Huber Loss (与 PER-SAC 一致)
        critic_loss = (F.smooth_l1_loss(current_q1, y) +
                       F.smooth_l1_loss(current_q2, y))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=GRAD_CLIP)
        self.critic_optimizer.step()

        # ---- Actor 延迟更新 ----
        actor_loss_val = 0.0
        if self.learn_step % TD3_POLICY_DELAY == 0:
            actor_loss = -self.critic.q1_only(states, self.actor(states)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=GRAD_CLIP)
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            # 软更新 target networks
            for tp, sp in zip(self.actor_target.parameters(),
                              self.actor.parameters()):
                tp.data.copy_(TAU_SOFT * sp.data + (1 - TAU_SOFT) * tp.data)
            for tp, sp in zip(self.critic_target.parameters(),
                              self.critic.parameters()):
                tp.data.copy_(TAU_SOFT * sp.data + (1 - TAU_SOFT) * tp.data)

        # TD-error for diagnostics
        td_error1 = (current_q1 - y).detach().cpu().numpy().flatten()
        td_error2 = (current_q2 - y).detach().cpu().numpy().flatten()
        td_errors = (np.abs(td_error1) + np.abs(td_error2)) / 2.0

        return {
            'critic_loss': critic_loss.item(),
            'avg_q': (current_q1.mean().item() + current_q2.mean().item()) / 2,
            'avg_td_error': np.mean(td_errors),
        }

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(checkpoint['actor'])
        self.actor_target.load_state_dict(checkpoint['actor_target'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
