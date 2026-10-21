"""
Attention-SAC+CVX (消融: 有 Attention, 无 PER)
================================================
与 PER-SAC+CVX (per_sac_agent.py) 的唯一区别: 均匀采样 vs Dual-Priority PER.
保留 Attention 机制, 用于隔离 PER 的贡献.

共享配置:
  - Attention-Enhanced Actor/Critic (UserAttention + MHA)
  - N-step Return (3步)
  - Huber Loss, 延迟 Actor 更新
  - LayerNorm, Xavier, 梯度裁剪

移除的:
  - PER (改为均匀采样)
  - IS 权重 (均匀采样无偏)
  - Dual-Priority / Reward Surprise
"""
import numpy as np
from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from config import (
    K, M, STATE_DIM, ACTION_DIM, ACTION_BOUND,
    LR_ACTOR, LR_CRITIC, GAMMA, TAU_SOFT,
    MEMORY_CAPACITY, BATCH_SIZE,
    HIDDEN_DIM_1, HIDDEN_DIM_2, GRAD_CLIP,
)

N_STEP = 3
ACTOR_DELAY = 2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

# 复用 DARS 的 Attention 组件和状态分解
from DARS import (
    UserAttention, decompose_state,
    UAV_FEAT_DIM, ATTENTION_EMBED_DIM,
)


# ==================================================================
# N-Step Buffer
# ==================================================================
class NStepBuffer:
    def __init__(self, n_step=N_STEP, gamma=GAMMA):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)

    def store(self, transition):
        self.buffer.append(transition)
        if len(self.buffer) == self.n_step:
            return self._compute_nstep()
        return None

    def flush(self):
        results = []
        while len(self.buffer) > 0:
            results.append(self._compute_nstep_partial())
            self.buffer.popleft()
        return results

    def _compute_nstep(self):
        reward = sum(
            self.gamma ** i * self.buffer[i][2]
            for i in range(self.n_step)
        )
        s, a = self.buffer[0][0], self.buffer[0][1]
        s_next = self.buffer[-1][3]
        done = any(self.buffer[i][4] for i in range(self.n_step))
        return (s, a, reward, s_next, done)

    def _compute_nstep_partial(self):
        n = len(self.buffer)
        reward = sum(
            self.gamma ** i * self.buffer[i][2]
            for i in range(n)
        )
        s, a = self.buffer[0][0], self.buffer[0][1]
        s_next = self.buffer[-1][3]
        done = any(self.buffer[i][4] for i in range(n))
        return (s, a, reward, s_next, done)


# ==================================================================
# Uniform Replay Buffer (无 PER)
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
# Attention-Enhanced SAC Actor (与 per_sac_agent 完全一致)
# ==================================================================
class AttSACActor(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.user_attn = UserAttention()
        combined_dim = UAV_FEAT_DIM + ATTENTION_EMBED_DIM
        self.fc1 = nn.Linear(combined_dim, HIDDEN_DIM_1)
        self.ln1 = nn.LayerNorm(HIDDEN_DIM_1)
        self.fc2 = nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2)
        self.ln2 = nn.LayerNorm(HIDDEN_DIM_2)
        self.mean_head = nn.Linear(HIDDEN_DIM_2, a_dim)
        self.log_std_head = nn.Linear(HIDDEN_DIM_2, a_dim)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in [self.mean_head, self.log_std_head]:
            nn.init.uniform_(layer.weight, -3e-3, 3e-3)
            nn.init.zeros_(layer.bias)

    def _encode(self, state):
        uav_feat, user_feat = decompose_state(state)
        user_embed = self.user_attn(user_feat)
        return torch.cat([uav_feat, user_embed], dim=1)

    def forward(self, state):
        x = self._encode(state)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        mean = self.mean_head(x)
        log_std = self.log_std_head(x).clamp(LOG_SIG_MIN, LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t) * ACTION_BOUND[1]
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - (action / ACTION_BOUND[1]).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, mean

    def deterministic(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean) * ACTION_BOUND[1]


# ==================================================================
# Attention-Enhanced Twin Critic (与 per_sac_agent 完全一致)
# ==================================================================
class AttTwinCritic(nn.Module):
    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.user_attn = UserAttention()
        combined_dim = UAV_FEAT_DIM + ATTENTION_EMBED_DIM + a_dim
        self.fc1_1 = nn.Linear(combined_dim, HIDDEN_DIM_1)
        self.ln1_1 = nn.LayerNorm(HIDDEN_DIM_1)
        self.fc1_2 = nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2)
        self.ln1_2 = nn.LayerNorm(HIDDEN_DIM_2)
        self.fc1_3 = nn.Linear(HIDDEN_DIM_2, 1)
        self.fc2_1 = nn.Linear(combined_dim, HIDDEN_DIM_1)
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
        uav_feat, user_feat = decompose_state(state)
        user_embed = self.user_attn(user_feat)
        x = torch.cat([uav_feat, user_embed, action], dim=1)
        q1 = torch.relu(self.ln1_1(self.fc1_1(x)))
        q1 = torch.relu(self.ln1_2(self.fc1_2(q1)))
        q1 = self.fc1_3(q1)
        q2 = torch.relu(self.ln2_1(self.fc2_1(x)))
        q2 = torch.relu(self.ln2_2(self.fc2_2(q2)))
        q2 = self.fc2_3(q2)
        return q1, q2


# ==================================================================
# Attention-SAC Agent (有 Attention, 无 PER)
# ==================================================================
class AttSACAgent:
    """
    消融实验: SAC + Attention + CVX, 但无 PER.
    与 PER-SAC+CVX 唯一区别: 均匀采样.
    """
    def __init__(self, s_dim=STATE_DIM, a_dim=ACTION_DIM):
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.actor = AttSACActor(s_dim, a_dim).to(DEVICE)
        self.critic = AttTwinCritic(s_dim, a_dim).to(DEVICE)
        self.critic_target = AttTwinCritic(s_dim, a_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # 自动温度 α
        self.target_entropy = -float(a_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ACTOR)

        # N-step buffer
        self.n_step_buffer = NStepBuffer(N_STEP, GAMMA)
        self.n_step_gamma = GAMMA ** N_STEP

        # ★ 均匀采样 buffer (无 PER — 唯一区别)
        self.buffer = ReplayBuffer(MEMORY_CAPACITY, s_dim, a_dim)

        self.learn_step = 0
        self.ou_noise = type('', (), {'reset': lambda s: None})()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, noise_std=0.0, **kw):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if noise_std == 0.0:
                action = self.actor.deterministic(state_tensor)
            else:
                action, _, _ = self.actor.sample(state_tensor)
        return action.cpu().numpy()[0]

    def store(self, s, a, r, s_, done):
        nstep = self.n_step_buffer.store((s, a, r, s_, done))
        if nstep is not None:
            self.buffer.store(*nstep)
        if done:
            for leftover in self.n_step_buffer.flush():
                self.buffer.store(*leftover)

    def learn(self):
        if not self.buffer.ready(BATCH_SIZE):
            return {}

        self.learn_step += 1

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(BATCH_SIZE)

        # ---- Critic 更新 ----
        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            y = rewards + self.n_step_gamma * (1 - dones) * target_q

        current_q1, current_q2 = self.critic(states, actions)

        critic_loss = (F.smooth_l1_loss(current_q1, y) +
                       F.smooth_l1_loss(current_q2, y))

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=GRAD_CLIP)
        self.critic_optimizer.step()

        # ---- Actor 延迟更新 ----
        actor_loss_val = 0.0
        if self.learn_step % ACTOR_DELAY == 0:
            new_actions, log_prob, _ = self.actor.sample(states)
            q1_new, q2_new = self.critic(states, new_actions)
            q_new = torch.min(q1_new, q2_new)
            actor_loss = (self.alpha.detach() * log_prob - q_new).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=GRAD_CLIP)
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            # α 自动调节
            alpha_loss = -(self.log_alpha *
                           (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # 软更新 target
            for tp, sp in zip(self.critic_target.parameters(),
                              self.critic.parameters()):
                tp.data.copy_(TAU_SOFT * sp.data + (1 - TAU_SOFT) * tp.data)

        return {
            'critic_loss': critic_loss.item(),
            'avg_q': (current_q1.mean().item() + current_q2.mean().item()) / 2,
            'avg_td_error': 0.0,
            'alpha': self.alpha.item(),
        }

    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'log_alpha': self.log_alpha.data,
        }, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.critic_target.load_state_dict(checkpoint['critic_target'])
        self.log_alpha.data = checkpoint['log_alpha']
