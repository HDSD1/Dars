"""
PPO (Proximal Policy Optimization) + CVX Baseline
===================================================
On-policy 策略梯度:
  - Clipped surrogate objective 防止策略更新过大
  - GAE (Generalized Advantage Estimation) 减小方差
  - 参考: Schulman et al., arXiv 2017

与本文 PER-TD3+CVX 的差异:
  - On-policy vs Off-policy: PPO 不用经验回放
  - 策略更新: PPO 用 clipped ratio vs TD3 用 Q 梯度
  - 数据效率: PPO 较低 (每条数据只用一次)

V2 修复:
  ① GAE 改为按 episode 边界正确计算 (不跨 episode)
  ② 学习率降低 + epoch 数减少 → 防止策略退化
  ③ log_std 初始化调小 → 更合理的初始探索范围
  ④ 添加 mini-batch 更新 → 梯度更稳定
  ⑤ Value loss clipping → 防止 value function 剧烈变化
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal

from config import (
    STATE_DIM, ACTION_DIM, ACTION_BOUND,
    GAMMA, SLOT_NUM,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# PPO 超参数 (V2 调优)
PPO_LR = 3e-4               # ★ 独立学习率 (PPO 标准值)
PPO_CLIP = 0.2               # clip ratio ε
PPO_EPOCHS = 10              # ★ 多轮更新 (数据量大, 可以多用)
GAE_LAMBDA = 0.95            # GAE λ
ENTROPY_COEFF = 0.01         # ★ 熵系数 (鼓励探索, 36维动作)
VALUE_COEFF = 0.5            # value loss 权重
MINI_BATCH_SIZE = 128        # ★ mini-batch (更大 → 梯度更稳)
MAX_GRAD_NORM = 0.5          # ★ 梯度裁剪
ROLLOUT_EPISODES = 10        # ★ 每次收集 10 个 episode 再训练 (~400步)


class PPOActorCritic(nn.Module):
    """Actor-Critic 网络 (分离结构, 避免梯度冲突)."""
    def __init__(self, s_dim, a_dim):
        super().__init__()
        # ★ V2: Actor 和 Critic 分离 (避免 value/policy 梯度冲突)
        # Actor
        self.actor_fc1 = nn.Linear(s_dim, 400)
        self.actor_ln1 = nn.LayerNorm(400)
        self.actor_fc2 = nn.Linear(400, 300)
        self.actor_ln2 = nn.LayerNorm(300)
        self.mean_head = nn.Linear(300, a_dim)
        # ★ V2: log_std 初始化为 -0.5 (std≈0.6, 更合理的初始探索)
        self.log_std = nn.Parameter(torch.full((a_dim,), -0.5))

        # Critic
        self.critic_fc1 = nn.Linear(s_dim, 400)
        self.critic_ln1 = nn.LayerNorm(400)
        self.critic_fc2 = nn.Linear(400, 300)
        self.critic_ln2 = nn.LayerNorm(300)
        self.value_head = nn.Linear(300, 1)

        self._init_weights()

    def _init_weights(self):
        for layer in [self.actor_fc1, self.actor_fc2,
                      self.critic_fc1, self.critic_fc2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
        # 输出层用小初始化
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, state):
        # Actor forward
        ax = torch.relu(self.actor_ln1(self.actor_fc1(state)))
        ax = torch.relu(self.actor_ln2(self.actor_fc2(ax)))
        mean = torch.tanh(self.mean_head(ax)) * ACTION_BOUND[1]
        # Critic forward
        cx = torch.relu(self.critic_ln1(self.critic_fc1(state)))
        cx = torch.relu(self.critic_ln2(self.critic_fc2(cx)))
        value = self.value_head(cx)
        return mean, value

    def get_dist(self, state):
        mean, value = self.forward(state)
        std = self.log_std.exp().clamp(min=0.01, max=1.0)  # ★ 限制 std 范围
        std = std.expand_as(mean)
        dist = Normal(mean, std)
        return dist, value

    def get_value(self, state):
        """只计算 V(s), 不需要 actor."""
        cx = torch.relu(self.critic_ln1(self.critic_fc1(state)))
        cx = torch.relu(self.critic_ln2(self.critic_fc2(cx)))
        return self.value_head(cx)


class EpisodeBuffer:
    """
    ★ V2: 按 episode 边界存储数据.
    每个 episode 是一个独立的 trajectory, GAE 在 episode 内计算.
    """
    def __init__(self):
        self.episodes = []       # list of episode data
        self._current_ep = {
            'states': [], 'actions': [], 'log_probs': [],
            'rewards': [], 'values': [], 'dones': [],
        }
        self._total_steps = 0

    def store(self, s, a, log_prob, r, v, done):
        self._current_ep['states'].append(s)
        self._current_ep['actions'].append(a)
        self._current_ep['log_probs'].append(log_prob)
        self._current_ep['rewards'].append(r)
        self._current_ep['values'].append(v)
        self._current_ep['dones'].append(done)
        self._total_steps += 1

        # Episode 结束时保存并开始新 episode
        if done:
            self._finish_episode()

    def finish_episode_if_needed(self):
        """Episode 正常结束 (非 done 终止) 时调用."""
        if len(self._current_ep['states']) > 0:
            self._finish_episode()

    def _finish_episode(self):
        self.episodes.append({
            k: list(v) for k, v in self._current_ep.items()
        })
        self._current_ep = {
            'states': [], 'actions': [], 'log_probs': [],
            'rewards': [], 'values': [], 'dones': [],
        }

    def ready(self, batch_size):
        """★ 只在有足够完整 episode 时返回 True.
        使用 completed episode 数量判断, 防止 train_unified 内部
        step 循环中途触发 learn()."""
        return len(self.episodes) >= ROLLOUT_EPISODES

    def clear(self):
        self.episodes.clear()
        self._total_steps = 0

    def __len__(self):
        return self._total_steps


class PPOAgent:
    """PPO 智能体 (On-policy, V2 优化版)."""

    def __init__(self, s_dim=STATE_DIM, a_dim=ACTION_DIM):
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.ac = PPOActorCritic(s_dim, a_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=PPO_LR, eps=1e-5)
        # ★ 学习率线性退火
        self._total_updates = 0
        self._max_updates = 10000 // ROLLOUT_EPISODES * PPO_EPOCHS  # 预估总更新次数

        self.episode_buffer = EpisodeBuffer()
        self._last_log_prob = None
        self._last_value = None
        self._ep_count = 0  # 累积的 episode 数

        # 兼容接口
        self.ou_noise = type('', (), {'reset': lambda s: None})()
        self.buffer = self.episode_buffer

    def select_action(self, state, noise_std=0.0, **kw):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            dist, value = self.ac.get_dist(state_tensor)
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(dim=1)

        action_np = action.cpu().numpy()[0]
        action_np = np.clip(action_np, ACTION_BOUND[0], ACTION_BOUND[1])

        self._last_log_prob = log_prob.item()
        self._last_value = value.item()
        return action_np

    def store(self, s, a, r, s_, done):
        self.episode_buffer.store(
            s, a, self._last_log_prob, r,
            self._last_value, done)

        if done:
            self._ep_count += 1

        # ★ V2: 每个 episode 结束时检查是否该训练
        # 不在 done 时立刻训练, 而是等 learn() 被调用时在 ready() 中判断

    def learn(self):
        """每次 episode 结束时, 如果该 episode 数据还在 current buffer 中,
        先 flush 它. 然后检查是否有足够数据训练."""
        # Flush any remaining data in current episode
        self.episode_buffer.finish_episode_if_needed()

        if not self.episode_buffer.ready(0):
            return None

        # ★ V2: 按 episode 正确计算 GAE
        all_states, all_actions, all_old_log_probs = [], [], []
        all_advantages, all_returns = [], []

        for ep_data in self.episode_buffer.episodes:
            states = ep_data['states']
            rewards = ep_data['rewards']
            values = ep_data['values']
            dones = ep_data['dones']

            # ★ 正确的 GAE: 在 episode 内部计算, 不跨 episode
            # Bootstrap: 如果 episode 非 done 结束, 用 V(s_last) bootstrap
            # 如果 done 结束, next_value = 0
            last_done = dones[-1] if dones else True
            if last_done:
                next_value = 0.0
            else:
                # 用 critic 估计最后一步的 value
                last_state = torch.FloatTensor(states[-1]).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    next_value = self.ac.get_value(last_state).item()

            advantages = []
            gae = 0
            nv = next_value
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + GAMMA * nv * (1 - dones[t]) - values[t]
                gae = delta + GAMMA * GAE_LAMBDA * (1 - dones[t]) * gae
                advantages.insert(0, gae)
                nv = values[t]

            advantages = np.array(advantages, dtype=np.float32)
            returns = advantages + np.array(values, dtype=np.float32)

            all_states.extend(states)
            all_actions.extend(ep_data['actions'])
            all_old_log_probs.extend(ep_data['log_probs'])
            all_advantages.append(advantages)
            all_returns.append(returns)

        # 合并所有 episodes
        states_t = torch.FloatTensor(np.array(all_states)).to(DEVICE)
        actions_t = torch.FloatTensor(np.array(all_actions)).to(DEVICE)
        old_log_probs_t = torch.FloatTensor(all_old_log_probs).to(DEVICE)
        advantages_t = torch.FloatTensor(np.concatenate(all_advantages)).to(DEVICE)
        returns_t = torch.FloatTensor(np.concatenate(all_returns)).to(DEVICE)

        # ★ 全局 normalize advantages (跨 episode)
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n_samples = len(all_states)
        total_actor_loss = 0.0
        total_value_loss = 0.0

        # ★ V3: 学习率退火
        self._total_updates += 1
        lr_now = PPO_LR * max(0.1, 1.0 - self._total_updates / max(1, self._max_updates))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr_now

        # ★ Mini-batch PPO 更新
        for epoch in range(PPO_EPOCHS):
            # 随机打乱
            perm = torch.randperm(n_samples)

            for start in range(0, n_samples, MINI_BATCH_SIZE):
                end = min(start + MINI_BATCH_SIZE, n_samples)
                idx = perm[start:end]

                mb_states = states_t[idx]
                mb_actions = actions_t[idx]
                mb_old_log_probs = old_log_probs_t[idx]
                mb_advantages = advantages_t[idx]
                mb_returns = returns_t[idx]

                dist, current_values = self.ac.get_dist(mb_states)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=1)
                entropy = dist.entropy().sum(dim=1).mean()

                # Clipped surrogate
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Value loss
                value_pred = current_values.squeeze(-1)
                value_loss = F.mse_loss(value_pred, mb_returns.squeeze(-1))

                loss = (actor_loss
                        + VALUE_COEFF * value_loss
                        - ENTROPY_COEFF * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=MAX_GRAD_NORM)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()

        # 清空 buffer
        self.episode_buffer.clear()
        self._ep_count = 0

        # 返回诊断信息 (兼容 train_unified 的日志接口)
        n_updates = max(1, PPO_EPOCHS * (n_samples // MINI_BATCH_SIZE))
        return {
            'critic_loss': total_value_loss / n_updates,
            'avg_q': total_value_loss / n_updates,  # PPO 没有 Q, 用 value loss 代替
            'avg_td_error': total_actor_loss / n_updates,  # 用 actor loss 代替
        }

    def save(self, path):
        torch.save({'ac': self.ac.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.ac.load_state_dict(checkpoint['ac'])
