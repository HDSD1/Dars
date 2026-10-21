"""
MADDPG (Multi-Agent DDPG) Baseline
====================================
多智能体 DDPG:
  - 每架 UAV 一个独立 Actor (去中心化执行)
  - 共享 Centralized Critic (集中训练)
  - 参考: Lowe et al., NeurIPS 2017

与本文 PER-TD3+CVX 的差异:
  - 架构: MADDPG 多个独立 Actor vs 单一集中式 Actor
  - Critic: MADDPG Critic 接收所有 Agent 的 obs+act
  - 算力分配: 使用 CVX 同本文方法
"""
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    K, M, STATE_DIM, ACTION_DIM, ACTION_BOUND,
    LR_ACTOR, LR_CRITIC, GAMMA, TAU_SOFT,
    MEMORY_CAPACITY, BATCH_SIZE, GRAD_CLIP,
)
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ★ MADDPG 稳定性参数
MADDPG_TAU = 0.005      # 比标准 TAU_SOFT(0.001) 快, 多 Actor 需要更快追踪

# 每个 Agent 的动作维度: 3 (cos, sin, speed) + M/K 个卸载
# 简化: 每个 Agent 控制自己的飞行(3维) + 所有用户的卸载(M维)/K
# 更合理: 每个 Agent 控制自己飞行(3维), 卸载由全局决定
AGENT_ACTION_DIM = 3  # 每个 UAV: cos, sin, speed
OFFLOAD_DIM = M       # 卸载比例仍然全局


class MADDPGActor(nn.Module):
    """每个 Agent 的独立 Actor (只输出自己的飞行动作)."""
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Linear(128, act_dim),
            nn.Tanh(),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, obs):
        return self.net(obs) * ACTION_BOUND[1]


class MADDPGCritic(nn.Module):
    """★ Twin Critic (防止 Q 值过估计, 抑制震荡)."""
    def __init__(self, total_obs_dim, total_act_dim):
        super().__init__()
        inp = total_obs_dim + total_act_dim
        # Q1
        self.q1 = nn.Sequential(
            nn.Linear(inp, 400), nn.LayerNorm(400), nn.ReLU(),
            nn.Linear(400, 300), nn.LayerNorm(300), nn.ReLU(),
            nn.Linear(300, 1),
        )
        # Q2
        self.q2 = nn.Sequential(
            nn.Linear(inp, 400), nn.LayerNorm(400), nn.ReLU(),
            nn.Linear(400, 300), nn.LayerNorm(300), nn.ReLU(),
            nn.Linear(300, 1),
        )
        self._init_weights()

    def _init_weights(self):
        for net in [self.q1, self.q2]:
            for layer in net:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    nn.init.zeros_(layer.bias)

    def forward(self, obs_all, act_all):
        x = torch.cat([obs_all, act_all], dim=1)
        return self.q1(x), self.q2(x)

    def q1_only(self, obs_all, act_all):
        x = torch.cat([obs_all, act_all], dim=1)
        return self.q1(x)


class MADDPGReplayBuffer:
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


class MADDPGAgent:
    """
    MADDPG: 每架 UAV 有独立 Actor, 共享一个 Centralized Critic.
    集中训练, 去中心化执行 (CTDE).
    """
    def __init__(self, s_dim=STATE_DIM, a_dim=ACTION_DIM):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.n_agents = K

        # 每个 Agent 的 Actor (独立)
        self.actors = [MADDPGActor(s_dim, AGENT_ACTION_DIM).to(DEVICE)
                       for _ in range(K)]
        self.actor_targets = [MADDPGActor(s_dim, AGENT_ACTION_DIM).to(DEVICE)
                              for _ in range(K)]

        # 共享的卸载 Actor (控制全局卸载比例)
        self.offload_actor = MADDPGActor(s_dim, OFFLOAD_DIM).to(DEVICE)
        self.offload_actor_target = MADDPGActor(s_dim, OFFLOAD_DIM).to(DEVICE)

        # Centralized Critic
        self.critic = MADDPGCritic(s_dim, a_dim).to(DEVICE)
        self.critic_target = MADDPGCritic(s_dim, a_dim).to(DEVICE)

        # 初始化 target
        for i in range(K):
            self.actor_targets[i].load_state_dict(self.actors[i].state_dict())
        self.offload_actor_target.load_state_dict(self.offload_actor.state_dict())
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器
        self.actor_optimizers = [
            optim.Adam(a.parameters(), lr=LR_ACTOR) for a in self.actors]
        self.offload_optimizer = optim.Adam(
            self.offload_actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(
            self.critic.parameters(), lr=LR_CRITIC)

        self.buffer = MADDPGReplayBuffer(MEMORY_CAPACITY, s_dim, a_dim)
        self.ou_noise = type('', (), {'reset': lambda s: None})()
        self._learn_step = 0

    def select_action(self, state, noise_std=0.0, **kw):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        action = np.zeros(self.a_dim)

        with torch.no_grad():
            # 每个 Agent 独立决策飞行
            for k in range(K):
                agent_act = self.actors[k](state_tensor).cpu().numpy()[0]
                action[3*k: 3*k+3] = agent_act

            # 卸载决策
            offload_act = self.offload_actor(state_tensor).cpu().numpy()[0]
            action[3*K:] = offload_act

        if noise_std > 0:
            action += np.random.normal(0, noise_std, size=self.a_dim)

        return np.clip(action, ACTION_BOUND[0], ACTION_BOUND[1])

    def store(self, s, a, r, s_, done):
        self.buffer.store(s, a, r, s_, done)

    def learn(self):
        if not self.buffer.ready(BATCH_SIZE):
            return
        self._learn_step += 1

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(BATCH_SIZE)

        # ---- Critic 更新 (★ Twin Critic + Huber Loss) ----
        with torch.no_grad():
            next_actions = torch.zeros_like(actions)
            for k in range(K):
                agent_act = self.actor_targets[k](next_states)
                next_actions[:, 3*k:3*k+3] = agent_act
            next_actions[:, 3*K:] = self.offload_actor_target(next_states)

            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)  # ★ Clipped double-Q
            y = rewards + GAMMA * (1 - dones) * target_q

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = (F.smooth_l1_loss(current_q1, y) +
                       F.smooth_l1_loss(current_q2, y))  # ★ Huber loss

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=GRAD_CLIP)
        self.critic_optimizer.step()

        # ---- Actor 更新 (每 2 步) ----
        if self._learn_step % 2 == 0:
            pred_actions = actions.clone().detach()
            for k in range(K):
                agent_act = self.actors[k](states)
                pred_actions[:, 3*k:3*k+3] = agent_act
            pred_actions[:, 3*K:] = self.offload_actor(states)

            actor_loss = -self.critic.q1_only(states, pred_actions).mean()  # ★ 只用 Q1

            # 更新所有 Actor
            for opt in self.actor_optimizers:
                opt.zero_grad()
            self.offload_optimizer.zero_grad()
            actor_loss.backward()
            for k in range(K):
                nn.utils.clip_grad_norm_(
                    self.actors[k].parameters(), max_norm=GRAD_CLIP)
                self.actor_optimizers[k].step()
            nn.utils.clip_grad_norm_(
                self.offload_actor.parameters(), max_norm=GRAD_CLIP)
            self.offload_optimizer.step()

            # ★ 软更新 (MADDPG_TAU=0.005, 比默认更快)
            for k in range(K):
                for tp, sp in zip(self.actor_targets[k].parameters(),
                                  self.actors[k].parameters()):
                    tp.data.copy_(MADDPG_TAU * sp.data + (1-MADDPG_TAU) * tp.data)
            for tp, sp in zip(self.offload_actor_target.parameters(),
                              self.offload_actor.parameters()):
                tp.data.copy_(MADDPG_TAU * sp.data + (1-MADDPG_TAU) * tp.data)
            for tp, sp in zip(self.critic_target.parameters(),
                              self.critic.parameters()):
                tp.data.copy_(MADDPG_TAU * sp.data + (1-MADDPG_TAU) * tp.data)

    def save(self, path):
        state_dict = {
            'critic': self.critic.state_dict(),
            'offload_actor': self.offload_actor.state_dict(),
        }
        for k in range(K):
            state_dict[f'actor_{k}'] = self.actors[k].state_dict()
        torch.save(state_dict, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.critic.load_state_dict(checkpoint['critic'])
        self.offload_actor.load_state_dict(checkpoint['offload_actor'])
        for k in range(K):
            self.actors[k].load_state_dict(checkpoint[f'actor_{k}'])
