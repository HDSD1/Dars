"""
TP-DRL baseline — Task-Priority guided Deep Reinforcement Learning
====================================================================
Adapted from:
  Hao et al., "Joint Task Offloading, Resource Allocation, and Trajectory
  Design for Multi-UAV Cooperative Edge Computing With Task Priority,"
  IEEE Transactions on Mobile Computing, vol. 23, no. 9, pp. 8649-8663, 2024.

Core ideas from the original paper:
  1. Task Priority: tasks have different urgency levels; high-priority tasks
     should be offloaded/served preferentially.
  2. Hybrid Action Space: binary offloading (discrete) + trajectory/resource
     (continuous). The original paper uses a latent-space DRL to handle this.
  3. Single-Critic DDPG backbone (not TD3/SAC).

Fair-comparison adaptation for this codebase:
  - Task Priority Encoder: learns a priority embedding from per-user features
    (task size, active status, distance to nearest UAV) and injects it into
    the actor/critic as auxiliary state information.
  - Binary Offloading Regularization: an auxiliary loss that encourages
    offloading ratios toward 0 or 1, mimicking the paper's binary mode.
  - Single Critic (not twin): matches the original DDPG backbone.
  - Uniform Replay Buffer (no PER): the original paper does not use PER.
  - Priority-Weighted Actor Loss: Q-gradient is scaled by task priority
    importance, so the actor focuses on high-demand decisions.

Key differences from other baselines in this project:
  vs DDPG:     + task priority encoder, + binary offloading regularization
  vs TD3:      single critic (not twin), + task priority, no policy smoothing
  vs SAC:      deterministic policy, single critic, + task priority
  vs DARS:     no PER, no Attention, no dual-priority, single critic, + task priority
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from config import (
    K, M, STATE_DIM, ACTION_DIM, ACTION_BOUND,
    LR_ACTOR, LR_CRITIC, GAMMA, TAU_SOFT,
    MEMORY_CAPACITY, BATCH_SIZE,
    HIDDEN_DIM_1, HIDDEN_DIM_2, GRAD_CLIP,
)

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# TP-DRL 专用超参数
PRIORITY_EMBED_DIM = 32       # 优先级嵌入维度
BINARY_REG_WEIGHT = 0.05      # 二值卸载正则化权重 (鼓励 offload→0 或 1)
BINARY_REG_DECAY = 0.9999     # 正则权重衰减 (训练后期减弱约束)
BINARY_REG_MIN = 0.005        # 正则权重下限
USER_FEAT_DIM = 4             # 每用户特征: (task, active, dist_to_uav_min, loc_norm)


# ==================================================================
# Part ①: Task Priority Encoder — 任务优先级编码器
# ==================================================================
class TaskPriorityEncoder(nn.Module):
    """将每用户特征编码为优先级嵌入向量.

    原论文核心思想: 不同任务具有不同优先级, 高优先级任务应被优先服务.
    本实现通过一个轻量 MLP 将每用户特征 (任务大小, 活跃状态, 距最近UAV距离)
    编码为一个固定维度的优先级向量, 再通过 max-pooling 聚合为全局优先级表示.

    与 DARS 的 UserAttention 的区别:
      - UserAttention 使用 Multi-Head Self-Attention (O(M²)), 学习用户间关系
      - TaskPriorityEncoder 使用逐用户 MLP + max-pool (O(M)), 只关注单用户优先级
      - 更简单、更快, 但缺少用户间交互建模能力 → 这正是消融对比的意义
    """

    def __init__(self, user_feat_dim=USER_FEAT_DIM,
                 embed_dim=PRIORITY_EMBED_DIM):
        super().__init__()
        self.embed_dim = embed_dim
        # 逐用户 MLP: (user_feat_dim) → (embed_dim)
        self.user_mlp = nn.Sequential(
            nn.Linear(user_feat_dim, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.ReLU(),
        )
        self._init_weights()

    def _init_weights(self):
        for layer in self.user_mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user_features):
        """
        Args:
            user_features: (batch, M, user_feat_dim)
        Returns:
            priority_embed: (batch, embed_dim) 全局优先级表示
        """
        # 逐用户编码: (batch, M, user_feat_dim) → (batch, M, embed_dim)
        per_user = self.user_mlp(user_features)
        # Max-pool 聚合: 取每维最大值, 保留最高优先级信号
        # (batch, M, embed_dim) → (batch, embed_dim)
        priority_embed, _ = per_user.max(dim=1)
        return priority_embed


def extract_user_features(state):
    """从 flat state 提取逐用户特征, 用于优先级编码.

    State layout:
      [battery(K), loc_uav(2K), loc_ue(2M), task(M), active(M), wct(3K), time(1)]

    对每个用户提取:
      - task_norm: 归一化任务大小 (0~1)
      - active: 是否活跃 (0 或 1)
      - min_dist_to_uav: 到最近 UAV 的归一化距离
      - loc_norm: 用户位置到场地中心的归一化距离

    Returns:
        user_feat: (batch, M, 4)
    """
    batch = state.shape[0]

    # 解析 state
    uav_loc = state[:, K:3*K].reshape(batch, K, 2)          # (batch, K, 2)
    ue_loc = state[:, 3*K:3*K+2*M].reshape(batch, M, 2)    # (batch, M, 2)
    task = state[:, 3*K+2*M:3*K+3*M]                        # (batch, M)
    active = state[:, 3*K+3*M:3*K+4*M]                      # (batch, M)

    # 归一化任务大小 (按 batch 内最大值)
    task_max = task.max(dim=1, keepdim=True)[0].clamp(min=1e-6)
    task_norm = task / task_max  # (batch, M)

    # 每用户到最近 UAV 的距离
    # ue_loc: (batch, M, 2) → (batch, M, 1, 2)
    # uav_loc: (batch, K, 2) → (batch, 1, K, 2)
    ue_expand = ue_loc.unsqueeze(2)    # (batch, M, 1, 2)
    uav_expand = uav_loc.unsqueeze(1)  # (batch, 1, K, 2)
    dists = torch.norm(ue_expand - uav_expand, dim=3)  # (batch, M, K)
    min_dist, _ = dists.min(dim=2)  # (batch, M)
    # 归一化到 [0, 1] (假设场地对角线 ~707m)
    min_dist_norm = min_dist / 707.0

    # 用户位置到场地中心的归一化距离
    center = torch.tensor([250.0, 250.0], device=state.device)
    loc_dist = torch.norm(ue_loc - center, dim=2) / 354.0  # (batch, M)

    # 拼接: (batch, M, 4)
    user_feat = torch.stack([task_norm, active, min_dist_norm, loc_dist], dim=2)
    return user_feat


# ==================================================================
# Part ②: TP-DRL Actor — 任务优先级增强的确定性策略网络
# ==================================================================
class TPActor(nn.Module):
    """Task-Priority enhanced deterministic Actor.

    与标准 DDPG Actor 的区别:
      - 输入不是 raw state, 而是 [state, priority_embed] 的拼接
      - TaskPriorityEncoder 作为网络的一部分, 端到端训练
      - 飞行动作 (3K 维) 和卸载动作 (M 维) 共享同一个 trunk,
        但卸载输出经过 sigmoid 鼓励二值化 (原论文 binary offloading)

    与 DARS 的 SACActor 的区别:
      - 确定性策略 (不输出 mean/std), 不采样
      - 无 Attention, 使用更简单的 priority encoder
      - 单网络, 无 log_std_head
    """

    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.n_fly = 3 * K     # 飞行动作维度
        self.n_offload = M     # 卸载动作维度

        # 优先级编码器 (端到端训练)
        self.priority_encoder = TaskPriorityEncoder()
        combined_dim = s_dim + PRIORITY_EMBED_DIM

        # 共享 trunk
        self.fc1 = nn.Linear(combined_dim, HIDDEN_DIM_1)
        self.ln1 = nn.LayerNorm(HIDDEN_DIM_1)
        self.fc2 = nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2)
        self.ln2 = nn.LayerNorm(HIDDEN_DIM_2)

        # 飞行头: 输出 cos, sin, speed (tanh 映射到 [-1, 1])
        self.fly_head = nn.Linear(HIDDEN_DIM_2, self.n_fly)
        # 卸载头: 输出卸载比例 (tanh 映射到 [-1, 1])
        self.offload_head = nn.Linear(HIDDEN_DIM_2, self.n_offload)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in [self.fly_head, self.offload_head]:
            nn.init.uniform_(layer.weight, -3e-3, 3e-3)
            nn.init.zeros_(layer.bias)

    def forward(self, state):
        # 提取优先级嵌入
        user_feat = extract_user_features(state)
        priority_embed = self.priority_encoder(user_feat)

        # 拼接: [state, priority_embed]
        x = torch.cat([state, priority_embed], dim=1)

        # 共享 trunk
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))

        # 飞行动作: tanh → [-1, 1]
        fly_action = torch.tanh(self.fly_head(x)) * ACTION_BOUND[1]
        # 卸载动作: tanh → [-1, 1]
        offload_action = torch.tanh(self.offload_head(x)) * ACTION_BOUND[1]

        return torch.cat([fly_action, offload_action], dim=1)

    def get_offload_logits(self, state):
        """返回卸载头的 raw logits (用于二值正则化).

        二值正则化: L_binary = -mean(|sigmoid(logit) - 0.5|)
        鼓励 sigmoid 输出趋向 0 或 1, 模拟原论文的 binary offloading.
        """
        user_feat = extract_user_features(state)
        priority_embed = self.priority_encoder(user_feat)
        x = torch.cat([state, priority_embed], dim=1)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        return self.offload_head(x)


# ==================================================================
# Part ③: TP-DRL Critic — 单 Q 网络 (非 TD3 双 Critic)
# ==================================================================
class TPCritic(nn.Module):
    """Task-Priority enhanced single Q-network.

    与 TD3/SAC 的 TwinCritic 的核心区别:
      - 只有 1 个 Q 网络 (原论文使用标准 DDPG, 不含 clipped double-Q)
      - 输入同样包含 priority_embed, 使 Q 估计感知任务优先级

    与标准 DDPG Critic 的区别:
      - 输入拼接了 priority_embed: [state, priority_embed, action]
      - 使 Critic 能够区分 "同样 reward 但不同任务优先级分布" 的状态
    """

    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.priority_encoder = TaskPriorityEncoder()
        combined_dim = s_dim + PRIORITY_EMBED_DIM + a_dim

        self.fc1 = nn.Linear(combined_dim, HIDDEN_DIM_1)
        self.ln1 = nn.LayerNorm(HIDDEN_DIM_1)
        self.fc2 = nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2)
        self.ln2 = nn.LayerNorm(HIDDEN_DIM_2)
        self.fc3 = nn.Linear(HIDDEN_DIM_2, 1)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        nn.init.uniform_(self.fc3.weight, -3e-3, 3e-3)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, state, action):
        user_feat = extract_user_features(state)
        priority_embed = self.priority_encoder(user_feat)
        x = torch.cat([state, priority_embed, action], dim=1)

        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        return self.fc3(x)


# ==================================================================
# Part ④: 均匀回放缓冲区 (无 PER)
# ==================================================================
class UniformReplayBuffer:
    """标准均匀采样经验回放.

    与 PER 的区别:
      - 无 SumTree, 无优先级, 无 IS 权重
      - 所有 transition 被采样的概率相同
      - 原论文 (Hao et al., TMC 2024) 未使用 PER
    """

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
# OU Noise (标准 Ornstein-Uhlenbeck 探索噪声)
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
# Part ⑤: TP-DRL Agent 主体
# ==================================================================
class TPDRLAgent:
    """Task-Priority guided DRL Agent.

    训练循环与标准 DDPG 类似, 但有三处关键差异:
      1. Actor/Critic 都包含 TaskPriorityEncoder (端到端训练)
      2. Actor loss 附加二值卸载正则化项
      3. 单 Critic (无 clipped double-Q)

    Loss 定义:
      L_critic = MSE(Q(s,a), r + γ Q'(s', μ'(s')))
      L_actor  = -Q(s, μ(s)) + λ_binary · L_binary
      L_binary = -mean(|sigmoid(offload_logits) - 0.5|)
                 (鼓励 sigmoid 输出趋向 0 或 1)
    """

    display_name = "TP-DRL"

    def __init__(self, s_dim=STATE_DIM, a_dim=ACTION_DIM):
        self.s_dim = s_dim
        self.a_dim = a_dim

        # 网络
        self.actor = TPActor(s_dim, a_dim).to(DEVICE)
        self.actor_target = TPActor(s_dim, a_dim).to(DEVICE)
        self.critic = TPCritic(s_dim, a_dim).to(DEVICE)
        self.critic_target = TPCritic(s_dim, a_dim).to(DEVICE)

        # 硬拷贝初始化 target 网络
        self._hard_update(self.actor_target, self.actor)
        self._hard_update(self.critic_target, self.critic)

        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # 均匀回放缓冲区 (无 PER)
        self.buffer = UniformReplayBuffer(MEMORY_CAPACITY, s_dim, a_dim)
        self.ou_noise = OUNoise(a_dim, sigma=0.2)

        # 二值正则化权重 (训练过程中衰减)
        self._binary_reg_weight = BINARY_REG_WEIGHT
        self._learn_step = 0

    def select_action(self, state, noise_std=0.0, use_ou=False, **kw):
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
            return {}

        self._learn_step += 1

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(BATCH_SIZE)

        # ---- Critic 更新 ----
        with torch.no_grad():
            next_actions = self.actor_target(next_states)
            target_q = self.critic_target(next_states, next_actions)
            y = rewards + GAMMA * (1 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.functional.mse_loss(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=GRAD_CLIP)
        self.critic_optimizer.step()

        # ---- Actor 更新 ----
        pred_actions = self.actor(states)
        # 策略梯度: 最大化 Q
        pg_loss = -self.critic(states, pred_actions).mean()

        # ★ 二值卸载正则化 (原论文 binary offloading 的近似)
        # L_binary = -mean(|sigmoid(logits) - 0.5|)
        # sigmoid(logit) ∈ (0,1), 当输出趋向 0 或 1 时 |·-0.5| → 0.5 (最大)
        # 取负号使其成为最小化目标 → 鼓励二值化
        offload_logits = self.actor.get_offload_logits(states)
        binary_loss = -torch.abs(torch.sigmoid(offload_logits) - 0.5).mean()

        actor_loss = pg_loss + self._binary_reg_weight * binary_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=GRAD_CLIP)
        self.actor_optimizer.step()

        # ---- 软更新 target 网络 ----
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)

        # ---- 衰减二值正则权重 ----
        self._binary_reg_weight = max(
            BINARY_REG_MIN,
            self._binary_reg_weight * BINARY_REG_DECAY,
        )

        return {
            'critic_loss': critic_loss.item(),
            'avg_q': current_q.mean().item(),
            'avg_td_error': (y - current_q).abs().mean().item(),
            'binary_reg': self._binary_reg_weight,
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
