"""
PER-SAC+CVX Agent (本文方法, V3 — 任务密度感知双优先级)
=======================================================
将 SAC 的最大熵框架与 **Dual-Priority PER** 结合:
  - SAC Actor: 随机策略 π(a|s), 自动温度 α
  - Twin Critic: 双 Q 网络, 取 min 防过估计
  - ★ Dual-Priority PER: TD-error + Reward Surprise 双优先级
  - CVX: 凸优化算力分配 (环境层, 此文件不涉及)

V3 改进 (vs V2):
  ① 双优先级 (Dual Priority): p_i = p_td + λ·p_surprise
     - p_td = (|δ_i| + ε)^α  — 标准 TD-error 优先级
     - p_surprise = (|r_i - r̄| + ε)^α  — 奖励偏差优先级
     r̄ 为奖励的 EMA, λ 控制两者权重.
     学术动机: 在热/冷簇轮转场景中, 任务密度突变时刻产生的
     transition 对策略学习价值极高, 但 TD-error 可能未能及时
     反映这种环境突变. 通过 reward surprise 补充, PER 能在
     簇切换时优先回放关键决策经验, 显著提升收敛速度.
  ② N-step Return (3步) → TD-error 信号更准, PER 优先级更有效
  ③ IS 加权 Actor Loss → 修正 PER 非均匀采样对 actor 梯度的偏差
  ④ Huber Loss (Critic) → 对高 TD-error outlier 更鲁棒
  ⑤ 延迟 Actor 更新 (每2步) → PER 方差大, 让 critic 先稳定

参考:
  - Haarnoja et al., ICML 2018 (SAC)
  - Schaul et al., ICLR 2016 (PER)
  - Hessel et al., AAAI 2018 (Rainbow: n-step + PER 组合)
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

# ★ PER-SAC 专用超参数 (覆盖 config.py, 不影响 PER-DDPG)
#   - α 降低: 高维动作空间需要更多样本多样性
#   - β 退火加速: 确保训练中期 IS 权重已无偏
#   - ε 增大: 防止低 TD-error 样本永远采不到
PER_SAC_ALPHA = 0.4
PER_SAC_BETA_START = 0.4
PER_SAC_BETA_FRAMES = 40000
PER_SAC_EPSILON = 1e-5
PER_SAC_MAX_IS_WEIGHT = 3.0

# ★ V3: 双优先级 — Reward Surprise 参数
LAMBDA_SURPRISE = 0.3           # 奖励偏差优先级的权重 (0=退化为纯 TD-error PER)
REWARD_EMA_DECAY = 0.995        # 奖励 EMA 衰减率 (越大越平滑)

# N-step return 参数
N_STEP = 3

# 延迟更新
ACTOR_DELAY = 2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20

# ★ V4: Task-Demand Attention 参数
ATTENTION_EMBED_DIM = 64    # 用户特征 attention 嵌入维度
ATTENTION_HEADS = 2         # Multi-Head Attention 头数
USER_FEAT_DIM = 4           # 每用户特征维度: (loc_x, loc_y, task, active)
UAV_FEAT_DIM = 6 * K + 1    # UAV 级特征维度: battery(K)+loc(2K)+wct(3K)+time(1)


def decompose_state(state):
    """将 flat state 向量拆分为 UAV 特征和逐用户特征.

    State layout:
      [battery(K), loc_uav(2K), loc_ue(2M), task(M), active(M), wct(3K), time(1)]

    Returns:
        uav_feat:  (batch, 6K+1) UAV 级特征
        user_feat: (batch, M, 4) 逐用户特征 [x, y, task, active]
    """
    batch = state.shape[0]
    battery = state[:, :K]
    uav_loc = state[:, K:3*K]
    wct_info = state[:, 3*K+4*M:6*K+4*M]
    time_step = state[:, 6*K+4*M:6*K+4*M+1]
    uav_feat = torch.cat([battery, uav_loc, wct_info, time_step], dim=1)

    ue_loc = state[:, 3*K:3*K+2*M].reshape(batch, M, 2)
    task = state[:, 3*K+2*M:3*K+3*M].unsqueeze(2)
    active = state[:, 3*K+3*M:3*K+4*M].unsqueeze(2)
    user_feat = torch.cat([ue_loc, task, active], dim=2)

    return uav_feat, user_feat


class UserAttention(nn.Module):
    """★ V4: Task-Demand Attention — 自动关注高任务密度用户.

    学术动机:
      在热/冷簇轮转场景中, 仅少数用户（热簇）贡献了大部分计算任务.
      标准 MLP 将所有用户特征扁平拼接, 无法区分重要用户.
      通过 Multi-Head Self-Attention, Agent 自动学习关注高任务密度
      区域的用户, 减少冷簇噪声干扰, 提升决策质量.

    参考:
      - Vaswani et al., NeurIPS 2017 (Transformer)
      - Iqbal & Sha, ICML 2019 (MARL + Attention)
    """
    def __init__(self, feat_dim=USER_FEAT_DIM, embed_dim=ATTENTION_EMBED_DIM,
                 n_heads=ATTENTION_HEADS):
        super().__init__()
        self.embed_dim = embed_dim
        self.user_proj = nn.Sequential(
            nn.Linear(feat_dim, embed_dim),
            nn.ReLU(),
        )
        self.mha = nn.MultiheadAttention(embed_dim, n_heads, batch_first=True)
        self.ln = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self):
        for layer in self.user_proj:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user_features, return_tokens=False):
        """
        Args:   user_features: (batch, M, 4)
        Returns: 默认返回 (batch, embed_dim) 全局表示；
            return_tokens=True 时返回 (batch, M, embed_dim) 逐用户表示。
        """
        x = self.user_proj(user_features)
        attn_out, _ = self.mha(x, x, x)
        attn_out = self.ln(attn_out + x)
        return attn_out if return_tokens else attn_out.mean(dim=1)


# ==================================================================
# N-Step Buffer — 计算 n-step return
# ==================================================================
class NStepBuffer:
    """
    存储最近 n 步 transition, 计算 n-step discounted return.
    R_n = r_0 + γ r_1 + γ² r_2 + ... + γ^(n-1) r_{n-1}
    s'  = s_n (n 步后的状态)
    done = any(done_0, ..., done_{n-1})

    配合 PER 使用时, n-step TD-error 信噪比更高,
    使得 PER 的优先级排序更有意义 (Rainbow, Hessel 2018).
    """
    def __init__(self, n_step=N_STEP, gamma=GAMMA):
        self.n_step = n_step
        self.gamma = gamma
        self.buffer = deque(maxlen=n_step)

    def store(self, transition):
        """
        存入一个 (s, a, r, s', done) 元组.
        当 buffer 满 n 步时, 返回合并后的 n-step transition.
        否则返回 None (等待更多 step).
        """
        self.buffer.append(transition)
        if len(self.buffer) == self.n_step:
            return self._compute_nstep()
        return None

    def flush(self):
        """
        Episode 结束时, 刷出 buffer 中剩余的 transition.
        返回剩余可合并的 transitions 列表.
        """
        results = []
        while len(self.buffer) > 0:
            results.append(self._compute_nstep_partial())
            self.buffer.popleft()
        return results

    def _compute_nstep(self):
        """从满的 buffer 中计算 n-step transition."""
        reward = sum(
            self.gamma ** i * self.buffer[i][2]
            for i in range(self.n_step)
        )
        s, a = self.buffer[0][0], self.buffer[0][1]
        s_next = self.buffer[-1][3]
        # 如果中途任意一步 done, 则整个 n-step 标记为 done
        done = any(self.buffer[i][4] for i in range(self.n_step))
        return (s, a, reward, s_next, done)

    def _compute_nstep_partial(self):
        """从不满的 buffer 中计算 partial n-step transition."""
        n = len(self.buffer)
        reward = sum(
            self.gamma ** i * self.buffer[i][2]
            for i in range(n)
        )
        s, a = self.buffer[0][0], self.buffer[0][1]
        s_next = self.buffer[-1][3]
        done = any(self.buffer[i][4] for i in range(n))
        return (s, a, reward, s_next, done)

    def reset(self):
        """新 episode 时清空."""
        self.buffer.clear()


# ==================================================================
# PER Buffer (SAC 专用版, 覆盖超参数)
# ==================================================================
from per_ddpg_agent import SumTree


class PERBufferSAC:
    """
    ★ V3: Dual-Priority PER 缓冲区 — 任务密度感知.

    优先级计算:
      p_i = p_td + λ · p_surprise
      p_td      = (|δ_i| + ε)^α          — TD-error 优先级 (标准 PER)
      p_surprise = (|r_i - r̄| + ε)^α     — 奖励偏差优先级 (本文创新)

    学术动机:
      在热/冷簇轮转 MEC 场景中, 任务密度在阶段切换时刻发生突变.
      此时 UAV 的服务策略需要快速适应, 但 TD-error 可能因 Critic
      滞后而无法及时识别这些关键转移. 通过 reward surprise (|r-r̄|)
      补充, 使 PER 在簇切换时优先回放负载突变经验, 显著加速策略适应.
    """
    def __init__(self, capacity, s_dim, a_dim, lambda_surprise=None):
        self.capacity = capacity
        self.tree = SumTree(capacity)
        self.size = 0
        self.lambda_surprise = LAMBDA_SURPRISE if lambda_surprise is None else lambda_surprise

        self.states = np.zeros((capacity, s_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, a_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, s_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self._frame = 0

        # ★ V3: 奖励 EMA 与 Surprise EMA 跟踪
        self._reward_ema = 0.0
        self._surprise_ema = 0.0
        self._ema_initialized = False

    def store(self, s, a, r, s_, done):
        idx = self.tree.data_pointer
        self.states[idx] = s
        self.actions[idx] = a
        self.rewards[idx] = r
        self.next_states[idx] = s_
        self.dones[idx] = float(done)

        # ★ V3: 更新奖励 EMA 与 Surprise EMA
        r_scalar = float(r) if np.isscalar(r) else float(r[0])
        if not self._ema_initialized:
            self._reward_ema = r_scalar
            self._surprise_ema = 0.0
            self._ema_initialized = True
        else:
            surprise = abs(r_scalar - self._reward_ema)
            self._reward_ema = (REWARD_EMA_DECAY * self._reward_ema +
                                (1 - REWARD_EMA_DECAY) * r_scalar)
            self._surprise_ema = (REWARD_EMA_DECAY * self._surprise_ema +
                                  (1 - REWARD_EMA_DECAY) * surprise)

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
                   PER_SAC_BETA_START + self._frame *
                   (1.0 - PER_SAC_BETA_START) / PER_SAC_BETA_FRAMES)

        indices, tree_indices, priorities = [], [], []
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            value = np.random.uniform(segment * i, segment * (i + 1))
            tree_idx, priority, data_idx = self.tree.get_leaf(value)
            data_idx = np.clip(data_idx, 0, self.size - 1)
            indices.append(data_idx)
            tree_indices.append(tree_idx)
            priorities.append(priority)

        priorities = np.array(priorities, dtype=np.float32) + PER_SAC_EPSILON
        sampling_probs = priorities / (self.tree.total_priority + PER_SAC_EPSILON)
        is_weights = np.power(self.size * sampling_probs, -beta)
        is_weights /= is_weights.max()
        is_weights = np.clip(is_weights, 0.0, PER_SAC_MAX_IS_WEIGHT)

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

    def update_priorities(self, tree_indices, td_errors, rewards_batch=None):
        """
        ★ V3: 双优先级更新.
        p_i = p_td + λ · p_surprise

        Args:
            tree_indices: SumTree 节点索引
            td_errors: |δ_i| 数组
            rewards_batch: 对应的 reward 值 (用于计算 surprise)
        """
        for i, (tree_idx, td_err) in enumerate(zip(tree_indices, td_errors)):
            # 标准 TD-error 优先级
            p_td = (abs(td_err) + PER_SAC_EPSILON) ** PER_SAC_ALPHA

            # ★ 奖励偏差优先级 (reward surprise)
            if rewards_batch is not None:
                r_i = float(rewards_batch[i])
                surprise = abs(r_i - self._reward_ema)
                p_surprise = (surprise + PER_SAC_EPSILON) ** PER_SAC_ALPHA
                priority = p_td + self.lambda_surprise * p_surprise
            else:
                priority = p_td

            self.tree.update(tree_idx, priority)

    @property
    def reward_ema(self):
        return self._reward_ema

    @property
    def surprise_ema(self):
        return self._surprise_ema

    def ready(self, batch_size):
        return self.size >= batch_size


# ==================================================================
# ★ V4: Attention-Enhanced SAC Actor
# ==================================================================
class SACActor(nn.Module):
    """★ V4: Attention-Enhanced Actor.

    全局上下文用于 UAV 动作；逐用户表示与全局上下文拼接后，
    通过参数共享的卸载头产生对应用户的动作分布。
    """
    def __init__(self, s_dim, a_dim):
        super().__init__()
        if a_dim != 3 * K + M:
            raise ValueError("SACActor requires action layout [3K movement, M offloading]")
        self.user_attn = UserAttention()
        combined_dim = UAV_FEAT_DIM + ATTENTION_EMBED_DIM
        self.fc1 = nn.Linear(combined_dim, HIDDEN_DIM_1)
        self.ln1 = nn.LayerNorm(HIDDEN_DIM_1)
        self.fc2 = nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2)
        self.ln2 = nn.LayerNorm(HIDDEN_DIM_2)
        self.mean_head = nn.Linear(HIDDEN_DIM_2, 3 * K)
        self.log_std_head = nn.Linear(HIDDEN_DIM_2, 3 * K)
        self.offload_fc = nn.Linear(
            ATTENTION_EMBED_DIM + HIDDEN_DIM_2, ATTENTION_EMBED_DIM)
        self.offload_mean_head = nn.Linear(ATTENTION_EMBED_DIM, 1)
        self.offload_log_std_head = nn.Linear(ATTENTION_EMBED_DIM, 1)
        self._init_weights()

    def _init_weights(self):
        for layer in [self.fc1, self.fc2, self.offload_fc]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in [self.mean_head, self.log_std_head,
                      self.offload_mean_head, self.offload_log_std_head]:
            nn.init.uniform_(layer.weight, -3e-3, 3e-3)
            nn.init.zeros_(layer.bias)

    def _encode(self, state):
        uav_feat, user_feat = decompose_state(state)
        user_tokens = self.user_attn(user_feat, return_tokens=True)
        return torch.cat([uav_feat, user_tokens.mean(dim=1)], dim=1), user_tokens

    def forward(self, state):
        x, user_tokens = self._encode(state)
        x = torch.relu(self.ln1(self.fc1(x)))
        x = torch.relu(self.ln2(self.fc2(x)))
        context = x.unsqueeze(1).expand(-1, user_tokens.shape[1], -1)
        offload = torch.relu(self.offload_fc(torch.cat([user_tokens, context], dim=-1)))
        mean = torch.cat([self.mean_head(x),
                          self.offload_mean_head(offload).squeeze(-1)], dim=1)
        log_std = torch.cat([self.log_std_head(x),
                             self.offload_log_std_head(offload).squeeze(-1)], dim=1)
        log_std = log_std.clamp(LOG_SIG_MIN, LOG_SIG_MAX)
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
# ★ V4: Attention-Enhanced Twin Critic
# ==================================================================
class TwinCritic(nn.Module):
    """★ V4: Attention-Enhanced Twin Critic.

    共享 UserAttention 编码器, 两个独立 Q 网络.
    """
    def __init__(self, s_dim, a_dim):
        super().__init__()
        if a_dim != 3 * K + M:
            raise ValueError("TwinCritic requires action layout [3K movement, M offloading]")
        self.user_attn = UserAttention()
        # 在聚合前编码用户与其卸载动作的配对，避免丢失对应关系。
        self.user_action_encoder = nn.Sequential(
            nn.Linear(ATTENTION_EMBED_DIM + 1, ATTENTION_EMBED_DIM),
            nn.ReLU(),
            nn.Linear(ATTENTION_EMBED_DIM, ATTENTION_EMBED_DIM),
            nn.ReLU(),
        )
        combined_dim = UAV_FEAT_DIM + 2 * ATTENTION_EMBED_DIM + 3 * K
        # Q1
        self.fc1_1 = nn.Linear(combined_dim, HIDDEN_DIM_1)
        self.ln1_1 = nn.LayerNorm(HIDDEN_DIM_1)
        self.fc1_2 = nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2)
        self.ln1_2 = nn.LayerNorm(HIDDEN_DIM_2)
        self.fc1_3 = nn.Linear(HIDDEN_DIM_2, 1)
        # Q2
        self.fc2_1 = nn.Linear(combined_dim, HIDDEN_DIM_1)
        self.ln2_1 = nn.LayerNorm(HIDDEN_DIM_1)
        self.fc2_2 = nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2)
        self.ln2_2 = nn.LayerNorm(HIDDEN_DIM_2)
        self.fc2_3 = nn.Linear(HIDDEN_DIM_2, 1)
        self._init_weights()

    def _init_weights(self):
        for layer in self.user_action_encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
        for layer in [self.fc1_1, self.fc1_2, self.fc2_1, self.fc2_2]:
            nn.init.xavier_uniform_(layer.weight)
            nn.init.zeros_(layer.bias)
        for layer in [self.fc1_3, self.fc2_3]:
            nn.init.uniform_(layer.weight, -3e-3, 3e-3)
            nn.init.zeros_(layer.bias)

    def forward(self, state, action):
        uav_feat, user_feat = decompose_state(state)
        user_tokens = self.user_attn(user_feat, return_tokens=True)
        offload_action = action[:, 3 * K:].unsqueeze(-1)
        user_action_embed = self.user_action_encoder(
            torch.cat([user_tokens, offload_action], dim=-1)).mean(dim=1)
        x = torch.cat([uav_feat, user_tokens.mean(dim=1),
                       user_action_embed, action[:, :3 * K]], dim=1)
        q1 = torch.relu(self.ln1_1(self.fc1_1(x)))
        q1 = torch.relu(self.ln1_2(self.fc1_2(q1)))
        q1 = self.fc1_3(q1)
        q2 = torch.relu(self.ln2_1(self.fc2_1(x)))
        q2 = torch.relu(self.ln2_2(self.fc2_2(q2)))
        q2 = self.fc2_3(q2)
        return q1, q2


# ==================================================================
# 兼容接口: OUNoise
# ==================================================================
from per_ddpg_agent import OUNoise


# ==================================================================
# PER-SAC Agent (V2 优化版)
# ==================================================================
class PERSACAgent:
    """
    PER-SAC+CVX: 最大熵 + 优先经验回放 (V2 优化版).

    核心: J(π) = E[Σ γ^t (r_t + α H(π(·|s_t)))]
    + PER 按 TD-error 优先采样高价值经验.

    V2 改进:
      ① n-step return (3步) → PER 优先级更精准
      ② IS 加权 actor loss → 无偏策略梯度
      ③ Huber loss → critic 对 outlier 鲁棒
      ④ 延迟 actor 更新 → 减少 PER 方差影响
      ⑤ SAC 专用 PER 超参数 → α↓ β退火↑
    """
    def __init__(self, s_dim=STATE_DIM, a_dim=ACTION_DIM):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.learn_step = 0

        # 网络
        self.actor = SACActor(s_dim, a_dim).to(DEVICE)
        self.critic = TwinCritic(s_dim, a_dim).to(DEVICE)
        self.critic_target = TwinCritic(s_dim, a_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # 优化器 (使用 config.py 统一管理的学习率)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # 自动温度 α
        self.target_entropy = -float(a_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ACTOR)

        # ★ PER 经验回放 (SAC 专用超参数)
        self.buffer = PERBufferSAC(MEMORY_CAPACITY, s_dim, a_dim)

        # ★ N-step buffer
        self.n_step_buffer = NStepBuffer(n_step=N_STEP, gamma=GAMMA)
        self.n_step_gamma = GAMMA ** N_STEP  # γ^n, 用于 target Q 折扣

        self.ou_noise = OUNoise(a_dim)  # 兼容接口

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, noise_std=0.0, **kw):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if noise_std == 0.0:
                # 确定性模式 (评估)
                action = self.actor.deterministic(state_tensor)
            else:
                # 随机策略 (训练)
                action, _, _ = self.actor.sample(state_tensor)
        return action.cpu().numpy()[0]

    def store(self, s, a, r, s_, done):
        """
        ★ V2: 通过 n-step buffer 存储.
        每次存入一步, 当累积 n 步后, 将 n-step transition 存入 PER buffer.
        Episode 结束时 (done=True), 刷出 buffer 中剩余的 transitions.
        """
        transition = (s, a, r, s_, done)
        nstep_transition = self.n_step_buffer.store(transition)

        if nstep_transition is not None:
            self.buffer.store(*nstep_transition)

        # Episode 结束时刷出剩余
        if done:
            remaining = self.n_step_buffer.flush()
            for t in remaining:
                self.buffer.store(*t)
            self.n_step_buffer.reset()

    def learn(self):
        if not self.buffer.ready(BATCH_SIZE):
            return {}

        self.learn_step += 1

        # PER 采样 (带 IS 权重)
        states, actions, rewards, next_states, dones, is_weights, tree_indices = \
            self.buffer.sample(BATCH_SIZE)

        # ---- Critic 更新 ----
        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            # ★ V2: 使用 n-step gamma (γ^n)
            y = rewards + self.n_step_gamma * (1 - dones) * target_q

        current_q1, current_q2 = self.critic(states, actions)

        # TD-error 用于 PER 优先级更新
        td_error1 = (current_q1 - y).detach().cpu().numpy().flatten()
        td_error2 = (current_q2 - y).detach().cpu().numpy().flatten()
        td_errors = (np.abs(td_error1) + np.abs(td_error2)) / 2.0

        # ★ V2: PER 加权 + Huber Loss (对高 TD-error outlier 更鲁棒)
        elementwise_loss = (F.smooth_l1_loss(current_q1, y, reduction='none') +
                            F.smooth_l1_loss(current_q2, y, reduction='none'))
        critic_loss = (is_weights * elementwise_loss).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=GRAD_CLIP)
        self.critic_optimizer.step()

        # ★ V3: 双优先级更新 (TD-error + reward surprise)
        rewards_np = rewards.detach().cpu().numpy().flatten()
        self.buffer.update_priorities(tree_indices, td_errors, rewards_np)

        # ---- Actor 更新 (★ V2: 延迟更新, 每 ACTOR_DELAY 步) ----
        actor_loss_val = 0.0
        if self.learn_step % ACTOR_DELAY == 0:
            new_actions, log_prob, _ = self.actor.sample(states)
            q1_new, q2_new = self.critic(states, new_actions)
            q_new = torch.min(q1_new, q2_new)

            # ★ V2: IS 加权 Actor Loss (修正 PER 非均匀采样偏差)
            actor_loss = (is_weights.detach() *
                          (self.alpha.detach() * log_prob - q_new)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=GRAD_CLIP)
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            # ---- α 自动调节 (也延迟, 与 actor 同步) ----
            # ★ Phase-Aware Adaptive Temperature (阶段感知自适应温度)
            # 当 surprise_ema 增大时, 目标熵约束更加严格(即要求更高的熵, target_entropy 更负)
            # 从而强制 α 升高，进行强力探索以适应簇切换。
            surprise = self.buffer.surprise_ema
            # surprise 一般在 0~50 波动, 适当缩放
            exploration_boost = 1.0 + min(2.0, surprise / 40.0) 
            dynamic_target_entropy = self.target_entropy * exploration_boost

            alpha_loss = -(self.log_alpha *
                           (log_prob + dynamic_target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # ---- 软更新 target (与 actor 同步) ----
            for tp, sp in zip(self.critic_target.parameters(),
                              self.critic.parameters()):
                tp.data.copy_(TAU_SOFT * sp.data + (1 - TAU_SOFT) * tp.data)

        # 返回诊断指标
        return {
            'critic_loss': critic_loss.item(),
            'avg_q': (current_q1.mean().item() + current_q2.mean().item()) / 2,
            'avg_td_error': np.mean(td_errors),
            'alpha': self.alpha.item(),
            'reward_ema': self.buffer.reward_ema,  # ★ V3: 监控 EMA
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
