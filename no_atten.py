"""
No-Attention ablation for DARS.

This agent keeps the DARS training pipeline unchanged:
  - SAC stochastic actor
  - twin critics
  - dual-priority PER
  - n-step return
  - IS-weighted actor/critic losses
  - Huber critic loss
  - delayed actor update

The only removed component is the user-level attention encoder. Actor and
critic receive the normalized flat state directly through MLP layers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

import DARS
from config import (
    STATE_DIM, ACTION_DIM, ACTION_BOUND,
    LR_ACTOR, LR_CRITIC, GAMMA,
    MEMORY_CAPACITY,
    HIDDEN_DIM_1, HIDDEN_DIM_2,
)


DEVICE = DARS.DEVICE
LOG_SIG_MAX = DARS.LOG_SIG_MAX
LOG_SIG_MIN = DARS.LOG_SIG_MIN


class NoAttentionSACActor(nn.Module):
    """Flat-state SAC actor used for the w/o Attention ablation."""

    def __init__(self, s_dim, a_dim):
        super().__init__()
        self.fc1 = nn.Linear(s_dim, HIDDEN_DIM_1)
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

    def forward(self, state):
        x = torch.relu(self.ln1(self.fc1(state)))
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


class NoAttentionTwinCritic(nn.Module):
    """Flat-state twin critic used for the w/o Attention ablation."""

    def __init__(self, s_dim, a_dim):
        super().__init__()
        input_dim = s_dim + a_dim
        self.fc1_1 = nn.Linear(input_dim, HIDDEN_DIM_1)
        self.ln1_1 = nn.LayerNorm(HIDDEN_DIM_1)
        self.fc1_2 = nn.Linear(HIDDEN_DIM_1, HIDDEN_DIM_2)
        self.ln1_2 = nn.LayerNorm(HIDDEN_DIM_2)
        self.fc1_3 = nn.Linear(HIDDEN_DIM_2, 1)

        self.fc2_1 = nn.Linear(input_dim, HIDDEN_DIM_1)
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


class NoAttentionPERSACAgent(DARS.PERSACAgent):
    """DARS w/o user attention.

    All inherited methods are reused from DARS.PERSACAgent. Only the actor and
    critic architectures are replaced with flat-state MLPs.
    """

    display_name = "No-Atten"

    def __init__(self, s_dim=STATE_DIM, a_dim=ACTION_DIM):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.learn_step = 0

        self.actor = NoAttentionSACActor(s_dim, a_dim).to(DEVICE)
        self.critic = NoAttentionTwinCritic(s_dim, a_dim).to(DEVICE)
        self.critic_target = NoAttentionTwinCritic(s_dim, a_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        self.target_entropy = -float(a_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ACTOR)

        self.buffer = DARS.PERBufferSAC(MEMORY_CAPACITY, s_dim, a_dim)
        self.n_step_buffer = DARS.NStepBuffer(n_step=DARS.N_STEP, gamma=GAMMA)
        self.n_step_gamma = GAMMA ** DARS.N_STEP
        self.ou_noise = DARS.OUNoise(a_dim)
