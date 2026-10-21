"""
ESAC-PER Baseline (TMC 2025)
=============================
Reference Paper:
  "Joint Offloading Decision, User Association, and Resource Allocation in Hierarchical
   Aerial Computing: Collaboration of UAVs and HAP"
   IEEE Transactions on Mobile Computing (TMC), Vol. 24, Issue 8, August 2025.
   Authors: Ahmadun Nabi and Sangman Moh.

Core Algorithmic Features:
  1. Enhanced Soft Actor-Critic (ESAC): Maximum entropy reinforcement learning framework 
     using Twin Critic Networks (min-Q value estimation) and dynamic policy search.
  2. Prioritized Experience Replay (PER): Prioritizes high TD-error transitions using
     a SumTree structure, accelerating exploration and convergence stability.
  3. Dynamic Temperature Auto-Tuning: Adapts the entropy regularizer alpha dynamically 
     to maintain consistent exploration in complex environments.
  4. Standard MLP Topology (No Attention): Uses multilayer perceptrons with LayerNorm
     and Xavier initialization as modeled in the journal paper's continuous control setup.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
from collections import deque

from config import (
    STATE_DIM, ACTION_DIM, ACTION_BOUND,
    LR_ACTOR, LR_CRITIC, GAMMA, TAU_SOFT,
    MEMORY_CAPACITY, BATCH_SIZE,
    HIDDEN_DIM_1, HIDDEN_DIM_2, GRAD_CLIP,
)
from per_ddpg_agent import SumTree, OUNoise

# Hyperparameters aligned with paper-style SAC-PER baseline
ESAC_PER_ALPHA = 0.4
ESAC_PER_BETA_START = 0.4
ESAC_PER_BETA_FRAMES = 40000
ESAC_PER_EPSILON = 1e-5
ESAC_PER_MAX_IS_WEIGHT = 3.0

N_STEP = 3
ACTOR_DELAY = 2

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
LOG_SIG_MAX = 2
LOG_SIG_MIN = -20


# ==================================================================
# N-Step Transition Buffer
# ==================================================================
class NStepBuffer:
    """Calculates n-step discounted return to reduce bias in TD targets."""
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

    def reset(self):
        self.buffer.clear()


# ==================================================================
# Prioritized Experience Replay Buffer for ESAC
# ==================================================================
class ESACPERBuffer:
    """Prioritized experience replay buffer using O(log N) SumTree."""
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

        # Assign maximum priority to new transitions to guarantee early replay
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
                   ESAC_PER_BETA_START + self._frame *
                   (1.0 - ESAC_PER_BETA_START) / ESAC_PER_BETA_FRAMES)

        indices, tree_indices, priorities = [], [], []
        segment = self.tree.total_priority / batch_size

        for i in range(batch_size):
            value = np.random.uniform(segment * i, segment * (i + 1))
            tree_idx, priority, data_idx = self.tree.get_leaf(value)
            data_idx = np.clip(data_idx, 0, self.size - 1)
            indices.append(data_idx)
            tree_indices.append(tree_idx)
            priorities.append(priority)

        priorities = np.array(priorities, dtype=np.float32) + ESAC_PER_EPSILON
        sampling_probs = priorities / (self.tree.total_priority + ESAC_PER_EPSILON)
        is_weights = np.power(self.size * sampling_probs, -beta)
        is_weights /= is_weights.max()
        is_weights = np.clip(is_weights, 0.0, ESAC_PER_MAX_IS_WEIGHT)

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
        """Update segment tree leaf priorities based on absolute TD errors."""
        for tree_idx, td_err in zip(tree_indices, td_errors):
            priority = (abs(td_err) + ESAC_PER_EPSILON) ** ESAC_PER_ALPHA
            self.tree.update(tree_idx, priority)

    def ready(self, batch_size):
        return self.size >= batch_size


# ==================================================================
# Soft Actor-Critic Networks (Standard MLP Topology)
# ==================================================================
class ESACActor(nn.Module):
    """Spatio-temporal MLP policy network with Xavier Uniform init."""
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
        x_t = normal.rsample()  # Reparameterization trick
        action = torch.tanh(x_t) * ACTION_BOUND[1]
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(1 - (action / ACTION_BOUND[1]).pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=1, keepdim=True)
        return action, log_prob, mean

    def deterministic(self, state):
        mean, _ = self.forward(state)
        return torch.tanh(mean) * ACTION_BOUND[1]


class TwinCritic(nn.Module):
    """Standard double Q-network structure for min-Q target estimation."""
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


# ==================================================================
# Enhanced Soft Actor-Critic (ESAC-PER) Agent
# ==================================================================
class ESACPERAgent:
    """
    Enhanced Soft Actor-Critic with Prioritized Experience Replay.
    Faithful baseline model adapted from the TMC 2025 JOUR paper.
    """
    def __init__(self, s_dim=STATE_DIM, a_dim=ACTION_DIM):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.learn_step = 0

        # Actor-Critic Networks
        self.actor = ESACActor(s_dim, a_dim).to(DEVICE)
        self.critic = TwinCritic(s_dim, a_dim).to(DEVICE)
        self.critic_target = TwinCritic(s_dim, a_dim).to(DEVICE)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers (matching configuration learning rates)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=LR_ACTOR)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=LR_CRITIC)

        # Dynamic Temperature Adjustment (Automatic Entropy Regularization Tuning)
        self.target_entropy = -float(a_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=LR_ACTOR)

        # Prioritized Experience Replay and N-Step Return components
        self.buffer = ESACPERBuffer(MEMORY_CAPACITY, s_dim, a_dim)
        self.n_step_buffer = NStepBuffer(n_step=N_STEP, gamma=GAMMA)
        self.n_step_gamma = GAMMA ** N_STEP

        # Compatibility placeholder
        self.ou_noise = OUNoise(a_dim)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, state, noise_std=0.0, **kw):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            if noise_std == 0.0:
                # Deterministic action selection (Evaluation)
                action = self.actor.deterministic(state_tensor)
            else:
                # Stochastic action sampling (Training)
                action, _, _ = self.actor.sample(state_tensor)
        return action.cpu().numpy()[0]

    def store(self, s, a, r, s_, done):
        nstep_transition = self.n_step_buffer.store((s, a, r, s_, done))
        if nstep_transition is not None:
            self.buffer.store(*nstep_transition)

        if done:
            remaining = self.n_step_buffer.flush()
            for t in remaining:
                self.buffer.store(*t)
            self.n_step_buffer.reset()

    def learn(self):
        if not self.buffer.ready(BATCH_SIZE):
            return {}

        self.learn_step += 1

        # PER Sampling with Importance Sampling weights
        states, actions, rewards, next_states, dones, is_weights, tree_indices = \
            self.buffer.sample(BATCH_SIZE)

        # ---- Twin Critic Optimization ----
        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_states)
            target_q1, target_q2 = self.critic_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_prob
            y = rewards + self.n_step_gamma * (1 - dones) * target_q

        current_q1, current_q2 = self.critic(states, actions)

        # Calculate TD-error for priority updates
        td_error1 = (current_q1 - y).detach().cpu().numpy().flatten()
        td_error2 = (current_q2 - y).detach().cpu().numpy().flatten()
        td_errors = (np.abs(td_error1) + np.abs(td_error2)) / 2.0

        # Huber Loss (smooth L1) combined with IS weights
        elementwise_loss = (F.smooth_l1_loss(current_q1, y, reduction='none') +
                            F.smooth_l1_loss(current_q2, y, reduction='none'))
        critic_loss = (is_weights * elementwise_loss).mean()

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=GRAD_CLIP)
        self.critic_optimizer.step()

        # Update transition priorities in segment tree
        self.buffer.update_priorities(tree_indices, td_errors)

        # ---- Actor Policy Optimization (Delayed Update) ----
        actor_loss_val = 0.0
        if self.learn_step % ACTOR_DELAY == 0:
            new_actions, log_prob, _ = self.actor.sample(states)
            q1_new, q2_new = self.critic(states, new_actions)
            q_new = torch.min(q1_new, q2_new)

            # Actor loss scaled by Importance Sampling weights
            actor_loss = (is_weights.detach() *
                          (self.alpha.detach() * log_prob - q_new)).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=GRAD_CLIP)
            self.actor_optimizer.step()
            actor_loss_val = actor_loss.item()

            # Dynamic Temperature Tuning
            alpha_loss = -(self.log_alpha *
                           (log_prob + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # Soft updates of Twin Critic targets
            for tp, sp in zip(self.critic_target.parameters(),
                              self.critic.parameters()):
                tp.data.copy_(TAU_SOFT * sp.data + (1 - TAU_SOFT) * tp.data)

        # Return diagnostics
        return {
            'critic_loss': critic_loss.item(),
            'avg_q': (current_q1.mean().item() + current_q2.mean().item()) / 2,
            'avg_td_error': np.mean(td_errors),
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
