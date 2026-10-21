"""
Joint Positioning and Task Offloading PPO Baseline
===================================================
Reference Paper:
  "Joint Positioning and Computation Offloading in Multi-UAV MEC for Low Latency
   Applications: A Proximal Policy Optimization Approach"
   IEEE Transactions on Mobile Computing (TMC), Vol. 24, No. 10, October 2025.

Core Algorithmic Features:
  1. Joint Continuous Decision Making: Maps high-dimensional spatio-temporal states
     to joint continuous actions (UAV flight direction/speed + user offloading ratios).
  2. On-Policy Clipped Surrogate Objective: Uses PPO clipping to ensure stable policy
     updates in highly non-stationary multi-UAV environments.
  3. Generalized Advantage Estimation (GAE): Computes advantages per episode trajectory
     without cross-episode leakage, reducing variance in sparse-reward environments.
  4. Separated Actor-Critic Networks: Orthogonal initialization + LayerNorm to resolve
     gradient conflicts between value fitting and policy gradient search.
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

# Device Configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Paper-Aligned Hyperparameters
TMC_PPO_LR = 3e-4             # Learning rate (PPO standard value)
TMC_PPO_CLIP = 0.2            # Clip ratio epsilon
TMC_PPO_EPOCHS = 10           # Number of optimization epochs per rollout
TMC_GAE_LAMBDA = 0.95         # GAE parameter lambda
TMC_ENTROPY_COEFF = 0.01      # Entropy coefficient to encourage exploration in 36D action space
TMC_VALUE_COEFF = 0.5         # Value function loss coefficient
TMC_MINI_BATCH_SIZE = 128     # Mini-batch size for training stability
TMC_MAX_GRAD_NORM = 0.5       # Gradient norm clipping
TMC_ROLLOUT_EPISODES = 10     # Number of episodes gathered per learning rollout (~400 steps)


class TMCPPOActorCritic(nn.Module):
    """
    Separated Actor-Critic Network.
    Avoids gradient sharing between policy and value function to prevent policy degradation.
    """
    def __init__(self, s_dim, a_dim):
        super().__init__()
        # Actor Network (Policy Head)
        self.actor_fc1 = nn.Linear(s_dim, 400)
        self.actor_ln1 = nn.LayerNorm(400)
        self.actor_fc2 = nn.Linear(400, 300)
        self.actor_ln2 = nn.LayerNorm(300)
        self.mean_head = nn.Linear(300, a_dim)
        
        # Learnable log standard deviation for continuous exploration
        self.log_std = nn.Parameter(torch.full((a_dim,), -0.5))

        # Critic Network (Value Head)
        self.critic_fc1 = nn.Linear(s_dim, 400)
        self.critic_ln1 = nn.LayerNorm(400)
        self.critic_fc2 = nn.Linear(400, 300)
        self.critic_ln2 = nn.LayerNorm(300)
        self.value_head = nn.Linear(300, 1)

        self._init_weights()

    def _init_weights(self):
        """Orthogonal Initialization is critical for PPO convergence stability."""
        for layer in [self.actor_fc1, self.actor_fc2,
                      self.critic_fc1, self.critic_fc2]:
            nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
            nn.init.zeros_(layer.bias)
            
        # Actor output layer initialized with small weights for smooth early policy search
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
        
        # Critic output layer initialized with standard gain
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, state):
        # Policy Forward Pass
        ax = torch.relu(self.actor_ln1(self.actor_fc1(state)))
        ax = torch.relu(self.actor_ln2(self.actor_fc2(ax)))
        mean = torch.tanh(self.mean_head(ax)) * ACTION_BOUND[1]
        
        # Value Forward Pass
        cx = torch.relu(self.critic_ln1(self.critic_fc1(state)))
        cx = torch.relu(self.critic_ln2(self.critic_fc2(cx)))
        value = self.value_head(cx)
        return mean, value

    def get_dist(self, state):
        mean, value = self.forward(state)
        # Clamp standard deviation to prevent numerical instability
        std = self.log_std.exp().clamp(min=0.01, max=1.0)
        std = std.expand_as(mean)
        dist = Normal(mean, std)
        return dist, value

    def get_value(self, state):
        cx = torch.relu(self.critic_ln1(self.critic_fc1(state)))
        cx = torch.relu(self.critic_ln2(self.critic_fc2(cx)))
        return self.value_head(cx)


class TMCEpisodeBuffer:
    """
    Episode-based trajectory buffer.
    Ensures correct GAE calculation isolated within episode boundaries.
    """
    def __init__(self):
        self.episodes = []
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

        if done:
            self._finish_episode()

    def finish_episode_if_needed(self):
        """Flushes any remaining in-progress trajectory when learn() is triggered."""
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

    def ready(self, batch_size=0):
        """Rollout is complete once target number of episodes is gathered."""
        return len(self.episodes) >= TMC_ROLLOUT_EPISODES

    def clear(self):
        self.episodes.clear()
        self._total_steps = 0

    def __len__(self):
        return self._total_steps


class TMCPPOAgent:
    """
    PPO Agent implementing the core decision-making of the TMC 2025 Paper.
    Optimizes Joint UAV Positioning and User Offloading decisions.
    """
    def __init__(self, s_dim=STATE_DIM, a_dim=ACTION_DIM):
        self.s_dim = s_dim
        self.a_dim = a_dim

        self.ac = TMCPPOActorCritic(s_dim, a_dim).to(DEVICE)
        self.optimizer = optim.Adam(self.ac.parameters(), lr=TMC_PPO_LR, eps=1e-5)
        
        # Policy updates tracking for learning rate annealing
        self._total_updates = 0
        self._max_updates = 10000 // TMC_ROLLOUT_EPISODES * TMC_PPO_EPOCHS

        self.episode_buffer = TMCEpisodeBuffer()
        self._last_log_prob = None
        self._last_value = None
        self._ep_count = 0

        # Uniform API Compatibility with train_unified.py
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
            self._last_value, done
        )
        if done:
            self._ep_count += 1

    def learn(self):
        self.episode_buffer.finish_episode_if_needed()

        if not self.episode_buffer.ready():
            return None

        all_states, all_actions, all_old_log_probs = [], [], []
        all_advantages, all_returns = [], []

        # Process each collected trajectory independently to calculate correct GAE
        for ep_data in self.episode_buffer.episodes:
            states = ep_data['states']
            rewards = ep_data['rewards']
            values = ep_data['values']
            dones = ep_data['dones']

            # Value bootstrapping at the terminal step
            last_done = dones[-1] if dones else True
            if last_done:
                next_value = 0.0
            else:
                last_state = torch.FloatTensor(states[-1]).unsqueeze(0).to(DEVICE)
                with torch.no_grad():
                    next_value = self.ac.get_value(last_state).item()

            advantages = []
            gae = 0
            nv = next_value
            for t in reversed(range(len(rewards))):
                delta = rewards[t] + GAMMA * nv * (1 - dones[t]) - values[t]
                gae = delta + GAMMA * TMC_GAE_LAMBDA * (1 - dones[t]) * gae
                advantages.insert(0, gae)
                nv = values[t]

            advantages = np.array(advantages, dtype=np.float32)
            returns = advantages + np.array(values, dtype=np.float32)

            all_states.extend(states)
            all_actions.extend(ep_data['actions'])
            all_old_log_probs.extend(ep_data['log_probs'])
            all_advantages.append(advantages)
            all_returns.append(returns)

        # Concatenate and transfer rollout trajectories to device
        states_t = torch.FloatTensor(np.array(all_states)).to(DEVICE)
        actions_t = torch.FloatTensor(np.array(all_actions)).to(DEVICE)
        old_log_probs_t = torch.FloatTensor(all_old_log_probs).to(DEVICE)
        advantages_t = torch.FloatTensor(np.concatenate(all_advantages)).to(DEVICE)
        returns_t = torch.FloatTensor(np.concatenate(all_returns)).to(DEVICE)

        # Normalize advantages globally across rollout to stabilize training updates
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n_samples = len(all_states)
        total_actor_loss = 0.0
        total_value_loss = 0.0

        # Linear Learning Rate Annealing
        self._total_updates += 1
        lr_now = TMC_PPO_LR * max(0.1, 1.0 - self._total_updates / max(1, self._max_updates))
        for pg in self.optimizer.param_groups:
            pg['lr'] = lr_now

        # PPO Multi-Epoch Mini-Batch Policy & Value Updates
        for epoch in range(TMC_PPO_EPOCHS):
            perm = torch.randperm(n_samples)

            for start in range(0, n_samples, TMC_MINI_BATCH_SIZE):
                end = min(start + TMC_MINI_BATCH_SIZE, n_samples)
                idx = perm[start:end]

                mb_states = states_t[idx]
                mb_actions = actions_t[idx]
                mb_old_log_probs = old_log_probs_t[idx]
                mb_advantages = advantages_t[idx]
                mb_returns = returns_t[idx]

                dist, current_values = self.ac.get_dist(mb_states)
                new_log_probs = dist.log_prob(mb_actions).sum(dim=1)
                entropy = dist.entropy().sum(dim=1).mean()

                # Clipped Policy surrogate objective
                ratio = (new_log_probs - mb_old_log_probs).exp()
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - TMC_PPO_CLIP, 1 + TMC_PPO_CLIP) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic Mean Squared Error value loss
                value_pred = current_values.squeeze(-1)
                value_loss = F.mse_loss(value_pred, mb_returns.squeeze(-1))

                # Combined Multi-Objective Loss
                loss = (actor_loss
                        + TMC_VALUE_COEFF * value_loss
                        - TMC_ENTROPY_COEFF * entropy)

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.ac.parameters(), max_norm=TMC_MAX_GRAD_NORM)
                self.optimizer.step()

                total_actor_loss += actor_loss.item()
                total_value_loss += value_loss.item()

        # Clear buffer for next rollout gathering phase
        self.episode_buffer.clear()
        self._ep_count = 0

        n_updates = max(1, TMC_PPO_EPOCHS * (n_samples // TMC_MINI_BATCH_SIZE))
        return {
            'critic_loss': total_value_loss / n_updates,
            'avg_q': total_value_loss / n_updates,
            'avg_td_error': total_actor_loss / n_updates,
        }

    def save(self, path):
        torch.save({'ac': self.ac.state_dict()}, path)

    def load(self, path):
        checkpoint = torch.load(path, map_location=DEVICE)
        self.ac.load_state_dict(checkpoint['ac'])
