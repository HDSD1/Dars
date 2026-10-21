"""
KNN-DDPG+CVX baseline adapted to the UAV-MEC-WCT scenario.

Reference baseline idea:
  "Joint Computation Offloading and Trajectory Optimization for Edge
  Computing UAV: A KNN-DDPG Algorithm", Drones, 2024.

This implementation keeps the same DDPG learner used in this project and
adds a K-nearest-neighbor action prior:
  1. When UAV battery is sufficient, steer each UAV toward a task-weighted
     centroid of its K nearest active users.
  2. When UAV battery is low, steer it toward the nearest WCT.
  3. Blend the KNN prior with the learned DDPG action, so the baseline remains
     trainable while reflecting the KNN-assisted decision mechanism.
"""

import math
import numpy as np
import torch
import torch.nn as nn

from config import (
    K, M, ACTION_DIM, ACTION_BOUND,
    GROUND_LENGTH, GROUND_WIDTH,
    BATCH_SIZE, GAMMA,
)
from ddpg_agent import DDPGAgent, DEVICE


class KNNDDPGAgent(DDPGAgent):
    """DDPG with a KNN task-demand prior for UAV movement."""

    def __init__(self, s_dim=None, a_dim=ACTION_DIM,
                 knn_k=5, blend=0.30, min_blend=0.15,
                 blend_decay=0.9999, battery_threshold=0.42):
        from config import STATE_DIM
        super().__init__(STATE_DIM if s_dim is None else s_dim, a_dim)
        self.knn_k = knn_k
        self.blend = blend
        self.min_blend = min_blend
        self.blend_decay = blend_decay
        self.battery_threshold = battery_threshold
        self._learn_step = 0
        self.wct_positions_norm = np.array([
            [0.25, 0.25],
            [0.75, 0.25],
            [0.25, 0.75],
            [0.75, 0.75],
        ], dtype=np.float64)

    def _parse_normalized_state(self, state):
        """Parse the normalized state used by train_unified.py."""
        idx = 0
        battery_ratio = state[idx: idx + K]
        idx += K
        uav_pos = state[idx: idx + 2 * K].reshape(K, 2)
        idx += 2 * K
        ue_pos = state[idx: idx + 2 * M].reshape(M, 2)
        idx += 2 * M
        task_norm = state[idx: idx + M]
        idx += M
        active = state[idx: idx + M]
        return battery_ratio, uav_pos, ue_pos, task_norm, active

    def _nearest_wct(self, uav_pos_k):
        dists = np.linalg.norm(self.wct_positions_norm - uav_pos_k, axis=1)
        idx = int(np.argmin(dists))
        return self.wct_positions_norm[idx], float(dists[idx])

    def _knn_task_centroid(self, uav_pos_k, ue_pos, task_norm, active):
        active_idx = np.where(active > 0.5)[0]
        if len(active_idx) == 0:
            return uav_pos_k, []

        active_pos = ue_pos[active_idx]
        dists = np.linalg.norm(active_pos - uav_pos_k, axis=1)
        order = np.argsort(dists)[:min(self.knn_k, len(active_idx))]
        selected = active_idx[order]

        # Task-weighted KNN centroid. The small floor keeps pure active users
        # relevant even after state normalization compresses task sizes.
        selected_dists = dists[order]
        distance_score = 1.0 / (selected_dists + 0.05)
        weights = (task_norm[selected] + 1e-3) * distance_score
        weights = weights / max(float(np.sum(weights)), 1e-9)
        target = np.sum(ue_pos[selected] * weights[:, None], axis=0)
        return target, selected

    @staticmethod
    def _movement_action(uav_pos_k, target):
        dx = float(target[0] - uav_pos_k[0])
        dy = float(target[1] - uav_pos_k[1])
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 1e-4:
            return np.array([0.0, 0.0, -1.0], dtype=np.float64)

        # Normalized map distance is in [0, sqrt(2)]. A distance of 0.08
        # corresponds to roughly 40 m in the 500 m default map.
        speed_raw = min(1.0, dist / 0.08) * 2.0 - 1.0
        return np.array([dx / dist, dy / dist, speed_raw], dtype=np.float64)

    def _knn_prior_action(self, state):
        battery_ratio, uav_pos, ue_pos, task_norm, active = \
            self._parse_normalized_state(state)

        prior = np.zeros(self.a_dim, dtype=np.float64)
        for k in range(K):
            nearest_wct, dist_to_wct = self._nearest_wct(uav_pos[k])
            if (battery_ratio[k] <= self.battery_threshold or
                    (battery_ratio[k] <= 0.60 and dist_to_wct > 0.24)):
                target = nearest_wct
                selected = []
            else:
                target, _ = self._knn_task_centroid(
                    uav_pos[k], ue_pos, task_norm, active)

            prior[3 * k: 3 * k + 3] = self._movement_action(
                uav_pos[k], target)

        return np.clip(prior, ACTION_BOUND[0], ACTION_BOUND[1])

    def select_action(self, state, noise_std=0.0, use_ou=False):
        ddpg_action = super().select_action(
            state, noise_std=noise_std, use_ou=use_ou)
        prior = self._knn_prior_action(state)

        action = (1.0 - self.blend) * ddpg_action + self.blend * prior
        return np.clip(action, ACTION_BOUND[0], ACTION_BOUND[1])

    def _knn_prior_action_batch(self, states_np):
        """Batch compute prior actions in numpy."""
        batch_size = states_np.shape[0]
        priors = np.zeros((batch_size, self.a_dim), dtype=np.float32)
        for i in range(batch_size):
            priors[i] = self._knn_prior_action(states_np[i])
        return priors

    def learn(self):
        if not self.buffer.ready(BATCH_SIZE):
            return None
        self._learn_step += 1

        states, actions, rewards, next_states, dones = \
            self.buffer.sample(BATCH_SIZE)

        # Convert states and next_states to numpy to batch compute prior actions
        states_np = states.cpu().numpy()
        next_states_np = next_states.cpu().numpy()

        priors_np = self._knn_prior_action_batch(states_np)
        next_priors_np = self._knn_prior_action_batch(next_states_np)

        priors = torch.FloatTensor(priors_np).to(DEVICE)
        next_priors = torch.FloatTensor(next_priors_np).to(DEVICE)

        blend = self.blend

        # --- 更新 Critic ---
        with torch.no_grad():
            next_actions_ddpg = self.actor_target(next_states)
            # Apply the blending function to target actions to match the experience in the buffer
            next_actions = (1.0 - blend) * next_actions_ddpg + blend * next_priors
            next_actions = torch.clamp(next_actions, ACTION_BOUND[0], ACTION_BOUND[1])
            
            target_q = self.critic_target(next_states, next_actions)
            y = rewards + GAMMA * (1 - dones) * target_q

        current_q = self.critic(states, actions)
        critic_loss = nn.SmoothL1Loss()(current_q, y)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # --- 更新 Actor ---
        pred_actions_ddpg = self.actor(states)
        # Apply the blending function to predicted actions so the gradient flows correctly
        pred_actions = (1.0 - blend) * pred_actions_ddpg + blend * priors
        pred_actions = torch.clamp(pred_actions, ACTION_BOUND[0], ACTION_BOUND[1])
        
        prior_loss = nn.SmoothL1Loss()(pred_actions_ddpg, priors)
        actor_loss = -self.critic(states, pred_actions).mean() + 0.03 * prior_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # --- 软更新 ---
        self._soft_update(self.actor_target, self.actor)
        self._soft_update(self.critic_target, self.critic)
        self.blend = max(self.min_blend, self.blend * self.blend_decay)

        return {
            'critic_loss': critic_loss.item(),
            'avg_q': current_q.mean().item(),
            'avg_td_error': (y - current_q).abs().mean().item(),
            'knn_blend': self.blend,
        }
