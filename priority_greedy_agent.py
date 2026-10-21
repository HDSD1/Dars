"""
Priority-aware greedy baseline for multi-UAV MEC with WCT charging.

This baseline adapts the task-priority idea from:
  Hao et al., "Joint Task Offloading, Resource Allocation, and Trajectory
  Design for Multi-UAV Cooperative Edge Computing With Task Priority",
  IEEE Transactions on Mobile Computing, 2024.

It is not a learning method. UAVs serve the highest-priority active users when
energy is sufficient and return to the nearest WCT under low battery.
"""

import math
import numpy as np

from config import (
    K, M, STATE_DIM, ACTION_DIM, ACTION_BOUND,
    GROUND_LENGTH, GROUND_WIDTH, E_BATTERY_INIT,
    V_MAX, T_FLY,
)


class PriorityGreedyAgent:
    """Task-priority heuristic with WCT-aware emergency charging."""

    def __init__(self, s_dim=STATE_DIM, a_dim=ACTION_DIM,
                 battery_threshold=0.35, top_users=8):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.battery_threshold = battery_threshold
        self.top_users = top_users
        self.wct_positions = np.array([
            [GROUND_LENGTH / 4, GROUND_WIDTH / 4],
            [3 * GROUND_LENGTH / 4, GROUND_WIDTH / 4],
            [GROUND_LENGTH / 4, 3 * GROUND_WIDTH / 4],
            [3 * GROUND_LENGTH / 4, 3 * GROUND_WIDTH / 4],
        ], dtype=np.float64)
        self.ou_noise = type('', (), {'reset': lambda s: None})()
        self.buffer = type('', (), {'ready': lambda s, b: False})()

    def _parse_state(self, state):
        idx = 0
        battery = state[idx: idx + K]
        idx += K
        uav_pos = state[idx: idx + 2 * K].reshape(K, 2)
        idx += 2 * K
        ue_pos = state[idx: idx + 2 * M].reshape(M, 2)
        idx += 2 * M
        task = state[idx: idx + M]
        idx += M
        active = state[idx: idx + M]
        return battery, uav_pos, ue_pos, task, active

    def _nearest_wct(self, uav_pos_k):
        dists = np.linalg.norm(self.wct_positions - uav_pos_k, axis=1)
        return self.wct_positions[int(np.argmin(dists))]

    def _priority_target(self, uav_pos_k, ue_pos, task, active):
        active_idx = np.where(active > 0.5)[0]
        if len(active_idx) == 0:
            return uav_pos_k, []

        # Priority combines task demand and proximity. Larger tasks are more
        # urgent, while far users are discounted because short-term service
        # quality degrades quickly with distance.
        dists = np.linalg.norm(ue_pos[active_idx] - uav_pos_k, axis=1)
        demand = task[active_idx]
        demand = demand / max(float(np.max(demand)), 1e-9)
        distance_score = 1.0 / (1.0 + (dists / 120.0) ** 2)
        scores = 0.65 * demand + 0.35 * distance_score

        order = np.argsort(-scores)[:min(self.top_users, len(active_idx))]
        selected = active_idx[order]
        weights = scores[order] * distance_score[order] + 1e-6
        weights = weights / float(np.sum(weights))
        target = np.sum(ue_pos[selected] * weights[:, None], axis=0)
        return target, selected

    @staticmethod
    def _to_movement(uav_pos_k, target):
        dx = float(target[0] - uav_pos_k[0])
        dy = float(target[1] - uav_pos_k[1])
        dist = math.sqrt(dx * dx + dy * dy)
        if dist < 5.0:
            return np.array([0.0, 0.0, -1.0], dtype=np.float64)
        # Use only one-slot reachable distance as the speed scale. The old
        # 50 m scale overestimated speed because the environment only flies
        # V_MAX*T_FLY meters per slot.
        speed = min(1.0, dist / max(V_MAX * T_FLY, 1e-6)) * 2.0 - 1.0
        return np.array([dx / dist, dy / dist, speed], dtype=np.float64)

    def select_action(self, state, noise_std=0.0, **kw):
        battery, uav_pos, ue_pos, task, active = self._parse_state(state)
        action = np.zeros(self.a_dim, dtype=np.float64)

        for k in range(K):
            battery_ratio = battery[k] / E_BATTERY_INIT if E_BATTERY_INIT > 0 else 1.0
            nearest_wct = self._nearest_wct(uav_pos[k])
            dist_to_wct = float(np.linalg.norm(nearest_wct - uav_pos[k]))
            # Return to WCT before the battery becomes critical. This makes the
            # priority heuristic viable under the hard episode-termination
            # energy constraint used in this project.
            should_charge = (
                battery_ratio <= self.battery_threshold or
                (battery_ratio <= 0.60 and dist_to_wct > 120.0)
            )
            if should_charge:
                target = nearest_wct
                selected = []
            else:
                target, selected = self._priority_target(
                    uav_pos[k], ue_pos, task, active)

            action[3 * k: 3 * k + 3] = self._to_movement(uav_pos[k], target)

        return np.clip(action, ACTION_BOUND[0], ACTION_BOUND[1])

    def store(self, *args):
        pass

    def learn(self):
        return {}

    def save(self, path):
        pass

    def load(self, path):
        pass
