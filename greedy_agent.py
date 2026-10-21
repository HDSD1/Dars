"""
贪心启发式 Baseline
====================
规则决策, 不涉及任何学习:
  1. 电量 > 30%: 飞向最近的 active 用户密集区中心
  2. 电量 ≤ 30%: 飞向最近的 WCT 充电
  3. 卸载比例和资源分配由环境内层优化器求解
"""
import numpy as np
import math
from config import (
    K, M, H, N_WCT, STATE_DIM, ACTION_DIM, ACTION_BOUND,
    GROUND_LENGTH, GROUND_WIDTH, E_BATTERY_INIT,
)


class GreedyAgent:
    """贪心启发式: 飞向用户群中心, 低电量时充电."""

    def __init__(self, s_dim=STATE_DIM, a_dim=ACTION_DIM,
                 battery_threshold=0.3, offload_ratio=1.0):
        self.s_dim = s_dim
        self.a_dim = a_dim
        self.battery_threshold = battery_threshold
        self.offload_ratio = offload_ratio
        # WCT 位置 (四象限中心)
        self.wct_positions = np.array([
            [GROUND_LENGTH / 4, GROUND_WIDTH / 4],
            [3 * GROUND_LENGTH / 4, GROUND_WIDTH / 4],
            [GROUND_LENGTH / 4, 3 * GROUND_WIDTH / 4],
            [3 * GROUND_LENGTH / 4, 3 * GROUND_WIDTH / 4],
        ])
        # 接口兼容
        self.ou_noise = type('', (), {'reset': lambda s: None})()
        self.buffer = type('', (), {'ready': lambda s, b: False})()

    def _parse_state(self, state):
        """从 state 向量中提取 UAV 位置、电量、用户位置、active_mask."""
        idx = 0
        battery = state[idx: idx + K]; idx += K
        uav_pos = state[idx: idx + 2*K].reshape(K, 2); idx += 2*K
        ue_pos = state[idx: idx + 2*M].reshape(M, 2); idx += 2*M
        task_list = state[idx: idx + M]; idx += M
        active_mask = state[idx: idx + M]; idx += M
        # 跳过 wct_info (3K) 和 time_step (1)
        return battery, uav_pos, ue_pos, active_mask

    def _find_user_cluster_center(self, uav_pos_k, ue_pos, active_mask):
        """找到距离 UAV k 最近的 active 用户群的重心."""
        active_ues = ue_pos[active_mask > 0.5]
        if len(active_ues) == 0:
            return uav_pos_k  # 没有 active 用户, 原地不动
        # 直接飞向最近的 active 用户
        dists = np.sqrt(np.sum((active_ues - uav_pos_k)**2, axis=1))
        nearest_5 = active_ues[np.argsort(dists)[:5]]
        return np.mean(nearest_5, axis=0)

    def _find_nearest_wct(self, uav_pos_k):
        dists = np.sqrt(np.sum((self.wct_positions - uav_pos_k)**2, axis=1))
        return self.wct_positions[np.argmin(dists)]

    def select_action(self, state, noise_std=0.0, **kw):
        """基于规则生成动作."""
        battery, uav_pos, ue_pos, active_mask = self._parse_state(state)
        action = np.zeros(self.a_dim)

        for k in range(K):
            batt_ratio = battery[k] / E_BATTERY_INIT if E_BATTERY_INIT > 0 else 1.0

            if batt_ratio <= self.battery_threshold:
                # 低电量 → 飞向最近 WCT
                target = self._find_nearest_wct(uav_pos[k])
            else:
                # 正常 → 飞向用户群中心
                target = self._find_user_cluster_center(
                    uav_pos[k], ue_pos, active_mask)

            dx = target[0] - uav_pos[k, 0]
            dy = target[1] - uav_pos[k, 1]
            dist = math.sqrt(dx*dx + dy*dy)

            if dist < 5.0:
                # 已经很近, 悬停
                action[3*k] = 0.0       # cos
                action[3*k+1] = 0.0     # sin
                action[3*k+2] = -1.0    # speed=0
            else:
                # 飞向目标
                action[3*k] = dx / dist       # cos(θ)
                action[3*k+1] = dy / dist     # sin(θ)
                action[3*k+2] = min(1.0, dist / 50.0) * 2 - 1  # 速度

        return np.clip(action, ACTION_BOUND[0], ACTION_BOUND[1])

    def store(self, *args): pass
    def learn(self): pass
    def save(self, path): pass
    def load(self, path): pass
