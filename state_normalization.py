"""状态归一化 V4."""
import numpy as np
from config import K, M, STATE_DIM, E_BATTERY_CAPACITY, GROUND_LENGTH, GROUND_WIDTH, TASK_MAX, N_WCT

class StateNormalization:
    def __init__(self, battery_capacity=E_BATTERY_CAPACITY):
        high = []
        high.extend([battery_capacity] * K)
        for _ in range(K):
            high.extend([GROUND_LENGTH, GROUND_WIDTH])
        for _ in range(M):
            high.extend([GROUND_LENGTH, GROUND_WIDTH])
        high.extend([TASK_MAX] * M)
        high.extend([1.0] * M)      # active_mask ∈ {0, 1}
        # WCT 距离信息: 每 UAV 3 维 (归一化距离, 方向x, 方向y)
        for _ in range(K):
            high.extend([1.0, 1.0, 1.0])  # 都在 [-1, 1] 或 [0, 1]
        high.append(1.0)

        self.high_state = np.array(high, dtype=np.float64)
        self.low_state = np.zeros(STATE_DIM, dtype=np.float64)
        assert len(self.high_state) == STATE_DIM

    def normalize(self, state):
        denom = self.high_state - self.low_state
        denom = np.where(denom == 0, 1.0, denom)
        return state / denom

    state_normal = normalize
