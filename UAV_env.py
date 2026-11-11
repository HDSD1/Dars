"""
多 UAV-MEC 计算卸载环境 (V6)
==============================
特性:
  1. UAV 初始位置随机化 (每 episode 不同, 增强泛化)
  2. 归一化加权目标: r = ω_t×Σρ_i + ω_e×η_E (延迟缩减 + 能量回收率)
  3. 电量约束: 耗尽终止 + 死亡惩罚 + 低电二次惩罚
  4. 凸优化算力分配: f_{k,i}* ∝ √(x_i·D_i·C_i) (可选)
"""

import math
import numpy as np

from config import (
    K, M, H, GROUND_LENGTH, GROUND_WIDTH, N_MAX,
    T_FLY, T_COM, TAU_SLOT, SLOT_NUM, V_MAX, R_COV,
    P0, PI, U_TIP, V0, D0, RHO, ROTOR_SOLIDITY, ROTOR_AREA,
    A_ENV, B_ENV, MU_LOS, MU_NLOS, ALPHA_PL, ALPHA0,
    BANDWIDTH, P_TX, SIGMA2,
    F_UE, F_UAV, KAPPA, S_CYCLES, V_UE,
    TASK_MIN, TASK_MAX, TASK_PROB, E_BATTERY_INIT,
    ACTION_DIM, STATE_DIM, ACTION_BOUND,
    REWARD_BATTERY_PENALTY, BATTERY_LOW_THRESHOLD,
    N_WCT, H_WCT, WCT_POSITIONS,
    ETA_WCT, THETA_WCT, MU_WCT, L_C, P_TH, P_MAX_WCT,
    N_CLUSTERS, CLUSTER_STD, V_CLUSTER,
    HOTSPOT_HOT_PROB, HOTSPOT_COLD_PROB,
)


class UAVEnv:
    """多 UAV-MEC 计算卸载环境 (V5, 支持凸优化算力分配)."""

    def __init__(self, use_cvx_alloc=False,
                 use_hotspot=True, disable_wct=False):
        """
        Args:
            use_cvx_alloc: True → 凸优化最优算力+带宽分配
                           False → 传统均分 (Baseline)
            use_hotspot: True → 分阶段热/冷簇轮转 (3阶段)
                         False → 全局统一 TASK_PROB
            disable_wct: True → 关闭充电 (消融实验用)
        """
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.action_bound = ACTION_BOUND
        self.slot_num = SLOT_NUM
        self.use_cvx_alloc = use_cvx_alloc
        self.use_hotspot = use_hotspot
        self.disable_wct = disable_wct

        self.wct_positions = self._init_wct_positions()

        self.e_battery_uav = None
        self.loc_uav = None
        self.loc_ue = None
        self.task_list = None
        self.active_mask = None  # 泊松任务到达: 哪些用户有任务
        self.current_step = 0

        # 日志
        self.trajectory_log = []
        self.ue_position_log = []
        self.association_log = []
        self.delay_log = []
        self.rho_log = []
        self.reward_log = []
        self.battery_log = []
        self.charge_log = []

    # ==================================================================
    def reset(self):
        self.e_battery_uav = np.full(K, E_BATTERY_INIT, dtype=np.float64)

        # ★ UAV 初始位置随机化 (在场地内随机撒点)
        # ★ 修改：让所有 UAV 初始集中在场地正中央区域 (250, 250) 附近
        # 这样它们距离位于四个角落 (125,125 等) 的充电站都有一定距离 (~176米)
        # 能更有力地证明它们是“主动”飞过去充电的，而不是“恰好”出生在充电站旁边
        self.loc_uav = np.column_stack([
            np.random.uniform(GROUND_LENGTH * 0.45, GROUND_LENGTH * 0.55, K),
            np.random.uniform(GROUND_WIDTH * 0.45, GROUND_WIDTH * 0.55, K),
        ])

        # ★ 四象限簇中心 (固定位置)
        self.cluster_centers = np.array([
            [GROUND_LENGTH * 0.25, GROUND_WIDTH * 0.75],  # 簇0: 左上 (北)
            [GROUND_LENGTH * 0.75, GROUND_WIDTH * 0.75],  # 簇1: 右上 (东)
            [GROUND_LENGTH * 0.75, GROUND_WIDTH * 0.25],  # 簇2: 右下 (南)
            [GROUND_LENGTH * 0.25, GROUND_WIDTH * 0.25],  # 簇3: 左下 (西)
        ], dtype=np.float64)
        self.user_cluster_ids = np.arange(M) % N_CLUSTERS

        # ★ 热/冷簇轮转方案 (3阶段)
        # 每阶段 2个热簇(0.8), 2个冷簇(0.2), 热簇顺时针旋转
        self._hotspot_phases = [
            [0, 1],  # Phase 0: 北+东 热
            [1, 2],  # Phase 1: 东+南 热
            [2, 3],  # Phase 2: 南+西 热
        ]
        self._hot_prob = HOTSPOT_HOT_PROB
        self._cold_prob = HOTSPOT_COLD_PROB

        self.loc_ue = self._init_ue_positions()

        # ★ 任务生成 (根据热点模式)
        self._generate_tasks()

        self.current_step = 0

        self.trajectory_log = []
        self.ue_position_log = []
        self.association_log = []
        self.delay_log = []
        self.rho_log = []
        self.reward_log = []
        self.battery_log = []
        self.charge_log = []

        return self._get_obs()

    # ==================================================================
    def _get_obs(self):
        # ★ WCT 距离信息: 每架 UAV 到最近 WCT 的 (归一化距离, 方向x, 方向y)
        wct_info = np.zeros(3 * K)
        for k in range(K):
            best_dist = float('inf')
            best_dx, best_dy = 0.0, 0.0
            for j in range(N_WCT):
                dx = self.wct_positions[j, 0] - self.loc_uav[k, 0]
                dy = self.wct_positions[j, 1] - self.loc_uav[k, 1]
                d = math.sqrt(dx*dx + dy*dy)
                if d < best_dist:
                    best_dist = d
                    best_dx, best_dy = dx, dy
            norm = max(best_dist, 1e-6)
            wct_info[3*k] = best_dist / GROUND_LENGTH   # 归一化距离
            wct_info[3*k+1] = best_dx / norm             # 方向 x
            wct_info[3*k+2] = best_dy / norm             # 方向 y

        state = np.concatenate([
            self.e_battery_uav,
            self.loc_uav.flatten(),
            self.loc_ue.flatten(),
            self.task_list,
            self.active_mask.astype(np.float64),
            wct_info,                                    # ★ WCT 几何线索
            [self.current_step / max(SLOT_NUM, 1)],
        ])
        assert state.shape[0] == STATE_DIM, \
            f"state dim {state.shape[0]} != {STATE_DIM}"
        return state

    # ==================================================================
    def step(self, action):
        """Execute only a phase-feasible action; replay retains the proposal.

        The deterministic shield is part of the environment transition. Its
        bounded candidate search is not an exact continuous-space projection.
        """
        proposed = np.asarray(action, dtype=np.float64).copy()
        if proposed.shape != (self.action_dim,) or not np.all(np.isfinite(proposed)):
            raise ValueError('Action must be a finite vector of action_dim entries')
        candidate = np.clip(proposed, -1.0, 1.0)
        original_check = self._step_candidate(candidate, evaluate_only=True)
        checks = 1
        selected = candidate if original_check['feasible'] else None
        if selected is None:
            # Closest feasible member of a finite set, measured in normalized
            # actor coordinates. Always recompute association/resources/charging.
            best_distance = float('inf')
            for alternative in self._safety_candidates(candidate):
                check = self._step_candidate(alternative, evaluate_only=True)
                checks += 1
                if check['feasible']:
                    distance = float(np.sum((alternative - candidate) ** 2))
                    if distance < best_distance:
                        selected = alternative.copy()
                        best_distance = distance
        if selected is None:
            result = self._energy_failure(
                'shield_search_exhausted', original_check['energy_ok'],
                original_failure_phase=original_check['phase'])
        else:
            result = self._step_candidate(selected)
        info = result[3]
        info.update({
            'proposed_action': proposed,
            'executed_action': None if selected is None else selected.copy(),
            'shield_intervened': selected is None or not np.array_equal(selected, proposed),
            'shield_checks': checks,
            'shield_original_failure': original_check['phase'],
            'shield_status': ('no_feasible_candidate_found' if selected is None
                              else 'accepted' if np.array_equal(selected, proposed)
                              else 'corrected'),
        })
        return result

    def _safety_candidates(self, action):
        """Bounded joint candidates; failure is not proof of an empty feasible set."""
        # Preserve motion and reduce service first, including local-only service.
        for scale in (0.75, 0.5, 0.25, 0.0):
            candidate = action.copy()
            candidate[3*K:] = (action[3*K:] + 1.0) * scale - 1.0
            yield candidate
        towards_wct = np.zeros(K)
        for k in range(K):
            if len(self.wct_positions):
                delta = self.wct_positions - self.loc_uav[k]
                target = delta[np.argmin(np.sum(delta ** 2, axis=1))]
                towards_wct[k] = math.atan2(target[1], target[0])
        original_angles = np.arctan2(action[1:3*K:3], action[0:3*K:3])
        # Include multiple speeds: rotary-wing power need not decrease with speed.
        for angles in (original_angles, towards_wct,
                       original_angles + math.pi / 2,
                       original_angles - math.pi / 2,
                       original_angles + math.pi):
            for speed_fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
                for scale in (1.0, 0.5, 0.0):
                    candidate = action.copy()
                    candidate[0:3*K:3] = np.cos(angles)
                    candidate[1:3*K:3] = np.sin(angles)
                    candidate[2:3*K:3] = 2.0 * speed_fraction - 1.0
                    candidate[3*K:] = (action[3*K:] + 1.0) * scale - 1.0
                    yield candidate

    def _step_candidate(self, action, evaluate_only=False):
        action = np.clip(action, -1.0, 1.0)
        # ★ cos/sin 角度表示: 消除 0/2π 不连续性
        theta_list = np.zeros(K)
        speed_list = np.zeros(K)
        for k in range(K):
            cos_k = action[3 * k]       # [-1, 1]
            sin_k = action[3 * k + 1]   # [-1, 1]
            v_raw = action[3 * k + 2]   # [-1, 1]
            theta_list[k] = math.atan2(sin_k, cos_k)     # (-π, π]
            speed_list[k] = (v_raw + 1.0) / 2.0 * V_MAX  # [0, V_MAX]
        offload_ratios = (action[3 * K: 3 * K + M] + 1.0) / 2.0  # [0, 1]

        # ---- 1. 飞行 ----
        dist_list = speed_list * T_FLY
        new_loc_uav = np.copy(self.loc_uav)
        fly_energy = np.zeros(K)

        for k in range(K):
            dx = dist_list[k] * math.cos(theta_list[k])
            dy = dist_list[k] * math.sin(theta_list[k])
            new_x = self.loc_uav[k, 0] + dx
            new_y = self.loc_uav[k, 1] + dy
            new_x, new_y = self._bounce(new_x, new_y)
            new_loc_uav[k] = [new_x, new_y]

            # Ideal mirror reflection changes heading, not speed or duration.
            # Use this candidate's speed (including any shield correction),
            # not the displacement between the initial and reflected positions.
            fly_energy[k] = self._fly_power(speed_list[k]) * T_FLY

        # Flight must be funded before any hovering-phase harvesting.
        battery_start = self.e_battery_uav.copy()
        flight_ok = battery_start >= fly_energy
        if not np.all(flight_ok):
            if evaluate_only:
                return {'feasible': False, 'phase': 'flight', 'energy_ok': flight_ok}
            raise RuntimeError('Shield attempted to execute an infeasible flight')
        battery_postflight = battery_start - fly_energy

        # ---- 2. 距离关联 ----
        association = self._distance_association(new_loc_uav)
        N_k = np.zeros(K, dtype=int)
        for i in range(M):
            # 仅统计“活跃且实际发生卸载(x_i>0)”用户，避免幽灵带宽占用
            if self.active_mask[i] and association[i] >= 0 and offload_ratios[i] > 1e-10:
                N_k[association[i]] += 1

        # ---- 3. 凸优化资源分配 ----
        # 给定关联和卸载比例, 求每个用户的最优/均分算力
        f_alloc = self._compute_allocation(
            association, N_k, offload_ratios)
        # ★ 带宽也用凸优化最优分配 (KKT 闭式解)
        b_alloc = self._compute_bandwidth_allocation(
            association, N_k, offload_ratios, new_loc_uav)

        # ---- 4. 延迟计算 ----
        delays = np.zeros(M)
        rho_list = np.zeros(M)
        comp_energy = np.zeros(K)
        edge_delays = np.zeros(M)

        for i in range(M):
            # ★ 无任务的用户跳过
            if not self.active_mask[i]:
                delays[i] = 0.0
                rho_list[i] = 0.0
                continue

            D_i = self.task_list[i]
            C_i = D_i * S_CYCLES
            x_i = offload_ratios[i]
            k = association[i]
            T_baseline = C_i / F_UE
            T_local = (1.0 - x_i) * C_i / F_UE

            if k < 0 or x_i <= 1e-10:
                T_i = T_baseline if k < 0 else T_local
            else:
                b_ik = b_alloc[i]  # ★ 最优带宽 (CVX) or 均分
                f_ik = f_alloc[i]  # ★ 最优算力 (CVX) or 均分
                g_ik = self._channel_gain(new_loc_uav[k], self.loc_ue[i])
                snr = P_TX * g_ik / SIGMA2
                if snr <= 0:
                    snr = 1e-10
                R_ik = b_ik * math.log2(1.0 + snr)

                T_trans = x_i * D_i / max(R_ik, 1e-10)
                T_edge_comp = x_i * C_i / max(f_ik, 1e-10)
                T_edge = T_trans + T_edge_comp
                edge_delays[i] = T_edge
                T_i = max(T_local, T_edge)
                comp_energy[k] += KAPPA * (f_ik ** 3) * T_edge_comp

            delays[i] = T_i
            rho_list[i] = 1.0 - T_i / T_baseline if T_baseline > 0 else 0.0

        # ---- 4. 能耗 ----
        hover_energy = np.full(K, self._hover_power() * T_COM)
        energy_out = fly_energy + hover_energy + comp_energy

        # ---- 5. WCT 充电 ----
        charge_power = self._compute_charging(new_loc_uav)
        energy_in = charge_power * T_COM

        # Phase-level service energy accounting: computation energy is represented
        # by its average power comp_energy / T_COM during this phase. This is
        # not a task-event-level execution model. Harvesting can fund service,
        # but can never fund the preceding flight (checked above).
        service_end = battery_postflight + energy_in - hover_energy - comp_energy
        energy_ok = service_end >= 0.0
        if not np.all(energy_ok):
            if evaluate_only:
                return {'feasible': False, 'phase': 'service', 'energy_ok': energy_ok}
            raise RuntimeError('Shield attempted to execute infeasible service')
        # Accepted UAV work must finish before the service phase ends.
        # Local completion latency remains an analytical metric, without a
        # local deadline or a cross-slot local queue in this model.
        deadline_ok = edge_delays <= T_COM
        if not np.all(deadline_ok):
            if evaluate_only:
                return {'feasible': False, 'phase': 'edge_deadline',
                        'energy_ok': energy_ok, 'deadline_ok': deadline_ok}
            raise RuntimeError('Shield attempted to execute overdue edge work')
        if evaluate_only:
            return {'feasible': True, 'phase': None, 'energy_ok': energy_ok,
                    'battery_postflight': battery_postflight,
                    'battery_end': np.minimum(E_BATTERY_INIT, service_end)}
        # Constant phase-average net power: endpoints suffice in this model.
        self.e_battery_uav = np.minimum(E_BATTERY_INIT, service_end)

        self.loc_uav = new_loc_uav

        # ---- 7. 奖励 (归一化延迟缩减目标) ----
        # r(t) = M × mean(ρ_active)  消除活跃用户数波动, 乘 M 保持 Q 值量级
        # 充电只是辅助手段: UAV 不充电 → 早死 → 少赚 ρ, 充电动机自然涌现
        n_active = max(int(np.sum(self.active_mask)), 1)
        delay_active_mean = float(np.sum(delays)) / n_active
        rho_mean = float(np.sum(rho_list)) / n_active

        # 能量回收率 (仅用于日志分析, 不参与奖励)
        total_consume = float(np.sum(energy_out))
        total_charge = float(np.sum(energy_in))
        eta_E = total_charge / max(total_consume, 1e-6)
        eta_E = min(eta_E, 1.0)

        reward = M * rho_mean  # ★ 归一化目标: M × mean(ρ_active)

        # (c) 低电量安全约束惩罚 (二次函数)
        battery_penalty = 0.0
        for k in range(K):
            ratio = self.e_battery_uav[k] / E_BATTERY_INIT
            if ratio < BATTERY_LOW_THRESHOLD:
                deficit = BATTERY_LOW_THRESHOLD - ratio
                battery_penalty += deficit ** 2
        reward -= REWARD_BATTERY_PENALTY * battery_penalty

        # (d) 电量耗尽 → 终止 (无额外惩罚, 失去后续奖励即为隐式惩罚)
        any_dead = not all(energy_ok)

        # ---- 8. UE 更新 ----
        self._update_ue()
        self.current_step += 1

        # ---- 9. 日志 ----
        self.trajectory_log.append(self.loc_uav.copy())
        self.ue_position_log.append(self.loc_ue.copy())
        self.association_log.append(association.copy())
        self.delay_log.append(delays.copy())
        self.rho_log.append(rho_list.copy())
        self.reward_log.append(reward)
        self.battery_log.append(self.e_battery_uav.copy())
        self.charge_log.append(charge_power.copy())

        # 终止条件: 任一 UAV 电池耗尽, 或到达最后时隙
        done = any_dead or (self.current_step >= self.slot_num)
        
        info = {
            'delays': delays, 'rho': rho_list,
            'association': association, 'energy_ok': energy_ok,
            'N_k': N_k,
            'delay_active_mean': delay_active_mean,
            'rho_mean': rho_mean,
            'eta_E': eta_E,
            'charge_power': charge_power,
            'energy_in': energy_in,
            'energy_out': energy_out,
            'battery_start': battery_start,
            'battery_postflight': battery_postflight,
            'comp_energy': comp_energy,
            'edge_delays': edge_delays,
            'energy_failure_phase': None,
            'service_executed': True,
            'step': self.current_step,
        }
        return self._get_obs(), reward, done, info

    def _energy_failure(self, phase, energy_ok, **diagnostics):
        """Search failed before execution: no motion, consumption, or charging.

        This does not certify that the continuous feasible set is empty.
        The returned state is terminal; reset is required before another step.
        """
        self.current_step += 1
        reward = -float(M)
        # Local-only fallback is a finite diagnostic, not completed UAV service.
        delays = np.where(self.active_mask, self.task_list * S_CYCLES / F_UE, 0.0)
        info = {
            'energy_ok': energy_ok, 'energy_failure_phase': phase,
            'service_executed': False, 'step': self.current_step,
            'delays': delays, 'rho': np.zeros(M),
            'association': np.full(M, -1, dtype=int), 'N_k': np.zeros(K, dtype=int),
            'delay_active_mean': float(np.sum(delays) / max(np.sum(self.active_mask), 1)),
            'rho_mean': 0.0, 'eta_E': 0.0,
            'charge_power': np.zeros(K), 'energy_in': np.zeros(K),
            'energy_out': np.zeros(K),
            'candidate_energy': diagnostics,
        }
        self.trajectory_log.append(self.loc_uav.copy())
        self.ue_position_log.append(self.loc_ue.copy())
        self.association_log.append(info['association'].copy())
        self.delay_log.append(delays.copy())
        self.rho_log.append(info['rho'].copy())
        self.reward_log.append(reward)
        self.battery_log.append(self.e_battery_uav.copy())
        self.charge_log.append(info['charge_power'].copy())
        return self._get_obs(), reward, True, info

    # ==================================================================
    # WCT 充电 (eq.9-14)
    # ==================================================================
    def _compute_charging(self, uav_locs):
        if self.disable_wct:
            return np.zeros(K)
        p_matrix = np.zeros((N_WCT, K))
        for j in range(N_WCT):
            for k in range(K):
                dx = uav_locs[k, 0] - self.wct_positions[j, 0]
                dy = uav_locs[k, 1] - self.wct_positions[j, 1]
                dz = H - H_WCT
                l_jk = math.sqrt(dx*dx + dy*dy + dz*dz)
                if l_jk <= L_C:
                    p_matrix[j, k] = ETA_WCT / (THETA_WCT * l_jk + MU_WCT) ** 2

        # WCT 输出约束
        for j in range(N_WCT):
            total_out_j = np.sum(p_matrix[j, :])
            if total_out_j > P_MAX_WCT:
                p_matrix[j, :] *= P_MAX_WCT / total_out_j

        # 汇总 + UAV 接收上限
        return np.minimum(np.sum(p_matrix, axis=0), P_TH)

    # ==================================================================
    @staticmethod
    def _bounce(x, y):
        for _ in range(5):
            bounced = False
            if x < 0:
                x = -x; bounced = True
            elif x > GROUND_LENGTH:
                x = 2 * GROUND_LENGTH - x; bounced = True
            if y < 0:
                y = -y; bounced = True
            elif y > GROUND_WIDTH:
                y = 2 * GROUND_WIDTH - y; bounced = True
            if not bounced:
                break
        return np.clip(x, 0, GROUND_LENGTH), np.clip(y, 0, GROUND_WIDTH)

    def _fly_power(self, v):
        if v < 1e-6:
            return self._hover_power()
        term1 = P0 * (1.0 + 3.0 * v**2 / U_TIP**2)
        inner = math.sqrt(1.0 + v**4 / (4.0 * V0**4)) - v**2 / (2.0 * V0**2)
        term2 = PI * math.sqrt(max(inner, 0.0))
        term3 = 0.5 * D0 * RHO * ROTOR_SOLIDITY * ROTOR_AREA * v**3
        return term1 + term2 + term3

    def _hover_power(self):
        return P0 + PI

    def _channel_gain(self, loc_uav_2d, loc_ue_2d):
        dx = loc_uav_2d[0] - loc_ue_2d[0]
        dy = loc_uav_2d[1] - loc_ue_2d[1]
        d_3d = math.sqrt(dx*dx + dy*dy + H*H)
        if d_3d < 1e-6:
            d_3d = 1e-6
        theta_deg = (180.0 / math.pi) * math.asin(H / d_3d)
        p_los = 1.0 / (1.0 + A_ENV * math.exp(-B_ENV * (theta_deg - A_ENV)))
        fspl = ALPHA0 / (d_3d ** ALPHA_PL)
        # Eq.(16): g = K0^{-1} d^{-alpha} / (P_LoS*mu_LoS + P_NLoS*mu_NLoS)
        avg_extra_loss = p_los * MU_LOS + (1 - p_los) * MU_NLOS
        return fspl / max(avg_extra_loss, 1e-12)

    def _distance_association(self, uav_locs):
        association = np.full(M, -1, dtype=int)
        capacity = np.zeros(K, dtype=int)
        pairs = []
        for i in range(M):
            if not self.active_mask[i]:  # 无任务的用户不关联
                continue
            for k in range(K):
                dx = uav_locs[k, 0] - self.loc_ue[i, 0]
                dy = uav_locs[k, 1] - self.loc_ue[i, 1]
                d_2d = math.sqrt(dx*dx + dy*dy)
                if d_2d > R_COV:  # ★ 超出覆盖范围, 不可服务
                    continue
                d_3d = math.sqrt(d_2d*d_2d + H*H)
                pairs.append((d_3d, i, k))
        pairs.sort(key=lambda x: x[0])
        for _, i, k in pairs:
            if association[i] == -1 and capacity[k] < N_MAX:
                association[i] = k
                capacity[k] += 1
        return association

    def _init_wct_positions(self):
        return WCT_POSITIONS.copy()

    def _init_ue_positions(self):
        # ★ 热点分布: 每个用户围绕其所属热点中心的高斯分布
        locs = np.zeros((M, 2))
        for i in range(M):
            c = self.user_cluster_ids[i]
            locs[i, 0] = self.cluster_centers[c, 0] + np.random.normal(0, CLUSTER_STD)
            locs[i, 1] = self.cluster_centers[c, 1] + np.random.normal(0, CLUSTER_STD)
        locs[:, 0] = np.clip(locs[:, 0], 0, GROUND_LENGTH)
        locs[:, 1] = np.clip(locs[:, 1], 0, GROUND_WIDTH)
        return locs

    def _compute_allocation(self, association, N_k, offload_ratios):
        """
        计算每个用户分到的算力 f_{k,i}.

        均分模式 (use_cvx_alloc=False):
            f_{k,i} = F_UAV / N_k  (所有用户平分, Baseline)

        凸优化最优模式 (use_cvx_alloc=True):
            给定 UAV k 服务的用户集合 S_k, 最小化总计算延迟:
                min  Σ_{i∈S_k} c_i / f_{k,i}
                s.t. Σ_{i∈S_k} f_{k,i} = F_UAV, f_{k,i} > 0
            其中 c_i = x_i · D_i · C_i (用户 i 的卸载计算负载)

            由 KKT 条件推导的闭式最优解:
                f_{k,i}* = F_UAV × √c_i / Σ_{j∈S_k} √c_j

            直觉: 负载大的用户分到更多算力 (按根号比例)
        """
        f_alloc = np.zeros(M)

        if not self.use_cvx_alloc:
            # ---- 均分 (Baseline) ----
            for i in range(M):
                k = association[i]
                if k >= 0 and N_k[k] > 0:
                    f_alloc[i] = F_UAV / N_k[k]
        else:
            # ---- 凸优化最优分配 (闭式解) ----
            for k in range(K):
                if N_k[k] == 0:
                    continue

                # 收集 UAV k 服务的用户及其计算负载
                users_k = []
                loads_k = []
                for i in range(M):
                    # 仅为活跃(有任务)、连接到无人机 k 且产生了真实卸载(x_i > 0)的用户分配算力
                    if self.active_mask[i] == 1 and association[i] == k and offload_ratios[i] > 0:
                        x_i = offload_ratios[i]
                        D_i = self.task_list[i]
                        C_i = D_i * S_CYCLES
                        c_i = x_i * C_i  
                        users_k.append(i)
                        loads_k.append(c_i)

                if len(users_k) == 0:
                    continue

                # 闭式解: f_i* = F_UAV × √c_i / Σ√c_j
                sqrt_loads = np.sqrt(loads_k)
                total_sqrt = np.sum(sqrt_loads)
                if total_sqrt < 1e-12:
                    # 所有负载都为零 → 均分
                    for i in users_k:
                        f_alloc[i] = F_UAV / len(users_k)
                else:
                    for idx, i in enumerate(users_k):
                        f_alloc[i] = F_UAV * sqrt_loads[idx] / total_sqrt

        return f_alloc

    def _compute_bandwidth_allocation(self, association, N_k, offload_ratios, loc_uav):
        """
        计算每个用户分到的带宽 b_{k,i}.

        均分模式 (use_cvx_alloc=False):
            b_{k,i} = B / N_k

        凸优化最优模式 (use_cvx_alloc=True):
            min  Σ_{i∈S_k} a_i / b_{k,i}
            s.t. Σ b_{k,i} = B, b_{k,i} > 0
            其中 a_i = x_i·D_i / log2(1+SNR_i) (传输负载)

            KKT 闭式解: b_{i,k}* = B × √a_i / Σ√a_j
        """
        b_alloc = np.zeros(M)

        if not self.use_cvx_alloc:
            # ---- 均分 (Baseline) ----
            for i in range(M):
                k = association[i]
                if k >= 0 and N_k[k] > 0:
                    b_alloc[i] = BANDWIDTH / N_k[k]
        else:
            # ---- 凸优化最优分配 (闭式解) ----
            for k in range(K):
                if N_k[k] == 0:
                    continue

                users_k = []
                loads_k = []
                for i in range(M):
                    if self.active_mask[i] and association[i] == k and offload_ratios[i] > 1e-10:
                        x_i = offload_ratios[i]
                        D_i = self.task_list[i]
                        g_ik = self._channel_gain(loc_uav[k], self.loc_ue[i])
                        snr = P_TX * g_ik / SIGMA2
                        if snr <= 0:
                            snr = 1e-10
                        a_i = x_i * D_i / math.log2(1.0 + snr)
                        users_k.append(i)
                        loads_k.append(a_i)

                if len(users_k) == 0:
                    continue

                sqrt_loads = np.sqrt(loads_k)
                total_sqrt = np.sum(sqrt_loads)
                if total_sqrt < 1e-12:
                    for i in users_k:
                        b_alloc[i] = BANDWIDTH / len(users_k)
                else:
                    for idx, i in enumerate(users_k):
                        b_alloc[i] = BANDWIDTH * sqrt_loads[idx] / total_sqrt

        return b_alloc

    def _get_current_phase(self):
        """将 episode 均分为3阶段 (自适应 SLOT_NUM)."""
        third = self.slot_num / 3
        if self.current_step < third:
            return 0
        elif self.current_step < 2 * third:
            return 1
        else:
            return 2

    def _generate_tasks(self):
        """根据热点模式生成任务: 热簇0.8到达率, 冷簇0.2."""
        if self.use_hotspot:
            phase = self._get_current_phase()
            hot_clusters = set(self._hotspot_phases[phase])
            probs = np.array([
                self._hot_prob if self.user_cluster_ids[i] in hot_clusters
                else self._cold_prob
                for i in range(M)
            ])
        else:
            probs = np.full(M, TASK_PROB)

        self.active_mask = np.random.random(M) < probs
        self.task_list = np.zeros(M, dtype=np.float64)
        n_active = int(np.sum(self.active_mask))
        if n_active > 0:
            self.task_list[self.active_mask] = np.random.randint(
                TASK_MIN, TASK_MAX + 1, n_active).astype(np.float64)

    def _update_ue(self):
        # ★ 1. 用户小范围随机移动 (围绕簇中心)
        for i in range(M):
            c = self.user_cluster_ids[i]
            self.loc_ue[i, 0] = self.cluster_centers[c, 0] + np.random.normal(0, CLUSTER_STD)
            self.loc_ue[i, 1] = self.cluster_centers[c, 1] + np.random.normal(0, CLUSTER_STD)
        self.loc_ue[:, 0] = np.clip(self.loc_ue[:, 0], 0, GROUND_LENGTH)
        self.loc_ue[:, 1] = np.clip(self.loc_ue[:, 1], 0, GROUND_WIDTH)

        # ★ 2. 分阶段热/冷簇任务到达
        self._generate_tasks()
