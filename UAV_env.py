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
import time
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
    """多 UAV-MEC 环境；策略控制飞行，卸载与资源由环境逐时隙求解。"""

    def __init__(self, use_cvx_alloc=False,
                 use_hotspot=True, disable_wct=False,
                 association_method="marginal",
                 association_local_search_passes=2,
                 association_capacity=None,
                 offload_mode="joint", fixed_offload_ratio=1.0):
        """
        Args:
            use_cvx_alloc: 在 joint 模式下，True 使用联合优化分配；
                           False 使用均分资源并解析选择卸载比例。
                           fixed 模式下保留原有固定卸载基线分配。
            use_hotspot: True → 分阶段热/冷簇轮转 (3阶段)
                         False → 全局统一 TASK_PROB
            disable_wct: True → 关闭充电 (消融实验用)
            association_method: "marginal" → 按当前 SLRR 边际增益关联
                               "distance" → 距离优先贪心关联 (对照方法)
            association_local_search_passes: 边际关联后的 move/swap 改进轮数上限
            association_capacity: 每架 UAV 的最大关联用户数; 默认使用 N_MAX
            offload_mode: "joint" → 联合求解卸载比例、带宽和算力；
                          "fixed" → 使用 fixed_offload_ratio 对照基线
            fixed_offload_ratio: fixed 模式下每个关联活跃用户的卸载比例
        """
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.action_bound = ACTION_BOUND
        self.slot_num = SLOT_NUM
        self.use_cvx_alloc = use_cvx_alloc
        self.use_hotspot = use_hotspot
        self.disable_wct = disable_wct
        if association_method not in {"marginal", "distance"}:
            raise ValueError("association_method must be 'marginal' or 'distance'")
        self.association_method = association_method
        self.association_local_search_passes = max(
            0, int(association_local_search_passes))
        self.association_capacity = (
            N_MAX if association_capacity is None else max(0, int(association_capacity)))
        if offload_mode not in {"joint", "fixed"}:
            raise ValueError("offload_mode must be 'joint' or 'fixed'")
        self.offload_mode = offload_mode
        self.fixed_offload_ratio = float(np.clip(fixed_offload_ratio, 0.0, 1.0))

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
        association_elapsed = original_check.get(
            'association_search', {}).get('seconds', 0.0)
        selected = candidate if original_check['feasible'] else None
        selected_check = original_check if original_check['feasible'] else None
        if selected is None:
            # Closest feasible member of a finite set, measured in normalized
            # actor coordinates. Every candidate gets its own association and
            # feasibility check; the selected association is reused at execution.
            best_distance = float('inf')
            for alternative in self._safety_candidates(candidate):
                check = self._step_candidate(alternative, evaluate_only=True)
                checks += 1
                association_elapsed += check.get(
                    'association_search', {}).get('seconds', 0.0)
                if check['feasible']:
                    distance = float(np.sum((alternative - candidate) ** 2))
                    if distance < best_distance:
                        selected = alternative.copy()
                        selected_check = check
                        best_distance = distance
        if selected is None:
            result = self._energy_failure(
                'shield_search_exhausted', original_check['energy_ok'],
                original_failure_phase=original_check['phase'])
        else:
            result = self._step_candidate(
                selected, precomputed_association=selected_check)
        info = result[3]
        info.update({
            'proposed_action': proposed,
            'executed_action': None if selected is None else selected.copy(),
            'shield_intervened': selected is None or not np.array_equal(selected, proposed),
            'shield_checks': checks,
            'shield_original_failure': original_check['phase'],
            'association_evaluation_seconds': association_elapsed,
            'shield_status': ('no_feasible_candidate_found' if selected is None
                              else 'accepted' if np.array_equal(selected, proposed)
                              else 'corrected'),
        })
        return result

    def _safety_candidates(self, action):
        """Bounded joint candidates; failure is not proof of an empty feasible set."""
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
                candidate = action.copy()
                candidate[0:3*K:3] = np.cos(angles)
                candidate[1:3*K:3] = np.sin(angles)
                candidate[2:3*K:3] = 2.0 * speed_fraction - 1.0
                yield candidate

    def _step_candidate(self, action, evaluate_only=False,
                        precomputed_association=None):
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

        # ---- 2. 用户关联 ----
        if (precomputed_association is not None and
                'association' in precomputed_association):
            association = precomputed_association['association'].copy()
            association_search = precomputed_association.get(
                'association_search', {})
            allocation = precomputed_association.get('allocation')
        else:
            association, association_search, allocation = self._associate(new_loc_uav)

        # ---- 3. 联合卸载与资源分配 ----
        if allocation is None:
            allocation = self._allocate_for_association(association, new_loc_uav)
        offload_ratios = allocation['offload_ratios'].copy()
        f_alloc = allocation['f_alloc'].copy()
        b_alloc = allocation['b_alloc'].copy()
        N_k = np.zeros(K, dtype=int)
        for i in range(M):
            if self.active_mask[i] and association[i] >= 0 and offload_ratios[i] > 1e-10:
                N_k[association[i]] += 1

        # Compute charging and hovering energy before service. The inner
        # optimizer omits computation energy; a safety projection below caps
        # offloading only if the phase energy budget requires it.
        hover_energy = np.full(K, self._hover_power() * T_COM)
        charge_power = self._compute_charging(new_loc_uav)
        energy_in = charge_power * T_COM

        # Preserve the environment's hard edge-service deadline. If the
        # unconstrained optimizer would exceed T_COM, reduce x_i to the largest
        # feasible value; this remains a deterministic execution projection.
        edge_time_per_ratio = np.zeros(M)
        for i in range(M):
            k = association[i]
            if (not self.active_mask[i] or k < 0 or offload_ratios[i] <= 0.0 or
                    b_alloc[i] <= 0.0 or f_alloc[i] <= 0.0):
                offload_ratios[i] = 0.0
                continue
            spectral = self._spectral_efficiency(new_loc_uav[k], self.loc_ue[i])
            edge_time_per_ratio[i] = (
                self.task_list[i] / (b_alloc[i] * spectral) +
                self.task_list[i] * S_CYCLES / f_alloc[i])
            if edge_time_per_ratio[i] > T_COM:
                offload_ratios[i] = min(
                    offload_ratios[i], T_COM / edge_time_per_ratio[i])

        # Computation energy is kappa*f^2*x*C. Scale x on each UAV, if needed,
        # to fit the service-phase battery budget after charging and hovering.
        for k in range(K):
            users_k = np.flatnonzero(
                self.active_mask & (association == k) & (offload_ratios > 0.0))
            if users_k.size == 0:
                continue
            cycles_k = self.task_list[users_k] * S_CYCLES
            comp_k = KAPPA * f_alloc[users_k] ** 2 * offload_ratios[users_k] * cycles_k
            available = max(
                0.0, battery_postflight[k] + energy_in[k] - hover_energy[k])
            total_comp = float(np.sum(comp_k))
            if total_comp > available and total_comp > 0.0:
                offload_ratios[users_k] *= available / total_comp

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
                spectral = self._spectral_efficiency(new_loc_uav[k], self.loc_ue[i])
                R_ik = b_ik * spectral

                T_trans = x_i * D_i / max(R_ik, 1e-10)
                T_edge_comp = x_i * C_i / max(f_ik, 1e-10)
                T_edge = T_trans + T_edge_comp
                edge_delays[i] = T_edge
                T_i = max(T_local, T_edge)
                comp_energy[k] += KAPPA * (f_ik ** 3) * T_edge_comp

            delays[i] = T_i
            rho_list[i] = 1.0 - T_i / T_baseline if T_baseline > 0 else 0.0

        # ---- 4. 能耗 ----
        energy_out = fly_energy + hover_energy + comp_energy

        # Phase-level service energy accounting: computation energy is represented
        # by its average power comp_energy / T_COM during this phase. This is
        # not a task-event-level execution model. Harvesting can fund service,
        # but can never fund the preceding flight (checked above).
        service_end = battery_postflight + energy_in - hover_energy - comp_energy
        energy_ok = service_end >= 0.0
        if not np.all(energy_ok):
            if evaluate_only:
                return {'feasible': False, 'phase': 'service',
                        'energy_ok': energy_ok,
                        'association': association.copy(),
                        'association_search': association_search,
                        'allocation': allocation}
            raise RuntimeError('Shield attempted to execute infeasible service')
        # Accepted UAV work must finish before the service phase ends.
        # Local completion latency remains an analytical metric, without a
        # local deadline or a cross-slot local queue in this model.
        deadline_ok = edge_delays <= T_COM
        if not np.all(deadline_ok):
            if evaluate_only:
                return {'feasible': False, 'phase': 'edge_deadline',
                        'energy_ok': energy_ok, 'deadline_ok': deadline_ok,
                        'association': association.copy(),
                        'association_search': association_search,
                        'allocation': allocation}
            raise RuntimeError('Shield attempted to execute overdue edge work')
        if evaluate_only:
            return {'feasible': True, 'phase': None, 'energy_ok': energy_ok,
                    'battery_postflight': battery_postflight,
                    'battery_end': np.minimum(E_BATTERY_INIT, service_end),
                    'association': association.copy(),
                    'association_search': association_search,
                    'allocation': allocation}
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
            'association_method': self.association_method,
            'association_search': association_search,
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

    def _spectral_efficiency(self, loc_uav_2d, loc_ue_2d):
        gain = self._channel_gain(loc_uav_2d, loc_ue_2d)
        snr = max(P_TX * gain / SIGMA2, 1e-10)
        return math.log2(1.0 + snr)

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
            if association[i] == -1 and capacity[k] < self.association_capacity:
                association[i] = k
                capacity[k] += 1
        return association

    def _associate_legacy(self, uav_locs, offload_ratios):
        """Dispatch to the selected association rule and return diagnostics."""
        started = time.perf_counter()
        if self.association_method == "distance":
            association = self._distance_association(uav_locs)
            diagnostics = {
                'objective': None, 'cache_misses': 0, 'cache_hits': 0,
                'local_search_passes': 0,
            }
        else:
            association, diagnostics = self._marginal_gain_association_legacy(
                uav_locs, offload_ratios)
        diagnostics['seconds'] = time.perf_counter() - started
        return association, diagnostics

    def _association_value_context_legacy(self, uav_locs, offload_ratios):
        """Precompute per-slot coefficients for the existing allocation model."""
        x = np.asarray(offload_ratios, dtype=np.float64)
        task = np.asarray(self.task_list, dtype=np.float64)
        baseline = task * S_CYCLES / F_UE
        cycles = task * S_CYCLES
        spectral_efficiency = np.zeros((K, M), dtype=np.float64)
        coverage = np.zeros(M, dtype=np.int64)

        for i in range(M):
            if not self.active_mask[i]:
                continue
            for k in range(K):
                dx = uav_locs[k, 0] - self.loc_ue[i, 0]
                dy = uav_locs[k, 1] - self.loc_ue[i, 1]
                if dx * dx + dy * dy > R_COV * R_COV:
                    continue
                coverage[i] |= 1 << k
                gain = self._channel_gain(uav_locs[k], self.loc_ue[i])
                snr = max(P_TX * gain / SIGMA2, 1e-10)
                spectral_efficiency[k, i] = math.log2(1.0 + snr)

        return {
            'x': x, 'task': task, 'baseline': baseline, 'cycles': cycles,
            'spectral_efficiency': spectral_efficiency,
            'coverage': coverage,
        }

    @staticmethod
    def _mask_users(mask):
        """Yield user indices encoded in a bit mask, in ascending order."""
        mask = int(mask)
        while mask:
            bit = mask & -mask
            yield bit.bit_length() - 1
            mask ^= bit

    @staticmethod
    def _mask_size(mask):
        """Count set bits without requiring a particular Python version."""
        mask = int(mask)
        count = 0
        while mask:
            mask &= mask - 1
            count += 1
        return count

    def _association_subset_slrr_legacy(self, k, mask, context):
        """Evaluate a UAV subset using the environment's current allocators.

        This oracle matches the current Actor-offloading plus separate CPU and
        bandwidth allocation path. It makes no monotonicity or submodularity
        assumption.
        """
        users = np.fromiter(self._mask_users(mask), dtype=np.int64)
        if users.size == 0:
            return 0.0

        x = context['x']
        task = context['task']
        baseline = context['baseline']
        cycles = context['cycles']
        spectral = context['spectral_efficiency'][k]
        ratios = x[users]
        edge_mask = ratios > 1e-10
        edge_users = users[edge_mask]

        # For positive ratios below the simulator's edge threshold, the task
        # is evaluated as local-only, as in _step_candidate.
        value = float(np.sum(ratios[(ratios > 0.0) & ~edge_mask]))
        if edge_users.size == 0:
            return value

        if self.use_cvx_alloc:
            # CPU allocation includes every positive x_i once at least one
            # user consumes edge resources; bandwidth includes x_i > 1e-10.
            cpu_users = users[ratios > 0.0]
            cpu_roots = np.sqrt(np.maximum(x[cpu_users] * cycles[cpu_users], 0.0))
            cpu_denom = max(float(np.sum(cpu_roots)), 1e-12)
            cpu_delay = (np.sqrt(np.maximum(x[edge_users] * cycles[edge_users], 0.0))
                         * cpu_denom / F_UAV)

            bw_load = (x[edge_users] * task[edge_users] /
                       np.maximum(spectral[edge_users], 1e-12))
            bw_roots = np.sqrt(np.maximum(bw_load, 0.0))
            bw_denom = max(float(np.sum(bw_roots)), 1e-12)
            bw_delay = bw_roots * bw_denom / BANDWIDTH
        else:
            n_edge = edge_users.size
            cpu_delay = x[edge_users] * cycles[edge_users] * n_edge / F_UAV
            bw_delay = (x[edge_users] * task[edge_users] * n_edge /
                        (BANDWIDTH * np.maximum(spectral[edge_users], 1e-12)))

        local_delay = (1.0 - x[edge_users]) * baseline[edge_users]
        edge_delay = cpu_delay + bw_delay
        valid = baseline[edge_users] > 0.0
        value += float(np.sum(
            1.0 - np.maximum(local_delay[valid], edge_delay[valid]) /
            baseline[edge_users[valid]]))
        return value

    def _marginal_gain_association_legacy(self, uav_locs, offload_ratios):
        """Greedy SLRR association followed by bounded move/swap improvement."""
        context = self._association_value_context_legacy(uav_locs, offload_ratios)
        caches = [dict() for _ in range(K)]
        cache_hits = 0
        cache_misses = 0

        def subset_value(k, mask):
            nonlocal cache_hits, cache_misses
            if mask in caches[k]:
                cache_hits += 1
                return caches[k][mask]
            cache_misses += 1
            value = self._association_subset_slrr_legacy(k, mask, context)
            caches[k][mask] = value
            return value

        active_users = [i for i in range(M) if self.active_mask[i]]
        masks = [0] * K
        counts = [0] * K
        assigned_mask = 0
        current_values = [0.0] * K

        # Greedily add the feasible user-UAV pair with the largest positive
        # change in the actual per-slot sum-SLRR objective.
        for _ in range(min(len(active_users), K * self.association_capacity)):
            best_gain = 1e-12
            best_pair = None
            for i in active_users:
                user_bit = 1 << i
                if assigned_mask & user_bit:
                    continue
                for k in self._mask_users(context['coverage'][i]):
                    if counts[k] >= self.association_capacity:
                        continue
                    candidate_mask = masks[k] | user_bit
                    gain = subset_value(k, candidate_mask) - current_values[k]
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (i, k)
            if best_pair is None:
                break
            i, k = best_pair
            user_bit = 1 << i
            masks[k] |= user_bit
            counts[k] += 1
            assigned_mask |= user_bit
            current_values[k] = subset_value(k, masks[k])

        # A bounded best-improvement search limits online cost. Every accepted
        # move strictly improves this same objective and remains feasible.
        completed_passes = 0
        for _ in range(self.association_local_search_passes):
            current_objective = sum(current_values)
            best_objective = current_objective + 1e-12
            best_masks = None
            owners = np.full(M, -1, dtype=int)
            for k, mask in enumerate(masks):
                for i in self._mask_users(mask):
                    owners[i] = k

            # Reconsider previously unassociated users and allow any assigned
            # user to leave. Cross-UAV moves are searched for overlap users.
            for i in active_users:
                source = int(owners[i])
                bit = 1 << i
                if source < 0:
                    for dest in self._mask_users(context['coverage'][i]):
                        if counts[dest] >= self.association_capacity:
                            continue
                        candidate_masks = list(masks)
                        candidate_masks[dest] |= bit
                        candidate_objective = sum(
                            subset_value(k, candidate_masks[k]) for k in range(K))
                        if candidate_objective > best_objective:
                            best_objective = candidate_objective
                            best_masks = candidate_masks
                    continue

                candidate_masks = list(masks)
                candidate_masks[source] &= ~bit
                candidate_objective = sum(
                    subset_value(k, candidate_masks[k]) for k in range(K))
                if candidate_objective > best_objective:
                    best_objective = candidate_objective
                    best_masks = candidate_masks

                if self._mask_size(context['coverage'][i]) < 2:
                    continue
                for dest in self._mask_users(context['coverage'][i]):
                    if dest == source or counts[dest] >= self.association_capacity:
                        continue
                    candidate_masks = list(masks)
                    candidate_masks[source] &= ~bit
                    candidate_masks[dest] |= bit
                    candidate_objective = sum(
                        subset_value(k, candidate_masks[k]) for k in range(K))
                    if candidate_objective > best_objective:
                        best_objective = candidate_objective
                        best_masks = candidate_masks

            # Swap two selected users only when both cross-links are covered.
            for pos, i in enumerate(active_users):
                source_i = int(owners[i])
                if source_i < 0 or self._mask_size(context['coverage'][i]) < 2:
                    continue
                for j in active_users[pos + 1:]:
                    source_j = int(owners[j])
                    if (source_j < 0 or source_i == source_j or
                            self._mask_size(context['coverage'][j]) < 2):
                        continue
                    if (not (context['coverage'][i] & (1 << source_j)) or
                            not (context['coverage'][j] & (1 << source_i))):
                        continue
                    candidate_masks = list(masks)
                    candidate_masks[source_i] ^= (1 << i) | (1 << j)
                    candidate_masks[source_j] ^= (1 << i) | (1 << j)
                    candidate_objective = sum(
                        subset_value(k, candidate_masks[k]) for k in range(K))
                    if candidate_objective > best_objective:
                        best_objective = candidate_objective
                        best_masks = candidate_masks

            if best_masks is None:
                break
            masks = best_masks
            counts = [self._mask_size(mask) for mask in masks]
            current_values = [subset_value(k, masks[k]) for k in range(K)]
            completed_passes += 1

        association = np.full(M, -1, dtype=int)
        for k, mask in enumerate(masks):
            for i in self._mask_users(mask):
                association[i] = k

        diagnostics = {
            'objective': float(sum(current_values)),
            'cache_misses': cache_misses,
            'cache_hits': cache_hits,
            'local_search_passes': completed_passes,
        }
        return association, diagnostics

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

    @staticmethod
    def _joint_waterfill(A, a, c, bandwidth, cpu_frequency,
                         tolerance=1e-6, max_iter=50):
        """Jointly maximize sum SLRR over offloading ratios, bandwidth and CPU.

        For each user, x*=A/(A+E), where E=a/b+c/f. The remaining concave
        problem is solved from its two KKT multipliers. For a fixed bandwidth
        multiplier, an active-set formula gives the CPU multiplier exactly;
        only the bandwidth multiplier needs bisection.
        """
        A = np.asarray(A, dtype=np.float64)
        a = np.asarray(a, dtype=np.float64)
        c = np.asarray(c, dtype=np.float64)
        n = A.size
        x = np.zeros(n, dtype=np.float64)
        b = np.zeros(n, dtype=np.float64)
        f = np.zeros(n, dtype=np.float64)
        if n == 0:
            return x, b, f, {'lambda': 0.0, 'mu': 0.0, 'iterations': 0}
        valid = (A > 0.0) & (a > 0.0) & (c > 0.0)
        if not np.all(valid) or bandwidth <= 0.0 or cpu_frequency <= 0.0:
            return x, b, f, {'lambda': 0.0, 'mu': 0.0, 'iterations': 0}

        sqrt_A = np.sqrt(A)
        sqrt_a = np.sqrt(a)
        sqrt_c = np.sqrt(c)
        lambda_low = 0.0
        lambda_high = float(np.max(A / a))
        def allocation_at_lambda(lam, include_cpu):
            delta = np.maximum(sqrt_A - np.sqrt(a * lam), 0.0)
            eligible = delta > 0.0
            if not np.any(eligible):
                b_zero = np.zeros(n, dtype=np.float64)
                return b_zero, (b_zero.copy() if include_cpu else None), 0.0, b_zero

            # For q=sqrt(mu), a CPU-active user receives
            # f_i=p_i/q-d_i when q<threshold_i. Sort thresholds and find
            # the consistent active prefix satisfying sum_i f_i=F.
            indices = np.flatnonzero(eligible)
            thresholds = delta[indices] / sqrt_c[indices]
            order = np.argsort(-thresholds)
            indices = indices[order]
            thresholds = thresholds[order]
            p = sqrt_c[indices] * delta[indices] / A[indices]
            d = c[indices] / A[indices]
            cumulative_p = np.cumsum(p)
            cumulative_d = np.cumsum(d)
            q_candidates = cumulative_p / (cpu_frequency + cumulative_d)
            next_thresholds = np.empty_like(thresholds)
            next_thresholds[:-1] = thresholds[1:]
            next_thresholds[-1] = 0.0
            consistent = ((q_candidates <= thresholds) &
                          (q_candidates >= next_thresholds))
            matches = np.flatnonzero(consistent)
            q = q_candidates[matches[0]] if matches.size else q_candidates[-1]
            y = np.maximum(delta - sqrt_c * q, 0.0) / A
            b_try = y * sqrt_a / np.sqrt(lam)
            f_try = y * sqrt_c / q if include_cpu else None
            return b_try, f_try, q, y

        for outer_iteration in range(1, max_iter + 1):
            lam = 0.5 * (lambda_low + lambda_high)
            # CPU allocations are not needed while searching for the bandwidth
            # multiplier. Avoid constructing this vector for every bisection step.
            b_try, _, _, _ = allocation_at_lambda(lam, include_cpu=False)
            b_sum = float(np.sum(b_try))
            if b_sum > bandwidth:
                lambda_low = lam
            else:
                lambda_high = lam
            if abs(b_sum - bandwidth) <= tolerance * bandwidth:
                break

        lam = 0.5 * (lambda_low + lambda_high)
        b, f, q, _ = allocation_at_lambda(lam, include_cpu=True)
        mu = float(q * q)
        positive = (b > 0.0) & (f > 0.0)
        if np.any(positive):
            edge_time = np.zeros(n, dtype=np.float64)
            edge_time[positive] = (
                a[positive] / b[positive] + c[positive] / f[positive])
            x[positive] = A[positive] / (A[positive] + edge_time[positive])
        return x, b, f, {
            'lambda': float(lam), 'mu': float(mu),
            'iterations': int(outer_iteration),
        }

    def _association_value_context(self, uav_locs):
        """Precompute coefficients for evaluating per-UAV subset values."""
        task = np.asarray(self.task_list, dtype=np.float64)
        cycles = task * S_CYCLES
        baseline = cycles / F_UE
        spectral = np.zeros((K, M), dtype=np.float64)
        coverage = np.zeros(M, dtype=np.int64)
        for i in range(M):
            if not self.active_mask[i]:
                continue
            for k in range(K):
                dx = uav_locs[k, 0] - self.loc_ue[i, 0]
                dy = uav_locs[k, 1] - self.loc_ue[i, 1]
                if dx * dx + dy * dy > R_COV * R_COV:
                    continue
                coverage[i] |= 1 << k
                spectral[k, i] = self._spectral_efficiency(
                    uav_locs[k], self.loc_ue[i])
        a = np.divide(task[None, :], np.maximum(spectral, 1e-12))
        return {
            'task': task, 'cycles': cycles, 'baseline': baseline,
            'spectral': spectral, 'a': a, 'coverage': coverage,
        }

    def _association_subset_solution(self, k, users, context):
        """Return the actual subset SLRR and its resource allocation."""
        users = np.asarray(users, dtype=np.int64)
        n = users.size
        result = {
            'value': 0.0, 'users': users,
            'offload_ratios': np.zeros(n, dtype=np.float64),
            'f_alloc': np.zeros(n, dtype=np.float64),
            'b_alloc': np.zeros(n, dtype=np.float64),
        }
        if n == 0:
            return result

        A = context['baseline'][users]
        a = context['a'][k, users]
        c = context['cycles'][users]
        task = context['task'][users]
        spectral = context['spectral'][k, users]

        if self.offload_mode == 'joint':
            if self.use_cvx_alloc:
                ratios, b_alloc, f_alloc, _ = self._joint_waterfill(
                    A, a, c, BANDWIDTH, F_UAV)
            else:
                b_alloc = np.full(n, BANDWIDTH / n, dtype=np.float64)
                f_alloc = np.full(n, F_UAV / n, dtype=np.float64)
                edge_time = a / b_alloc + c / f_alloc
                ratios = A / (A + edge_time)
        else:
            ratios = np.full(n, self.fixed_offload_ratio, dtype=np.float64)
            edge = ratios > 1e-10
            b_alloc = np.zeros(n, dtype=np.float64)
            f_alloc = np.zeros(n, dtype=np.float64)
            n_edge = int(np.sum(edge))
            if n_edge:
                if self.use_cvx_alloc:
                    cpu_roots = np.sqrt(ratios[edge] * c[edge])
                    b_roots = np.sqrt(
                        ratios[edge] * task[edge] /
                        np.maximum(spectral[edge], 1e-12))
                    f_alloc[edge] = F_UAV * cpu_roots / np.sum(cpu_roots)
                    b_alloc[edge] = BANDWIDTH * b_roots / np.sum(b_roots)
                else:
                    f_alloc[edge] = F_UAV / n_edge
                    b_alloc[edge] = BANDWIDTH / n_edge

        edge_time = np.full(n, np.inf, dtype=np.float64)
        edge = (ratios > 1e-10) & (b_alloc > 0.0) & (f_alloc > 0.0)
        edge_time[edge] = a[edge] / b_alloc[edge] + c[edge] / f_alloc[edge]
        local_time = (1.0 - ratios) * A
        completion = np.maximum(local_time, ratios * edge_time)
        completion[ratios <= 1e-10] = A[ratios <= 1e-10]
        slrr = 1.0 - completion / np.maximum(A, 1e-12)
        result.update({
            'value': float(np.sum(slrr)),
            'offload_ratios': ratios,
            'f_alloc': f_alloc,
            'b_alloc': b_alloc,
        })
        return result

    def _association_subset_slrr(self, k, mask, context):
        users = np.fromiter(self._mask_users(mask), dtype=np.int64)
        return self._association_subset_solution(k, users, context)['value']

    def _association_subset_value(self, k, users, context):
        """Score a user subset without retaining per-user allocation arrays."""
        users = np.asarray(users, dtype=np.int64)
        if users.size == 0:
            return 0.0
        if self.offload_mode == 'joint' and self.use_cvx_alloc:
            A = context['baseline'][users]
            a = context['a'][k, users]
            c = context['cycles'][users]
            ratios, b_alloc, f_alloc, _ = self._joint_waterfill(
                A, a, c, BANDWIDTH, F_UAV)

            edge_time = np.full(users.size, np.inf, dtype=np.float64)
            edge = ((ratios > 1e-10) & (b_alloc > 0.0) & (f_alloc > 0.0))
            edge_time[edge] = (a[edge] / b_alloc[edge] +
                               c[edge] / f_alloc[edge])
            local_time = (1.0 - ratios) * A
            completion = np.maximum(local_time, ratios * edge_time)
            completion[ratios <= 1e-10] = A[ratios <= 1e-10]
            slrr = 1.0 - completion / np.maximum(A, 1e-12)
            return float(np.sum(slrr))
        return self._association_subset_solution(k, users, context)['value']

    def _allocate_for_association(self, association, uav_locs):
        context = self._association_value_context(uav_locs)
        allocation = {
            'offload_ratios': np.zeros(M, dtype=np.float64),
            'f_alloc': np.zeros(M, dtype=np.float64),
            'b_alloc': np.zeros(M, dtype=np.float64),
        }
        for k in range(K):
            users = np.flatnonzero(self.active_mask & (association == k))
            solution = self._association_subset_solution(k, users, context)
            allocation['offload_ratios'][users] = solution['offload_ratios']
            allocation['f_alloc'][users] = solution['f_alloc']
            allocation['b_alloc'][users] = solution['b_alloc']
        return allocation

    def _associate(self, uav_locs):
        """Run the selected association rule and return a reusable allocation."""
        started = time.perf_counter()
        if self.association_method == 'distance':
            association = self._distance_association(uav_locs)
            diagnostics = {
                'objective': None, 'cache_misses': 0, 'cache_hits': 0,
                'local_search_passes': 0,
            }
            allocation = None
        else:
            association, diagnostics, allocation = self._marginal_gain_association(
                uav_locs)
        diagnostics['seconds'] = time.perf_counter() - started
        return association, diagnostics, allocation

    def _marginal_gain_association(self, uav_locs):
        """Greedy actual-SLRR association followed by bounded move/swap search."""
        context = self._association_value_context(uav_locs)
        value_caches = [dict() for _ in range(K)]
        cache_hits = 0
        cache_misses = 0

        def subset_value(k, mask):
            nonlocal cache_hits, cache_misses
            if mask in value_caches[k]:
                cache_hits += 1
                return value_caches[k][mask]
            cache_misses += 1
            users = np.fromiter(self._mask_users(mask), dtype=np.int64)
            value = self._association_subset_value(k, users, context)
            value_caches[k][mask] = value
            return value

        active_users = [i for i in range(M) if self.active_mask[i]]
        covered_uavs = {
            i: tuple(self._mask_users(context['coverage'][i]))
            for i in active_users
        }
        masks = [0] * K
        counts = [0] * K
        assigned_mask = 0
        current_values = [0.0] * K
        for _ in range(min(len(active_users), K * self.association_capacity)):
            best_gain = 1e-12
            best_pair = None
            for i in active_users:
                user_bit = 1 << i
                if assigned_mask & user_bit:
                    continue
                for k in covered_uavs[i]:
                    if counts[k] >= self.association_capacity:
                        continue
                    candidate_mask = masks[k] | user_bit
                    gain = subset_value(k, candidate_mask) - current_values[k]
                    if gain > best_gain:
                        best_gain = gain
                        best_pair = (i, k)
            if best_pair is None:
                break
            i, k = best_pair
            user_bit = 1 << i
            masks[k] |= user_bit
            counts[k] += 1
            assigned_mask |= user_bit
            current_values[k] = subset_value(k, masks[k])

        completed_passes = 0
        for _ in range(self.association_local_search_passes):
            current_objective = sum(current_values)
            best_objective = current_objective + 1e-12
            best_masks = None
            owners = np.full(M, -1, dtype=int)
            for k, mask in enumerate(masks):
                for i in self._mask_users(mask):
                    owners[i] = k

            for i in active_users:
                source = int(owners[i])
                bit = 1 << i
                if source < 0:
                    for dest in covered_uavs[i]:
                        if counts[dest] >= self.association_capacity:
                            continue
                        candidate_masks = list(masks)
                        candidate_masks[dest] |= bit
                        candidate_objective = sum(
                            subset_value(k, candidate_masks[k]) for k in range(K))
                        if candidate_objective > best_objective:
                            best_objective = candidate_objective
                            best_masks = candidate_masks
                    continue

                candidate_masks = list(masks)
                candidate_masks[source] &= ~bit
                candidate_objective = sum(
                    subset_value(k, candidate_masks[k]) for k in range(K))
                if candidate_objective > best_objective:
                    best_objective = candidate_objective
                    best_masks = candidate_masks

                if self._mask_size(context['coverage'][i]) < 2:
                    continue
                for dest in covered_uavs[i]:
                    if dest == source or counts[dest] >= self.association_capacity:
                        continue
                    candidate_masks = list(masks)
                    candidate_masks[source] &= ~bit
                    candidate_masks[dest] |= bit
                    candidate_objective = sum(
                        subset_value(k, candidate_masks[k]) for k in range(K))
                    if candidate_objective > best_objective:
                        best_objective = candidate_objective
                        best_masks = candidate_masks

            for pos, i in enumerate(active_users):
                source_i = int(owners[i])
                if source_i < 0 or self._mask_size(context['coverage'][i]) < 2:
                    continue
                for j in active_users[pos + 1:]:
                    source_j = int(owners[j])
                    if (source_j < 0 or source_i == source_j or
                            self._mask_size(context['coverage'][j]) < 2):
                        continue
                    if (not (context['coverage'][i] & (1 << source_j)) or
                            not (context['coverage'][j] & (1 << source_i))):
                        continue
                    candidate_masks = list(masks)
                    candidate_masks[source_i] ^= (1 << i) | (1 << j)
                    candidate_masks[source_j] ^= (1 << i) | (1 << j)
                    candidate_objective = sum(
                        subset_value(k, candidate_masks[k]) for k in range(K))
                    if candidate_objective > best_objective:
                        best_objective = candidate_objective
                        best_masks = candidate_masks

            if best_masks is None:
                break
            masks = best_masks
            counts = [self._mask_size(mask) for mask in masks]
            current_values = [subset_value(k, masks[k]) for k in range(K)]
            completed_passes += 1

        association = np.full(M, -1, dtype=int)
        allocation = {
            'offload_ratios': np.zeros(M, dtype=np.float64),
            'f_alloc': np.zeros(M, dtype=np.float64),
            'b_alloc': np.zeros(M, dtype=np.float64),
        }
        for k, mask in enumerate(masks):
            users = np.fromiter(self._mask_users(mask), dtype=np.int64)
            association[users] = k
            subset_value(k, mask)  # Preserve the cache diagnostics for the final subsets.
            solution = self._association_subset_solution(k, users, context)
            allocation['offload_ratios'][users] = solution['offload_ratios']
            allocation['f_alloc'][users] = solution['f_alloc']
            allocation['b_alloc'][users] = solution['b_alloc']

        diagnostics = {
            'objective': float(sum(current_values)),
            'cache_misses': cache_misses,
            'cache_hits': cache_hits,
            'local_search_passes': completed_passes,
        }
        return association, diagnostics, allocation

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
