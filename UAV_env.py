"""Multi-UAV MEC environment with pre-execution energy admission.

The policy proposes movement and per-user offloading in one action. The
environment filters movement, associates users, scales offloading if needed,
and recomputes resources before committing a slot. Failed admission produces
a terminal self-transition. The finite movement search is not a continuous
global projection. Energy checks use ENERGY_TOL joules of numerical tolerance.
"""

import math
import time
from experiment_config import EnvironmentConfig
import numpy as np

# Joules, ratio threshold, and conservative correction margin.
ENERGY_TOL = 1e-9
OFFLOAD_EPS = 1e-10
OFFLOAD_SAFETY_MARGIN = 1e-8

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
                 use_hotspot=True, disable_wct=False, experiment_config=None, seed=None):
        """
        Args:
            use_cvx_alloc: True → 凸优化最优算力+带宽分配
                           False → 传统均分 (Baseline)
            use_hotspot: True → 使用 experiment_config 中的热点规则
                         False → 全局统一 TASK_PROB
            disable_wct: True → 关闭充电 (消融实验用)
        """
        self.state_dim = STATE_DIM
        self.action_dim = ACTION_DIM
        self.action_bound = ACTION_BOUND
        self.experiment_config = experiment_config or EnvironmentConfig()
        self.slot_num = self.experiment_config.horizon
        self.battery_capacity = self.experiment_config.battery_capacity
        self.initial_energy = self.experiment_config.initial_energy
        self.wct_power = self.experiment_config.wct_power
        self._set_seed(seed if seed is not None else int(np.random.randint(0, 2**31)))
        self.use_cvx_alloc = use_cvx_alloc
        self.use_hotspot = use_hotspot
        self.disable_wct = disable_wct

        self.wct_positions = self._init_wct_positions()
        self.n_wct = len(self.wct_positions)

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
    def _set_seed(self, seed):
        self.episode_seed = int(seed)
        children = np.random.SeedSequence(self.episode_seed).spawn(4)
        self.init_rng, self.mobility_rng, self.task_rng, self.hotspot_rng = [
            np.random.default_rng(child) for child in children]

    def _build_hotspot_schedule(self):
        cfg = self.experiment_config
        phases, segments = [], []
        if cfg.hotspot_mode == 'fixed':
            phases = [min(2, int(3 * (t % cfg.hotspot_reference_slots) /
                                 cfg.hotspot_reference_slots)) for t in range(self.slot_num + 1)]
        else:
            orders = cfg.test_orders if cfg.hotspot_mode == 'unseen_order' else cfg.train_orders
            # Pick one order per episode, avoiding accidental held-out transitions
            # at boundaries between randomly chosen three-phase blocks.
            order = orders[int(self.hotspot_rng.integers(len(orders)))]
            idx = 0
            while len(phases) <= self.slot_num:
                bounds = cfg.train_dwell
                if cfg.hotspot_mode == 'unseen_timing':
                    bounds = cfg.test_dwell_ranges[int(self.hotspot_rng.integers(len(cfg.test_dwell_ranges)))]
                dwell = int(self.hotspot_rng.integers(bounds[0], bounds[1] + 1))
                segments.append(dict(start=len(phases), phase=int(order[idx % 3]), duration=dwell))
                phases.extend([order[idx % 3]] * dwell)
                idx += 1
        self.hotspot_schedule = np.asarray(phases[:self.slot_num + 1], dtype=int)
        self.hotspot_segments = segments

    def reset(self, seed=None):
        if seed is not None:
            self._set_seed(seed)
        self.current_step = 0
        self._build_hotspot_schedule()
        self.e_battery_uav = np.full(K, self.initial_energy, dtype=np.float64)

        # ★ UAV 初始位置随机化 (在场地内随机撒点)
        # ★ 修改：让所有 UAV 初始集中在场地正中央区域 (250, 250) 附近
        # 这样它们距离位于四个角落 (125,125 等) 的充电站都有一定距离 (~176米)
        # 能更有力地证明它们是“主动”飞过去充电的，而不是“恰好”出生在充电站旁边
        self.loc_uav = np.column_stack([
            self.init_rng.uniform(GROUND_LENGTH * 0.45, GROUND_LENGTH * 0.55, K),
            self.init_rng.uniform(GROUND_WIDTH * 0.45, GROUND_WIDTH * 0.55, K),
        ])

        # ★ 四象限簇中心 (固定位置)
        self.cluster_centers = np.array([
            [GROUND_LENGTH * 0.25, GROUND_WIDTH * 0.75],  # 簇0: 左上 (北)
            [GROUND_LENGTH * 0.75, GROUND_WIDTH * 0.75],  # 簇1: 右上 (东)
            [GROUND_LENGTH * 0.75, GROUND_WIDTH * 0.25],  # 簇2: 右下 (南)
            [GROUND_LENGTH * 0.25, GROUND_WIDTH * 0.25],  # 簇3: 左下 (西)
        ], dtype=np.float64)
        self.user_cluster_ids = np.arange(M) % N_CLUSTERS

        # Three original hotspot pairs; order/dwell come from the schedule.
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
            for j in range(self.n_wct):
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
            [min(self.current_step / self.experiment_config.time_reference_slots, 1.0)],
        ])
        assert state.shape[0] == STATE_DIM, \
            f"state dim {state.shape[0]} != {STATE_DIM}"
        return state

    # ==================================================================
    def _movement_energy_state(self, headings, speeds):
        """Return candidate endpoints, flight energy, charge, and service margins."""
        endpoints = np.copy(self.loc_uav)
        fly_energy = np.zeros(K, dtype=np.float64)
        for k in range(K):
            distance = speeds[k] * T_FLY
            x = self.loc_uav[k, 0] + distance * math.cos(headings[k])
            y = self.loc_uav[k, 1] + distance * math.sin(headings[k])
            endpoints[k] = self._bounce(x, y)
            fly_energy[k] = self._fly_power(speeds[k]) * T_FLY

        charge_power = self._compute_charging(endpoints)
        harvested = charge_power * T_COM
        service_margin = self.e_battery_uav - fly_energy + harvested - self._hover_power() * T_COM
        flight_ok = fly_energy <= self.e_battery_uav + ENERGY_TOL
        return endpoints, fly_energy, charge_power, harvested, service_margin, flight_ok

    def _project_movement_for_energy(self, headings, speeds):
        """Project movement to a verified flight- and hover-feasible candidate."""
        headings = np.asarray(headings, dtype=np.float64).copy()
        speeds = np.asarray(speeds, dtype=np.float64).copy()
        requested_headings = headings.copy()
        requested_speeds = speeds.copy()
        grid = np.linspace(0.0, V_MAX, 513)

        # First project each flight speed onto the sampled, explicitly verified
        # set satisfying E_fly(v) <= B(t). Keep the requested speed unchanged
        # whenever it is already feasible.
        for k in range(K):
            if self._fly_power(speeds[k]) * T_FLY <= self.e_battery_uav[k] + ENERGY_TOL:
                continue
            feasible_speeds = [v for v in grid
                               if self._fly_power(v) * T_FLY <= self.e_battery_uav[k] + ENERGY_TOL]
            if not feasible_speeds:
                return headings, speeds, None, None, None, None, False
            speeds[k] = min(feasible_speeds, key=lambda v: abs(v - requested_speeds[k]))

        state = self._movement_energy_state(headings, speeds)
        if np.all(state[4] >= -ENERGY_TOL) and np.all(state[5]):
            return headings, speeds, *state[:4], True

        # If the requested destination cannot support even hovering, search
        # charging-directed movement candidates. Every accepted candidate is
        # checked against the exact energy equations.
        speed_candidates = set(float(v) for v in np.linspace(0.0, V_MAX, 33))
        speed_candidates.update(float(v) for v in speeds)
        heading_candidates = []
        for k in range(K):
            angles = [float(headings[k])]
            for wct in self.wct_positions:
                dx = wct[0] - self.loc_uav[k, 0]
                dy = wct[1] - self.loc_uav[k, 1]
                if abs(dx) + abs(dy) > 1e-12:
                    angles.append(math.atan2(dy, dx))
            angles.extend(float(a) for a in np.linspace(-math.pi, math.pi, 9)[:-1])
            heading_candidates.append(list(dict.fromkeys(angles)))

            for wct in self.wct_positions:
                distance = math.hypot(wct[0] - self.loc_uav[k, 0],
                                      wct[1] - self.loc_uav[k, 1])
                if T_FLY > 0 and distance / T_FLY <= V_MAX:
                    speed_candidates.add(distance / T_FLY)

        speed_candidates = sorted(speed_candidates)

        def deficit(candidate_state):
            return float(np.sum(np.maximum(0.0, -candidate_state[4])))

        def movement_cost(candidate_headings, candidate_speeds):
            cost = 0.0
            for idx in range(K):
                angle_delta = abs(math.atan2(
                    math.sin(candidate_headings[idx] - requested_headings[idx]),
                    math.cos(candidate_headings[idx] - requested_headings[idx])))
                speed_delta = abs(candidate_speeds[idx] - requested_speeds[idx]) / max(V_MAX, 1e-9)
                cost += angle_delta / math.pi + speed_delta
            return cost

        # Coordinate search accounts for charging-power caps shared by UAVs.
        for _ in range(max(2 * K, 1)):
            current_deficit = deficit(state)
            if np.all(state[4] >= -ENERGY_TOL) and np.all(state[5]):
                return headings, speeds, *state[:4], True
            improved = False
            for k in range(K):
                # Moving a currently feasible UAV can release shared WCT power.
                current_deficit = deficit(state)
                best = None
                best_score = (current_deficit, movement_cost(headings, speeds))
                for angle in heading_candidates[k]:
                    for speed in speed_candidates:
                        if self._fly_power(speed) * T_FLY > self.e_battery_uav[k] + ENERGY_TOL:
                            continue
                        test_headings = headings.copy()
                        test_speeds = speeds.copy()
                        test_headings[k] = angle
                        test_speeds[k] = speed
                        test_state = self._movement_energy_state(test_headings, test_speeds)
                        if not np.all(test_state[5]):
                            continue
                        score = (deficit(test_state), movement_cost(test_headings, test_speeds))
                        if score < best_score:
                            best = (test_headings, test_speeds, test_state)
                            best_score = score
                if best is not None and best_score[0] < current_deficit - ENERGY_TOL:
                    headings, speeds, state = best
                    improved = True
                    if np.all(state[4] >= -ENERGY_TOL):
                        return headings, speeds, *state[:4], True
            if not improved:
                break

        return headings, speeds, *state[:4], bool(np.all(state[4] >= -ENERGY_TOL) and np.all(state[5]))

    def _compute_comp_energy(self, association, offload_ratios, f_alloc):
        """Compute UAV CPU energy for the current association and offloading."""
        comp_energy = np.zeros(K, dtype=np.float64)
        for i in range(M):
            k = association[i]
            if not self.active_mask[i] or k < 0 or offload_ratios[i] <= OFFLOAD_EPS:
                continue
            cycles = self.task_list[i] * S_CYCLES
            frequency = f_alloc[i]
            if not np.isfinite(frequency) or frequency <= 0.0:
                comp_energy[k] = np.inf
            else:
                comp_energy[k] += KAPPA * frequency ** 2 * offload_ratios[i] * cycles
        return comp_energy
    def _reject_action(self, raw_action, failure_phase, diagnostics=None):
        """Return a terminal self-transition; candidate energy is never consumed."""
        delays = np.zeros(M)
        association = np.full(M, -1, dtype=int)
        zeros = np.zeros(K)
        info = {
            'decision_seconds': time.perf_counter() - self._decision_started,
            'battery_before': self.e_battery_uav.copy(),
            'battery_after_flight': self.e_battery_uav.copy(),
            'battery_after_service': self.e_battery_uav.copy(),
            'energy_hover': zeros.copy(),
            'energy_charge_accepted': zeros.copy(),
            'energy_charge_overflow': zeros.copy(),
            'hotspot_phase': self._get_current_phase(),
            'termination_reason': failure_phase,
            'delays': delays.copy(), 'rho': delays.copy(),
            'association': association, 'N_k': np.zeros(K, dtype=int),
            'energy_ok': np.zeros(K, dtype=bool),
            'flight_ok': np.zeros(K, dtype=bool),
            'service_ok': np.zeros(K, dtype=bool),
            'movement_feasible': False, 'final_feasible': False,
            'action_executed': False,
            'movement_projected': False, 'offload_projected': False,
            'action_projected': False, 'association_masked': False,
            'offload_scale': np.ones(K), 'offload_scale_ideal': np.ones(K),
            'offload_ratios': np.zeros(M),
            'cpu_allocation': np.zeros(M), 'bandwidth_allocation': np.zeros(M),
            'delay_active_mean': 0.0, 'rho_mean': 0.0, 'eta_E': 0.0,
            'charge_power': zeros.copy(), 'energy_in': zeros.copy(),
            'energy_out': zeros.copy(), 'energy_fly': zeros.copy(),
            'energy_service': zeros.copy(), 'energy_comp': zeros.copy(),
            'raw_action': raw_action.copy(), 'executed_action': None,
            'failure_phase': failure_phase, 'step': self.current_step,
            'energy_tolerance': ENERGY_TOL,
        }
        if diagnostics:
            info.update(diagnostics)
        # Keep attempt logs for inspection, but freeze the observation and clock.
        self.trajectory_log.append(self.loc_uav.copy())
        self.ue_position_log.append(self.loc_ue.copy())
        self.association_log.append(association.copy())
        self.delay_log.append(delays.copy())
        self.rho_log.append(delays.copy())
        self.reward_log.append(0.0)
        self.battery_log.append(self.e_battery_uav.copy())
        self.charge_log.append(zeros.copy())
        return self._get_obs(), 0.0, True, info

    def step(self, action):
        self._decision_started = time.perf_counter()
        battery_before = self.e_battery_uav.copy()
        hotspot_phase = self._get_current_phase()
        # SAC controls the normalized proposal; never mutate the caller's array.
        raw_action = np.asarray(action, dtype=np.float64).copy()
        if raw_action.shape != (self.action_dim,) or not np.all(np.isfinite(raw_action)):
            raise ValueError("Expected a finite action vector of shape (action_dim,).")
        action = np.clip(raw_action, -1.0, 1.0)
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

        # ---- 1. Energy-safety projection for movement ----
        (theta_list, speed_list, new_loc_uav, fly_energy, charge_power,
         energy_in, movement_safe) = self._project_movement_for_energy(
            theta_list, speed_list)
        if not movement_safe:
            diagnostics = {}
            if fly_energy is not None:
                diagnostics = {
                    'flight_ok': fly_energy <= self.e_battery_uav + ENERGY_TOL,
                    'candidate_energy_fly': fly_energy.copy(),
                    'candidate_energy_in': energy_in.copy(),
                }
            return self._reject_action(
                raw_action, 'safety_filter_no_feasible_candidate', diagnostics)

        flight_ok = fly_energy <= self.e_battery_uav + ENERGY_TOL
        executed_action = action.copy()
        movement_projected = False
        for k in range(K):
            angle_delta = abs(math.atan2(
                math.sin(theta_list[k] - math.atan2(action[3*k+1], action[3*k])),
                math.cos(theta_list[k] - math.atan2(action[3*k+1], action[3*k]))))
            requested_speed = (action[3*k+2] + 1.0) * 0.5 * V_MAX
            movement_projected |= angle_delta > 1e-8 or abs(speed_list[k] - requested_speed) > 1e-8
            executed_action[3*k] = math.cos(theta_list[k])
            executed_action[3*k+1] = math.sin(theta_list[k])
            executed_action[3*k+2] = 2.0 * speed_list[k] / V_MAX - 1.0


        # ---- 2. 距离关联 ----
        association = self._distance_association(new_loc_uav)
        proposed_offload_ratios = offload_ratios.copy()
        eligible = (association >= 0) & np.asarray(self.active_mask, dtype=bool)
        offload_ratios[~eligible | (offload_ratios <= OFFLOAD_EPS)] = 0.0
        association_masked = bool(np.any(offload_ratios != proposed_offload_ratios))
        initial_offload_ratios = offload_ratios.copy()
        N_k = np.zeros(K, dtype=int)
        for i in range(M):
            # 仅统计“活跃且实际发生卸载(x_i>0)”用户，避免幽灵带宽占用
            if self.active_mask[i] and association[i] >= 0 and offload_ratios[i] > OFFLOAD_EPS:
                N_k[association[i]] += 1

        # ---- 3. 凸优化资源分配 ----
        # 给定关联和卸载比例, 求每个用户的最优/均分算力
        f_alloc = self._compute_allocation(
            association, N_k, offload_ratios)
        # ★ 带宽也用凸优化最优分配 (KKT 闭式解)
        b_alloc = self._compute_bandwidth_allocation(
            association, N_k, offload_ratios, new_loc_uav)

        # Project offloading ratios, per UAV, onto the service-energy budget.
        # Scaling all offloading ratios of one UAV by the same factor preserves
        # the conditional KKT allocation ratios; allocations are nevertheless
        # recomputed afterward to keep the implementation exact.
        hover_energy = np.full(K, self._hover_power() * T_COM)
        battery_after_flight = self.e_battery_uav - fly_energy
        service_available = battery_after_flight + energy_in
        comp_energy = self._compute_comp_energy(association, offload_ratios, f_alloc)
        offload_projected = False
        offload_scale = np.ones(K, dtype=np.float64)
        offload_scale_ideal = np.ones(K, dtype=np.float64)
        initial_comp_energy = comp_energy.copy()
        for k in range(K):
            capacity_for_computation = service_available[k] - hover_energy[k]
            if comp_energy[k] <= capacity_for_computation + ENERGY_TOL:
                continue
            users_k = np.where((association == k) & eligible)[0]
            if capacity_for_computation <= 0.0 or comp_energy[k] <= 1e-12:
                scale = 0.0
                offload_scale_ideal[k] = 0.0
            else:
                scale = min(1.0, capacity_for_computation / comp_energy[k])
                offload_scale_ideal[k] = scale
                scale = max(0.0, scale * (1.0 - OFFLOAD_SAFETY_MARGIN))
            offload_ratios[users_k] *= scale
            offload_scale[k] = scale
            tiny_users = users_k[offload_ratios[users_k] <= OFFLOAD_EPS]
            offload_ratios[tiny_users] = 0.0
            offload_projected = True

        if offload_projected:
            executed_action[3*K:3*K+M] = 2.0 * offload_ratios - 1.0
            N_k = np.zeros(K, dtype=int)
            for i in range(M):
                if self.active_mask[i] and association[i] >= 0 and offload_ratios[i] > OFFLOAD_EPS:
                    N_k[association[i]] += 1
            f_alloc = self._compute_allocation(association, N_k, offload_ratios)
            b_alloc = self._compute_bandwidth_allocation(
                association, N_k, offload_ratios, new_loc_uav)
            comp_energy = self._compute_comp_energy(association, offload_ratios, f_alloc)

            # Handle numerical or active-set changes conservatively. Since the
            # movement filter verified the zero-computation hover budget, setting
            # the affected UAVs' offloading to zero restores feasibility.
            infeasible = hover_energy + comp_energy > service_available + ENERGY_TOL
            if np.any(infeasible):
                for k in np.where(infeasible)[0]:
                    offload_ratios[association == k] = 0.0
                    offload_scale[k] = 0.0
                executed_action[3*K:3*K+M] = 2.0 * offload_ratios - 1.0
                N_k = np.zeros(K, dtype=int)
                for i in range(M):
                    if self.active_mask[i] and association[i] >= 0 and offload_ratios[i] > OFFLOAD_EPS:
                        N_k[association[i]] += 1
                f_alloc = self._compute_allocation(association, N_k, offload_ratios)
                b_alloc = self._compute_bandwidth_allocation(
                    association, N_k, offload_ratios, new_loc_uav)
                comp_energy = self._compute_comp_energy(association, offload_ratios, f_alloc)


        # Final admission check precedes ALL physical updates.
        flight_ok = (np.isfinite(fly_energy) & (fly_energy >= 0.0)
                     & (fly_energy <= self.e_battery_uav + ENERGY_TOL))
        service_energy = hover_energy + comp_energy
        service_ok = (np.isfinite(service_energy) & np.isfinite(energy_in)
                      & (energy_in >= 0.0) & (service_energy >= 0.0)
                      & (service_energy <= service_available + ENERGY_TOL))
        energy_ok = flight_ok & service_ok
        executed_action[3*K:3*K+M] = 2.0 * offload_ratios - 1.0
        if not np.all(energy_ok):
            return self._reject_action(raw_action, 'final_energy_check', {
                'movement_feasible': True,
                'flight_ok': flight_ok.copy(), 'service_ok': service_ok.copy(),
                'energy_ok': energy_ok.copy(),
                'movement_projected': bool(movement_projected),
                'offload_projected': bool(offload_projected),
                'action_projected': bool(movement_projected or offload_projected),
                'association_masked': association_masked,
                'offload_scale': offload_scale.copy(),
                'offload_scale_ideal': offload_scale_ideal.copy(),
                'candidate_energy_fly': fly_energy.copy(),
                'candidate_energy_service': service_energy.copy(),
                'candidate_energy_in': energy_in.copy(),
            })

        decision_seconds = time.perf_counter() - self._decision_started

        # ---- 4. 延迟计算 ----
        delays = np.zeros(M)
        rho_list = np.zeros(M)

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

            if k < 0 or x_i <= OFFLOAD_EPS:
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
                T_i = max(T_local, T_edge)

            delays[i] = T_i
            rho_list[i] = 1.0 - T_i / T_baseline if T_baseline > 0 else 0.0

        # ---- 4. 能耗 ----
        energy_out = fly_energy + hover_energy + comp_energy

        # Both phases are feasible; commit the end-of-slot battery update.
        self.e_battery_uav = np.clip(
            battery_after_flight - hover_energy - comp_energy + energy_in,
            0.0,
            self.battery_capacity,
        )

        overflow = np.maximum(0.0, battery_after_flight - service_energy + energy_in - self.battery_capacity)
        accepted_charge = energy_in - overflow
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
            ratio = self.e_battery_uav[k] / self.battery_capacity
            if ratio < BATTERY_LOW_THRESHOLD:
                deficit = BATTERY_LOW_THRESHOLD - ratio
                battery_penalty += deficit ** 2
        reward -= REWARD_BATTERY_PENALTY * battery_penalty

        # (d) 电量耗尽 → 终止 (无额外惩罚, 失去后续奖励即为隐式惩罚)
        any_dead = False  # Rejected proposals returned before committing this slot.

        # ---- 8. UE 更新 ----
        self.current_step += 1
        self._update_ue()

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
            'battery_before': battery_before,
            'battery_after_flight': battery_after_flight.copy(),
            'battery_after_service': self.e_battery_uav.copy(),
            'energy_hover': hover_energy.copy(),
            'energy_charge_accepted': accepted_charge,
            'energy_charge_overflow': overflow,
            'decision_seconds': decision_seconds,
            'hotspot_phase': hotspot_phase,
            'termination_reason': 'horizon' if done else None,
            'delays': delays, 'rho': rho_list,
            'association': association, 'energy_ok': energy_ok,
            'flight_ok': flight_ok, 'service_ok': service_ok,
            'N_k': N_k,
            'delay_active_mean': delay_active_mean,
            'rho_mean': rho_mean,
            'eta_E': eta_E,
            'charge_power': charge_power,
            'energy_in': energy_in,
            'energy_out': energy_out,
            'energy_fly': fly_energy,
            'energy_service': hover_energy + comp_energy,
            'action_projected': bool(movement_projected or offload_projected),
            'movement_projected': movement_projected, 'offload_scale': offload_scale.copy(),
            'raw_action': raw_action.copy(),
            'executed_action': executed_action.copy(),
            'failure_phase': None,
            'action_executed': True,
            'movement_feasible': True, 'final_feasible': True,
            'offload_projected': bool(offload_projected),
            'association_masked': association_masked,
            'offload_scale_ideal': offload_scale_ideal.copy(),
            'initial_offload_ratios': initial_offload_ratios.copy(),
            'offload_ratios': offload_ratios.copy(),
            'initial_comp_energy': initial_comp_energy.copy(),
            'energy_comp': comp_energy.copy(),
            'computation_budget': (service_available - hover_energy).copy(),
            'cpu_allocation': f_alloc.copy(),
            'bandwidth_allocation': b_alloc.copy(),
            'flight_energy_violation': np.maximum(0.0, -battery_after_flight),
            'service_energy_violation': np.maximum(0.0, service_energy - service_available),
            'energy_tolerance': ENERGY_TOL,
            'step': self.current_step,
        }
        return self._get_obs(), reward, done, info

    # ==================================================================
    # WCT 充电 (eq.9-14)
    # ==================================================================
    def _compute_charging(self, uav_locs):
        if self.disable_wct:
            return np.zeros(K)
        p_matrix = np.zeros((self.n_wct, K))
        for j in range(self.n_wct):
            for k in range(K):
                dx = uav_locs[k, 0] - self.wct_positions[j, 0]
                dy = uav_locs[k, 1] - self.wct_positions[j, 1]
                dz = H - H_WCT
                l_jk = math.sqrt(dx*dx + dy*dy + dz*dz)
                if l_jk <= L_C:
                    p_matrix[j, k] = ETA_WCT / (THETA_WCT * l_jk + MU_WCT) ** 2

        # WCT 输出约束
        for j in range(self.n_wct):
            total_out_j = np.sum(p_matrix[j, :])
            if total_out_j > self.wct_power:
                p_matrix[j, :] *= self.wct_power / total_out_j

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
        return WCT_POSITIONS[list(self.experiment_config.wct_indices)].copy()

    def _init_ue_positions(self):
        # ★ 热点分布: 每个用户围绕其所属热点中心的高斯分布
        locs = np.zeros((M, 2))
        for i in range(M):
            c = self.user_cluster_ids[i]
            locs[i, 0] = self.cluster_centers[c, 0] + self.mobility_rng.normal(0, CLUSTER_STD)
            locs[i, 1] = self.cluster_centers[c, 1] + self.mobility_rng.normal(0, CLUSTER_STD)
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
                if (self.active_mask[i] and k >= 0 and N_k[k] > 0
                        and offload_ratios[i] > OFFLOAD_EPS):
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
                    if self.active_mask[i] == 1 and association[i] == k and offload_ratios[i] > OFFLOAD_EPS:
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
                if (self.active_mask[i] and k >= 0 and N_k[k] > 0
                        and offload_ratios[i] > OFFLOAD_EPS):
                    b_alloc[i] = BANDWIDTH / N_k[k]
        else:
            # ---- 凸优化最优分配 (闭式解) ----
            for k in range(K):
                if N_k[k] == 0:
                    continue

                users_k = []
                loads_k = []
                for i in range(M):
                    if self.active_mask[i] and association[i] == k and offload_ratios[i] > OFFLOAD_EPS:
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
        """Phase for the CURRENT observation; schedule is independent of horizon."""
        return int(self.hotspot_schedule[min(self.current_step, self.slot_num)])

    def _generate_tasks(self):
        """Generate tasks using configured hot/cold probabilities."""
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

        self.active_mask = self.task_rng.random(M) < probs
        self.task_list = np.zeros(M, dtype=np.float64)
        sizes = self.task_rng.integers(TASK_MIN, TASK_MAX + 1, M)
        self.task_list[self.active_mask] = sizes[self.active_mask].astype(np.float64)

    def _update_ue(self):
        # ★ 1. 用户小范围随机移动 (围绕簇中心)
        for i in range(M):
            c = self.user_cluster_ids[i]
            self.loc_ue[i, 0] = self.cluster_centers[c, 0] + self.mobility_rng.normal(0, CLUSTER_STD)
            self.loc_ue[i, 1] = self.cluster_centers[c, 1] + self.mobility_rng.normal(0, CLUSTER_STD)
        self.loc_ue[:, 0] = np.clip(self.loc_ue[:, 0], 0, GROUND_LENGTH)
        self.loc_ue[:, 1] = np.clip(self.loc_ue[:, 1], 0, GROUND_WIDTH)

        # ★ 2. 分阶段热/冷簇任务到达
        self._generate_tasks()
