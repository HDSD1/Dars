"""Adapted INFOCOM-2024 matching and IoTJ-2025 TMA association baselines.
Not reproductions of the complete papers: DARS utilities/safety are retained.
"""
from collections import deque
import copy
import math
import time
import numpy as np
from config import K, M, N_MAX, R_COV, F_UAV, F_UE, BANDWIDTH, P_TX, SIGMA2, S_CYCLES
from UAV_env import ENERGY_TOL, OFFLOAD_EPS, OFFLOAD_SAFETY_MARGIN


def deferred_acceptance(scores, eligible, capacity, positive_only=True):
    """Fixed pair preferences; user/UAV IDs break ties deterministically."""
    users, servers = scores.shape
    preferences = []
    for i in range(users):
        candidates = [k for k in range(servers) if eligible[i, k]
                      and (not positive_only or scores[i, k] > 0)]
        preferences.append(sorted(candidates, key=lambda k: (-scores[i, k], k)))
    queue = deque(i for i in range(users) if preferences[i])
    next_choice = np.zeros(users, dtype=int)
    held = [[] for _ in range(servers)]
    proposals = 0
    while queue:
        i = queue.popleft()
        if next_choice[i] == len(preferences[i]):
            continue
        k = preferences[i][next_choice[i]]
        next_choice[i] += 1
        proposals += 1
        held[k].append(i)
        held[k].sort(key=lambda j: (-scores[j, k], j))
        if len(held[k]) > capacity:
            rejected = held[k].pop()
            if next_choice[rejected] < len(preferences[rejected]):
                queue.append(rejected)
    result = np.full(users, -1, dtype=int)
    for k, group in enumerate(held):
        result[group] = k
    return result, proposals


class AssociationSnapshot:
    """Post-movement decision state; calls original allocators without writes.

    Safety arithmetic mirrors UAVEnv.step. The runner checks the greedy
    reconstruction against the live environment at EVERY measured state.
    """
    def __init__(self, before_env, positions, raw_action, info):
        self.env = copy.copy(before_env)
        for name in ('active_mask', 'task_list', 'loc_ue', 'loc_uav', 'e_battery_uav'):
            setattr(self.env, name, np.array(getattr(before_env, name), copy=True))
        self.positions = np.array(positions, copy=True)
        self.active = np.asarray(self.env.active_mask, dtype=bool)
        self.proposal = (np.clip(np.asarray(raw_action, dtype=np.float64), -1, 1)[3*K:3*K+M] + 1) / 2
        self.fly = np.asarray(info['energy_fly']).copy()
        self.harvest = np.asarray(info['energy_in']).copy()
        self.hover = np.asarray(info['energy_hover']).copy()
        self.available = self.env.e_battery_uav - self.fly + self.harvest
        self.budget = self.available - self.hover
        self.eligible = (np.linalg.norm(self.env.loc_ue[:, None, :] -
                         self.positions[None, :, :], axis=2) <= R_COV)
        self.eligible &= self.active[:, None]

    def spectral_efficiency(self):
        spectral = np.zeros((M, K))
        for i, k in zip(*np.where(self.eligible)):
            snr = P_TX * self.env._channel_gain(self.positions[k], self.env.loc_ue[i]) / SIGMA2
            spectral[i, k] = math.log2(1 + (snr if snr > 0 else 1e-10))
        return spectral

    def valid_association(self, assoc):
        a = np.asarray(assoc)
        if a.shape != (M,) or not np.issubdtype(a.dtype, np.integer):
            return False
        if np.any((a < -1) | (a >= K)) or np.any(a[~self.active] != -1):
            return False
        return (all(np.sum(a == k) <= N_MAX for k in range(K)) and
                all(self.eligible[i, a[i]] for i in np.flatnonzero(a >= 0)))

    def evaluate(self, assoc):
        a = np.asarray(assoc)
        if not self.valid_association(a):
            raise ValueError('Association violates coverage, activity or quota')
        x = self.proposal.copy()
        eligible = (a >= 0) & self.active
        x[~eligible | (x <= OFFLOAD_EPS)] = 0

        def allocate():
            counts = np.array([np.sum(self.active & (a == k) & (x > OFFLOAD_EPS))
                               for k in range(K)])
            f = self.env._compute_allocation(a, counts, x)
            b = self.env._compute_bandwidth_allocation(a, counts, x, self.positions)
            return f, b, self.env._compute_comp_energy(a, x, f)

        f, b, comp = allocate()
        projected = False
        scales = np.ones(K)
        for k in range(K):
            if comp[k] <= self.budget[k] + ENERGY_TOL:
                continue
            ids = np.where((a == k) & eligible)[0]
            scale = (0.0 if self.budget[k] <= 0 or comp[k] <= 1e-12 else
                     min(1.0, self.budget[k]/comp[k]) * (1-OFFLOAD_SAFETY_MARGIN))
            x[ids] *= scale
            scales[k] = scale
            x[ids[x[ids] <= OFFLOAD_EPS]] = 0
            projected = True
        if projected:
            f, b, comp = allocate()
            infeasible = self.hover + comp > self.available + ENERGY_TOL
            if np.any(infeasible):
                for k in np.flatnonzero(infeasible):
                    x[a == k] = 0
                    scales[k] = 0
                f, b, comp = allocate()
        flight_ok = (np.isfinite(self.fly) & (self.fly >= 0) &
                     (self.fly <= self.env.e_battery_uav + ENERGY_TOL))
        service = self.hover + comp
        service_ok = (np.isfinite(service) & np.isfinite(self.harvest) &
                      (self.harvest >= 0) & (service >= 0) &
                      (service <= self.available + ENERGY_TOL))
        delay = np.zeros(M)
        rho = np.zeros(M)
        for i in np.flatnonzero(self.active):
            cycles = self.env.task_list[i] * S_CYCLES
            base = cycles / F_UE
            local = (1-x[i]) * base
            k = a[i]
            if k < 0 or x[i] <= OFFLOAD_EPS:
                delay[i] = base if k < 0 else local
            else:
                snr = P_TX * self.env._channel_gain(self.positions[k], self.env.loc_ue[i]) / SIGMA2
                spectral = math.log2(1 + (snr if snr > 0 else 1e-10))
                edge = x[i]*self.env.task_list[i] / max(b[i]*spectral, 1e-10)
                edge += x[i]*cycles / max(f[i], 1e-10)
                delay[i] = max(local, edge)
            rho[i] = 1-delay[i]/base if base > 0 else 0
        return dict(association=a.copy(), feasible=bool(np.all(flight_ok & service_ok)),
                    offload=x, cpu=f, bandwidth=b, delay=delay, rho=rho,
                    J=float(np.sum(1-rho[self.active])),
                    slrr=float(np.mean(rho[self.active])) if np.any(self.active) else 0.0,
                    projected=projected, scales=scales, computation_energy_j=comp,
                    associated_users=int(np.sum(a >= 0)),
                    offloaded_users=int(np.sum(self.active & (x > OFFLOAD_EPS))),
                    energy_excess_j=float(max(0, np.max(service-self.available))))

    def run(self, method, max_evaluations=1000, max_rounds=20, improvement_tol=1e-8):
        """Timing includes preferences and ALL candidate resource/safety calls."""
        started = time.perf_counter()
        proposals = swaps = 0
        termination = 'not_applicable'
        if method == 'greedy':
            a = self.env._distance_association(self.positions)
        elif method == 'matching':
            spectral = self.spectral_efficiency()
            scores = np.full((M, K), -np.inf)
            for i, k in zip(*np.where(self.eligible)):
                base = self.env.task_list[i]*S_CYCLES/F_UE
                x = self.proposal[i]
                edge = x*self.env.task_list[i]/((BANDWIDTH/N_MAX)*spectral[i, k])
                edge += x*self.env.task_list[i]*S_CYCLES/(F_UAV/N_MAX)
                scores[i, k] = 1-max((1-x)*base, edge)/base if base > 0 else 0
            a, proposals = deferred_acceptance(scores, self.eligible, N_MAX)
        elif method == 'swap':
            # Adapted TMA phase I: distance initialization followed by fixed-rate DA.
            initial = self.env._distance_association(self.positions)
            counts = np.array([max(1, np.sum(initial == k)) for k in range(K)])
            rates = self.spectral_efficiency() * (BANDWIDTH/counts)[None, :]
            a, proposals = deferred_acceptance(rates, self.eligible, N_MAX, positive_only=False)
        else:
            raise ValueError(method)
        selected_at = time.perf_counter()
        result = self.evaluate(a)
        evaluations = 1
        initial_j = result['J']
        if not result['feasible']:
            raise RuntimeError('Candidate failed final energy check')
        if method == 'swap':
            termination = 'round_limit'
            for _ in range(max_rounds):
                improved = False
                for candidate in self.neighbors(result['association']):
                    if evaluations >= max_evaluations:
                        termination = 'evaluation_limit'
                        break
                    alternative = self.evaluate(candidate)
                    evaluations += 1
                    if alternative['feasible'] and alternative['J'] < result['J']-improvement_tol:
                        result = alternative
                        swaps += 1
                        improved = True
                        break
                if termination == 'evaluation_limit':
                    break
                if not improved:
                    termination = 'no_improving_neighbor'
                    break
            selected_at = time.perf_counter()
        result.update(method=method, initial_J=initial_j, proposals=proposals,
                      evaluations=evaluations, accepted_changes=swaps,
                      search_termination=termination,
                      budget_exhausted=termination in ('round_limit', 'evaluation_limit'),
                      selection_ms=(selected_at-started)*1000,
                      total_ms=(time.perf_counter()-started)*1000)
        return result

    def neighbors(self, a):
        """Pair exchanges plus dummy-slot/local moves (explicit DARS adaptation)."""
        ids = np.flatnonzero(self.active)
        counts = np.bincount(a[a >= 0], minlength=K)
        for i in ids:
            for k in [-1] + np.flatnonzero(self.eligible[i]).tolist():
                if k == a[i] or (k >= 0 and counts[k] >= N_MAX):
                    continue
                b = a.copy()
                b[i] = k
                yield b
        for ii, i in enumerate(ids):
            for j in ids[ii+1:]:
                ki, kj = a[i], a[j]
                if ki == kj or (kj >= 0 and not self.eligible[i, kj]) or (ki >= 0 and not self.eligible[j, ki]):
                    continue
                b = a.copy()
                b[i], b[j] = kj, ki
                yield b
