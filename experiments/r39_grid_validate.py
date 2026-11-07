"""Independent original-equation audit for the R3.9 terminal-energy benchmark.

Inputs contain the evaluated checkpoint's constants, not working-tree imports.
Neither solver channel coefficients nor solver energy auxiliaries are used to
compute latency, energy, or battery evolution.
"""
import numpy as np


def positive_max(values):
    return float(np.maximum(np.max(values, initial=0.0), 0.0))


def flight_energy(speeds, c):
    v = np.asarray(speeds, dtype=float)
    induced = np.sqrt(np.maximum(
        np.sqrt(1.0 + v**4 / (4.0 * c['V0']**4))
        - v**2 / (2.0 * c['V0']**2), 0.0))
    power = (c['P0'] * (1.0 + 3.0 * v**2 / c['U_TIP']**2)
             + c['PI'] * induced
             + 0.5*c['D0']*c['RHO']*c['ROTOR_SOLIDITY']*c['ROTOR_AREA']*v**3)
    power = np.where(v < 1e-6, c['P0'] + c['PI'], power)
    return power*c['T_FLY']


def charging_energy(positions, case):
    """Per-tower proportional sharing followed by per-UAV receiver cap."""
    c, env = case['constants'], case['environment']
    positions = np.asarray(positions, dtype=float)
    if (case.get('disable_wct', False) or env.get('disable_wct', False)
            or case.get('env_kwargs', {}).get('disable_wct', False)):
        return np.zeros(len(positions))
    towers = np.asarray(case['wct_positions'], dtype=float).reshape((-1, 2))
    distances = np.sqrt(np.sum((towers[:, None, :]-positions[None, :, :])**2, axis=2)
                        + (c['H']-c['H_WCT'])**2)
    raw = np.where(distances <= c['L_C'],
                   c['ETA_WCT']/(c['THETA_WCT']*distances+c['MU_WCT'])**2, 0.0)
    for j in range(len(towers)):
        total = float(raw[j].sum())
        if total > env['wct_power']:
            raw[j] *= env['wct_power']/total
    return np.minimum(raw.sum(axis=0), c['P_TH'])*c['T_COM']


def channel_efficiency(positions, users, c):
    distances = np.linalg.norm(np.asarray(users)[:, None, :]
                               - np.asarray(positions)[None, :, :], axis=2)
    d3 = np.sqrt(distances**2+c['H']**2)
    theta = np.degrees(np.arcsin(c['H']/d3))
    plos = 1.0/(1.0+c['A_ENV']*np.exp(-c['B_ENV']*(theta-c['A_ENV'])))
    gain = c['ALPHA0']/d3**c['ALPHA_PL']/(plos*c['MU_LOS']+(1-plos)*c['MU_NLOS'])
    return distances, np.log2(1.0+c['P_TX']*gain/c['SIGMA2'])


def _bounce(positions, c):
    q = np.array(positions, dtype=float, copy=True)
    upper = np.array([c['GROUND_LENGTH'], c['GROUND_WIDTH']])
    for _ in range(5):
        below = q < 0
        above = q > upper
        if not np.any(below | above):
            break
        q = np.where(below, -q, q)
        q = np.where(above, 2*upper-q, q)
    return np.clip(q, 0, upper)


def validate_solution(case, solution, tolerance=1e-7, energy_tolerance_j=1e-4):
    """Audit all stages and per-UAV terminal energy using physical replay.

    ``solution`` is a stage list or a dict with a ``stages`` list.  Each stage
    provides positions_before/after, association, x_ik, fraction_cpu, bwshare,
    and time_normalized.  Resource/action matrices contain active-user rows.
    Optional speeds are actual flown speeds; headings allow exact wall-bounce
    validation.  Optional battery_before/after and energy_*_j are compared with
    independently reconstructed quantities.  A stage cannot substitute its
    claimed battery for the battery propagated by this audit.
    """
    rows = solution['stages'] if isinstance(solution, dict) else solution
    c, env = case['constants'], case['environment']
    horizon = len(case['stages'])
    if len(rows) != horizon:
        raise ValueError('Solution and exogenous input horizons differ')
    capacity = float(env['battery_capacity'])
    battery = np.array(case['initial_battery'], dtype=float, copy=True)
    previous_position = np.array(case['initial_positions'], dtype=float, copy=True)
    target = np.array(case['terminal_battery_dars'], dtype=float)
    field = np.array([c['GROUND_LENGTH'], c['GROUND_WIDTH']])
    audits = []
    for t, (exogenous, row) in enumerate(zip(case['stages'], rows)):
        ids = np.asarray(exogenous['active_indices'], dtype=int)
        users = np.asarray(exogenous['user_positions'], dtype=float)[ids]
        tasks = np.asarray(exogenous['tasks'], dtype=float)[ids]
        q0 = np.asarray(row['positions_before'], dtype=float)
        q1 = np.asarray(row['positions_after'], dtype=float)
        shape = (len(ids), len(battery))
        a = np.asarray(row['association'], dtype=float).reshape(shape)
        x = np.asarray(row['x_ik'], dtype=float).reshape(shape)
        f = np.asarray(row['fraction_cpu'], dtype=float).reshape(shape)
        b = np.asarray(row['bwshare'], dtype=float).reshape(shape)
        z = np.asarray(row['time_normalized'], dtype=float)
        distance, spectral = channel_efficiency(q1, users, c)
        cycles = tasks*c['S_CYCLES']
        baseline = cycles/c['F_UE']
        local = (1.0-x.sum(axis=1))*baseline
        transmission, computation = np.zeros(shape), np.zeros(shape)
        with np.errstate(divide='ignore', invalid='ignore'):
            np.divide(x*tasks[:, None], c['BANDWIDTH']*b*spectral,
                      out=transmission, where=x > 0)
            np.divide(x*cycles[:, None], c['F_UAV']*f,
                      out=computation, where=x > 0)
            completion = np.maximum(local, (transmission+computation).max(axis=1))
            normalized = completion/baseline
        displacement = np.linalg.norm(q1-q0, axis=1)
        speeds = np.asarray(row.get('speeds', displacement/c['T_FLY']), dtype=float)
        fly = flight_energy(speeds, c)
        hover = np.full(len(battery), (c['P0']+c['PI'])*c['T_COM'])
        comp = c['KAPPA']*np.sum(x*cycles[:, None]*(c['F_UAV']*f)**2, axis=0)
        harvested = charging_energy(q1, case)
        after_flight = battery-fly
        before_cap = after_flight-hover-comp+harvested
        # Feasibility is checked before the simulator's lower-bound clipping.
        after_service = np.clip(before_cap, 0.0, capacity)
        links = a > 0.5
        residuals = {
            'position_continuity': positive_max(abs(q0-previous_position)/field),
            'area': max(positive_max(-q1/field), positive_max(q1/field-1)),
            'speed': max(positive_max(-speeds/c['V_MAX']), positive_max(speeds/c['V_MAX']-1)),
            'displacement': positive_max((displacement-speeds*c['T_FLY'])/(c['V_MAX']*c['T_FLY'])),
            'association_integrality': positive_max(abs(a-np.rint(a))),
            'association_bounds': max(positive_max(-a), positive_max(a-1)),
            'one_uav_per_user': positive_max(a.sum(axis=1)-1),
            'user_capacity': positive_max((a.sum(axis=0)-c['N_MAX'])/c['N_MAX']),
            'coverage': positive_max((distance[links]-c['R_COV'])/c['R_COV']),
            'offloading_bounds': max(positive_max(-x), positive_max(x.sum(axis=1)-1)),
            'offloading_association': positive_max(x-a),
            'cpu_nonnegative': positive_max(-f),
            'cpu_association': positive_max(f-a),
            'cpu_capacity': positive_max(f.sum(axis=0)-1),
            'bandwidth_nonnegative': positive_max(-b),
            'bandwidth_association': positive_max(b-a),
            'bandwidth_capacity': positive_max(b.sum(axis=0)-1),
            'normalized_time_nonnegative': positive_max(-z),
            'latency_epigraph': positive_max(normalized-z),
        }
        if 'headings' in row or 'executed_action' in row:
            if 'headings' in row:
                headings = np.asarray(row['headings'], dtype=float)
            else:
                action = np.asarray(row['executed_action'], dtype=float)
                headings = np.arctan2(action[1:3*len(battery):3], action[:3*len(battery):3])
            proposed = q0 + speeds[:, None]*c['T_FLY']*np.column_stack((np.cos(headings), np.sin(headings)))
            residuals['heading_endpoint'] = positive_max(abs(_bounce(proposed, c)-q1)/field)
        energy_residuals = {
            'flight_phase_j': positive_max(-after_flight),
            'service_phase_j': positive_max(-before_cap),
            'initial_battery_bounds_j': max(positive_max(-battery), positive_max(battery-capacity)),
        }
        for key, physical in [('battery_before', battery), ('battery_after', after_service),
                              ('energy_fly_j', fly), ('energy_hover_j', hover),
                              ('energy_comp_j', comp), ('energy_in_j', harvested)]:
            if key in row:
                energy_residuals[key+'_mismatch_j'] = positive_max(abs(np.asarray(row[key], dtype=float)-physical))
        max_scaled = max(residuals.values(), default=0.0)
        max_energy = max(energy_residuals.values(), default=0.0)
        finite = bool(all(np.isfinite(v) for v in residuals.values())
                      and all(np.isfinite(v) for v in energy_residuals.values())
                      and np.all(np.isfinite(normalized)))
        audits.append({
            'stage': t, 'feasible': bool(finite and max_scaled <= tolerance and max_energy <= energy_tolerance_j),
            'max_scaled_violation': max_scaled, 'max_energy_violation_j': max_energy,
            'residuals': residuals, 'energy_residuals': energy_residuals,
            'objective_j': float(normalized.sum()), 'total_slrr': float(np.sum(1-normalized)),
            'solver_epigraph_j': float(z.sum()), 'normalized_delay': normalized.tolist(),
            'completion_delay_s': completion.tolist(),
            'mean_delay_s': float(completion.mean()) if len(completion) else 0.0,
            'battery_before': battery.tolist(), 'battery_after_flight': after_flight.tolist(),
            'battery_after': after_service.tolist(),
            'energy_fly_j': fly.tolist(), 'energy_hover_j': hover.tolist(),
            'energy_comp_j': comp.tolist(), 'energy_in_j': harvested.tolist(),
            'spectral_efficiency': spectral.tolist(),
        })
        battery = after_service
        previous_position = q1
    terminal_deficit = np.maximum(target-battery, 0.0)
    maximum_energy = max([positive_max(terminal_deficit)]
                         + [a['max_energy_violation_j'] for a in audits])
    maximum_scaled = max([positive_max(terminal_deficit)/capacity]
                         + [a['max_scaled_violation'] for a in audits])
    return {
        'feasible': bool(all(a['feasible'] for a in audits)
                         and positive_max(terminal_deficit) <= energy_tolerance_j),
        'tolerance': tolerance, 'energy_tolerance_j': energy_tolerance_j,
        'max_scaled_violation': maximum_scaled, 'max_energy_violation_j': maximum_energy,
        'total_slrr': sum(a['total_slrr'] for a in audits),
        'objective_j': sum(a['objective_j'] for a in audits),
        'terminal_battery': battery.tolist(), 'terminal_battery_dars': target.tolist(),
        'terminal_deficit_j': terminal_deficit.tolist(), 'stages': audits,
    }
