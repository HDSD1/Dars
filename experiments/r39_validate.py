"""Independent physical checks for the R3.9 continuous small-instance oracle.

All constants must come from the evaluated model's source snapshot.  Arrays
indexed by users contain only active rows for association, x_ik, fraction_cpu,
bwshare and time_normalized; user_positions/tasks retain all M input rows.
The energy check gives no credit for charging, which is sufficient for the
short, high-initial-battery instances used by this experiment.
"""

import numpy as np


def energy_upper_bound(c):
    """A conservative per-UAV, per-slot bound, independent of the action."""
    vmax = c["V_MAX"]
    fly = c["T_FLY"] * (
        c["P0"] * (1.0 + 3.0 * vmax**2 / c["U_TIP"]**2)
        + c["PI"]
        + 0.5 * c["D0"] * c["RHO"] * c["ROTOR_SOLIDITY"]
        * c["ROTOR_AREA"] * vmax**3
    )
    hover = (c["P0"] + c["PI"]) * c["T_COM"]
    # Sum_i f_i^2 <= (sum_i f_i)^2 <= F_UAV^2, and x_i <= 1.
    compute = c["KAPPA"] * c["TASK_MAX"] * c["S_CYCLES"] * c["F_UAV"]**2
    return {
        "flight_j": float(fly),
        "hover_j": float(hover),
        "compute_j": float(compute),
        "total_per_slot_j": float(fly + hover + compute),
    }


def validate_solution(solution, constants, tolerance=1e-6):
    """Recompute delays and constraints without solver expressions/auxiliaries.

    Required solution keys: positions_before, positions_after, user_positions,
    tasks, active_indices, association, x_ik, fraction_cpu, bwshare,
    time_normalized, battery.  CPU and bandwidth values are resource fractions.
    Reported constraint residuals are dimensionless and positive for violations.
    """
    c = constants
    q0 = np.asarray(solution["positions_before"], dtype=float)
    q1 = np.asarray(solution["positions_after"], dtype=float)
    indices = np.asarray(solution["active_indices"], dtype=int)
    users = np.asarray(solution["user_positions"], dtype=float)[indices]
    tasks = np.asarray(solution["tasks"], dtype=float)[indices]
    assoc = np.asarray(solution["association"], dtype=float)
    offload = np.asarray(solution["x_ik"], dtype=float)
    cpu = np.asarray(solution["fraction_cpu"], dtype=float)
    bandwidth = np.asarray(solution["bwshare"], dtype=float)
    z = np.asarray(solution["time_normalized"], dtype=float)
    battery = np.asarray(solution["battery"], dtype=float)

    distances = np.linalg.norm(users[:, None, :] - q1[None, :, :], axis=2)
    distance3d = np.sqrt(distances**2 + c["H"]**2)
    angle = np.degrees(np.arcsin(c["H"] / distance3d))
    plos = 1.0 / (1.0 + c["A_ENV"] * np.exp(-c["B_ENV"] * (angle - c["A_ENV"])))
    gain = c["ALPHA0"] / distance3d**c["ALPHA_PL"] / (
        plos * c["MU_LOS"] + (1.0 - plos) * c["MU_NLOS"]
    )
    spectral_efficiency = np.log2(1.0 + c["P_TX"] * gain / c["SIGMA2"])
    rate = c["BANDWIDTH"] * bandwidth * spectral_efficiency
    frequency = c["F_UAV"] * cpu
    cycles = tasks * c["S_CYCLES"]
    baseline = cycles / c["F_UE"]
    local = (1.0 - offload.sum(axis=1)) * baseline
    transmission = np.zeros_like(offload)
    computation = np.zeros_like(offload)
    selected = offload > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        np.divide(offload * tasks[:, None], rate, out=transmission, where=selected)
        np.divide(offload * cycles[:, None], frequency, out=computation, where=selected)
    edge = transmission + computation
    completion = np.maximum(local, edge.max(axis=1))
    normalized = completion / baseline

    speeds = np.linalg.norm(q1 - q0, axis=1) / c["T_FLY"]
    induced = np.sqrt(np.maximum(
        np.sqrt(1.0 + speeds**4 / (4.0 * c["V0"]**4))
        - speeds**2 / (2.0 * c["V0"]**2), 0.0
    ))
    flight = c["T_FLY"] * (
        c["P0"] * (1.0 + 3.0 * speeds**2 / c["U_TIP"]**2)
        + c["PI"] * induced
        + 0.5 * c["D0"] * c["RHO"] * c["ROTOR_SOLIDITY"]
        * c["ROTOR_AREA"] * speeds**3
    )
    compute_energy = c["KAPPA"] * np.sum(offload * cycles[:, None] * frequency**2, axis=0)
    hover = np.full(len(q1), (c["P0"] + c["PI"]) * c["T_COM"])
    service = compute_energy + hover

    def positive_max(values):
        return float(np.maximum(np.max(values, initial=0.0), 0.0))

    active_links = assoc > 0.5
    field_scale = np.array([c["GROUND_LENGTH"], c["GROUND_WIDTH"]])
    residuals = {
        "association_integrality": positive_max(np.abs(assoc - np.rint(assoc))),
        "association_bounds": max(positive_max(-assoc), positive_max(assoc - 1.0)),
        "one_uav_per_user": positive_max(assoc.sum(axis=1) - 1.0),
        "uav_user_capacity": positive_max((assoc.sum(axis=0) - c["N_MAX"]) / c["N_MAX"]),
        "offloading_bounds": max(positive_max(-offload), positive_max(offload.sum(axis=1) - 1.0)),
        "offloading_association": positive_max(offload - assoc),
        "cpu_nonnegative": positive_max(-cpu),
        "cpu_association": positive_max(cpu - assoc),
        "cpu_capacity": positive_max(cpu.sum(axis=0) - 1.0),
        "bandwidth_nonnegative": positive_max(-bandwidth),
        "bandwidth_association": positive_max(bandwidth - assoc),
        "bandwidth_capacity": positive_max(bandwidth.sum(axis=0) - 1.0),
        "coverage": positive_max((distances[active_links] - c["R_COV"]) / c["R_COV"]),
        "movement": positive_max(speeds / c["V_MAX"] - 1.0),
        "area": max(positive_max(-q1 / field_scale), positive_max(q1 / field_scale - 1.0)),
        "flight_energy": positive_max((flight - battery) / np.maximum(battery, 1.0)),
        "service_energy_without_harvest": positive_max((flight + service - battery) / np.maximum(battery, 1.0)),
        "normalized_time_bounds": max(positive_max(-z), positive_max(z - 1.0)),
        "latency_epigraph": positive_max(normalized - z),
    }
    finite = bool(all(np.isfinite(v) for v in residuals.values()) and np.all(np.isfinite(normalized)))
    largest = max(residuals.values())
    return {
        "feasible": bool(finite and largest <= tolerance),
        "energy_check": "zero charging credit; sufficient for physical feasibility",
        "tolerance": float(tolerance),
        "max_scaled_violation": float(largest),
        "residuals": residuals,
        "objective_j": float(normalized.sum()),
        "total_slrr": float(np.sum(1.0 - normalized)),
        "solver_epigraph_j": float(z.sum()),
        "mean_delay_s": float(completion.mean()),
        "normalized_delay": normalized.tolist(),
        "completion_delay_s": completion.tolist(),
        "energy_fly_j": flight.tolist(),
        "energy_hover_j": hover.tolist(),
        "energy_comp_j": compute_energy.tolist(),
        "remaining_without_harvest_j": (battery - flight - service).tolist(),
        "spectral_efficiency": spectral_efficiency.tolist(),
        "energy_upper_bound_per_slot": energy_upper_bound(c),
    }
