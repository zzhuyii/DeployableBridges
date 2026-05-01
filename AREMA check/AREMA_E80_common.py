import os
import sys

import numpy as np

AREMA_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(AREMA_DIR, ".."))
if AREMA_DIR not in sys.path:
    sys.path.insert(0, AREMA_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(1, PROJECT_DIR)

from AREMA_common import check_bar_members_arema, deck_weight
from Solver_NR_Loading import Solver_NR_Loading


KIP_TO_N = 4448.2216152605
FT_TO_M = 0.3048

# Cooper E80 axle pattern for one track. Axle loads are in kips, spacing is in
# feet after each listed axle; the final spacing is to the uniform trailing load.
E80_AXLE_LOADS_KIP = np.array(
    [40, 80, 80, 80, 80, 52, 52, 52, 52, 40, 80, 80, 80, 80, 52, 52, 52, 52],
    dtype=float,
)
E80_SPACING_AFTER_FT = np.array(
    [8, 5, 5, 5, 9, 5, 6, 5, 8, 8, 5, 5, 5, 9, 5, 6, 5, 5],
    dtype=float,
)
E80_TRAILING_LOAD_KIP_PER_FT = 8.0


def e80_axle_positions_m():
    positions_ft = np.concatenate([[0.0], np.cumsum(E80_SPACING_AFTER_FT[:-1])])
    return positions_ft * FT_TO_M


def e80_trailing_start_m():
    return float(np.sum(E80_SPACING_AFTER_FT) * FT_TO_M)


def arema_impact_percent(span_m, ballasted=False):
    """AREMA vertical impact percentage for rolling equipment without hammer blow."""
    span_ft = max(float(span_m) / FT_TO_M, 1e-9)
    if span_ft < 80.0:
        impact = 40.0 - 3.0 * span_ft * span_ft / 1600.0
    else:
        impact = 16.0 + 600.0 / (span_ft - 30.0)
    if ballasted:
        impact *= 0.90
    return max(float(impact), 0.0)


def station_groups_from_nodes(node, node_indices, decimals=6):
    """Group load-path nodes by the same longitudinal x station."""
    coords = node.coordinates_mat
    buckets = {}
    for idx in node_indices:
        idx = int(idx)
        x_key = round(float(coords[idx, 0]), decimals)
        buckets.setdefault(x_key, []).append(idx)
    groups = []
    for x_key in sorted(buckets):
        members = sorted(set(buckets[x_key]))
        x = float(np.mean(coords[members, 0]))
        groups.append({"x": x, "nodes": members})
    if not groups:
        raise ValueError("At least one load-path station is required for E80 loading.")
    return groups


def station_positions(groups, node):
    x_min = float(np.min(node.coordinates_mat[:, 0]))
    x_max = float(np.max(node.coordinates_mat[:, 0]))
    span = max(x_max - x_min, 1e-9)
    xs = np.array([g["x"] - x_min for g in groups], dtype=float)
    order = np.argsort(xs)
    groups = [groups[i] for i in order]
    xs = xs[order]
    return groups, xs, span


def _add_to_group(fz, group, load_n):
    share = float(load_n) / max(len(group["nodes"]), 1)
    for node_idx in group["nodes"]:
        fz[int(node_idx)] += share


def distribute_point_load(fz, groups, xs, x, load_n):
    if x < 0.0 or x > float(xs[-1] if xs.size else 0.0) + 1e9:
        return
    if len(groups) == 1:
        _add_to_group(fz, groups[0], load_n)
        return
    if x <= xs[0]:
        _add_to_group(fz, groups[0], load_n)
        return
    if x >= xs[-1]:
        _add_to_group(fz, groups[-1], load_n)
        return
    right = int(np.searchsorted(xs, x, side="right"))
    left = right - 1
    denom = max(float(xs[right] - xs[left]), 1e-12)
    w_right = float((x - xs[left]) / denom)
    w_left = 1.0 - w_right
    _add_to_group(fz, groups[left], load_n * w_left)
    _add_to_group(fz, groups[right], load_n * w_right)


def distribute_uniform_load(fz, groups, xs, span_m, x0, x1, q_n_per_m):
    if x1 <= 0.0 or x0 >= span_m:
        return
    a = max(float(x0), 0.0)
    b = min(float(x1), float(span_m))
    if b <= a:
        return
    if len(groups) == 1:
        _add_to_group(fz, groups[0], q_n_per_m * (b - a))
        return
    mids = 0.5 * (xs[:-1] + xs[1:])
    bounds = np.concatenate([[0.0], mids, [span_m]])
    for i, group in enumerate(groups):
        left = max(a, float(bounds[i]))
        right = min(b, float(bounds[i + 1]))
        if right > left:
            _add_to_group(fz, group, q_n_per_m * (right - left))


def e80_live_load_vector(node_count, groups, xs, span_m, train_start_m, impact_factor=1.0, reverse=False):
    fz = np.zeros(int(node_count), dtype=float)
    axle_positions = e80_axle_positions_m()
    axle_loads = E80_AXLE_LOADS_KIP * KIP_TO_N * float(impact_factor)
    trailing_start = e80_trailing_start_m()
    trailing_q = E80_TRAILING_LOAD_KIP_PER_FT * KIP_TO_N / FT_TO_M * float(impact_factor)
    trailing_length = span_m + max(span_m / 8.0, 1.0)

    for pos, load_n in zip(axle_positions, axle_loads):
        x = float(train_start_m + pos)
        if reverse:
            x = span_m - x
        if 0.0 <= x <= span_m:
            distribute_point_load(fz, groups, xs, x, load_n)

    u0 = float(train_start_m + trailing_start)
    u1 = u0 + trailing_length
    if reverse:
        u0, u1 = span_m - u1, span_m - u0
    distribute_uniform_load(fz, groups, xs, span_m, u0, u1, trailing_q)
    return fz


def load_rows_from_fz(fz):
    node_ids = np.arange(len(fz), dtype=float)
    return np.column_stack([node_ids, np.zeros(len(fz)), np.zeros(len(fz)), -np.asarray(fz, dtype=float)])


def dead_load_vector(node_count, total_dead_load_n):
    return np.full(int(node_count), float(total_dead_load_n) / max(int(node_count), 1), dtype=float)


def run_e80_service_check(
    assembly,
    node,
    bar,
    load_station_groups,
    supports,
    An,
    r_val,
    Fy,
    Fu,
    bridge_weight_n,
    include_deck_weight=True,
    include_impact=True,
    ballasted=False,
    train_step_m=None,
    extra_check=None,
):
    groups, xs, span_m = station_positions(load_station_groups, node)
    node_count = node.coordinates_mat.shape[0]
    impact_pct = arema_impact_percent(span_m, ballasted=ballasted) if include_impact else 0.0
    impact_factor = 1.0 + impact_pct / 100.0
    step_m = train_step_m or max(span_m / 40.0, 0.25)
    pattern_length = e80_trailing_start_m() + span_m + max(span_m / 8.0, 1.0)
    train_starts = np.arange(-pattern_length, span_m + step_m, step_m)
    base_dead = float(bridge_weight_n) + (deck_weight() if include_deck_weight else 0.0)

    nr = Solver_NR_Loading()
    nr.assembly = assembly
    nr.supp = np.asarray(supports, dtype=float)
    nr.increStep = 1
    nr.iterMax = 50
    nr.tol = 1.0e-5
    nr.verbose = False

    worst = {
        "safe": True,
        "max_dcr": -np.inf,
        "train_start_m": None,
        "direction": "forward",
        "impact_percent": impact_pct,
        "span_m": span_m,
        "step_m": step_m,
        "checked_positions": 0,
        "member_safe": True,
        "extra_safe": True,
        "extra": {},
        "U_end": None,
        "truss_strain": None,
        "pass_yn": None,
        "dcr": None,
    }

    dead_fz = dead_load_vector(node_count, base_dead)
    for direction in ("forward", "reverse"):
        reverse = direction == "reverse"
        for train_start in train_starts:
            live_fz = e80_live_load_vector(
                node_count, groups, xs, span_m, train_start, impact_factor=impact_factor, reverse=reverse
            )
            if not np.any(live_fz > 0.0):
                continue
            nr.load = load_rows_from_fz(dead_fz + live_fz)
            U_end = nr.Solve()[-1]
            truss_strain, pass_yn, dcr = check_bar_members_arema(bar, node, U_end, An, r_val, Fy, Fu)
            member_safe = bool(np.all(pass_yn))
            max_dcr = float(np.nanmax(dcr))
            extra_safe = True
            extra = {}
            if extra_check is not None:
                extra_safe, extra = extra_check(U_end)
                if "dcr" in extra:
                    max_dcr = max(max_dcr, float(extra["dcr"]))
            safe = member_safe and bool(extra_safe)
            worst["checked_positions"] += 1
            if max_dcr > worst["max_dcr"]:
                worst.update(
                    {
                        "safe": safe,
                        "max_dcr": max_dcr,
                        "train_start_m": float(train_start),
                        "direction": direction,
                        "member_safe": member_safe,
                        "extra_safe": bool(extra_safe),
                        "extra": extra,
                        "U_end": U_end,
                        "truss_strain": truss_strain,
                        "pass_yn": pass_yn,
                        "dcr": dcr,
                    }
                )

    if worst["U_end"] is None:
        raise RuntimeError("No E80 train positions overlapped the bridge load path.")
    return worst
