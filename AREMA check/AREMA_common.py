import os
import sys

import numpy as np

AREMA_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(AREMA_DIR, ".."))
if AREMA_DIR not in sys.path:
    sys.path.insert(0, AREMA_DIR)
if PROJECT_DIR not in sys.path:
    insert_at = 1 if sys.path and os.path.abspath(sys.path[0]) == AREMA_DIR else 0
    sys.path.insert(insert_at, PROJECT_DIR)

from AREMA_Checks import (
    AREMA_K_BOLTED_OR_WELDED,
    check_truss_arema,
    arema_outstanding_element_pass,
)


def history_path(name):
    local_path = os.path.join(AREMA_DIR, name)
    if os.path.exists(local_path):
        return local_path
    return os.path.join(PROJECT_DIR, name)


def bar_length_and_weight(node, bar, rho_steel=7850.0, g=9.81):
    total_length = 0.0
    total_weight = 0.0
    for idx, (n1, n2) in enumerate(bar.node_ij_mat):
        length = np.linalg.norm(node.coordinates_mat[n1 - 1] - node.coordinates_mat[n2 - 1])
        total_length += length
        total_weight += length * bar.A_vec[idx] * rho_steel * g
    return total_length, total_weight


def member_area_vector(value, Ag_vec, default_ratio=None):
    """Return a per-member area vector from a scalar, vector, or ratio assumption."""
    Ag_vec = np.asarray(Ag_vec, dtype=float).reshape(-1)
    if value is None:
        if default_ratio is None:
            raise ValueError("A member area value or default_ratio is required.")
        return default_ratio * Ag_vec

    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == Ag_vec.size:
        return arr
    if arr.size == 1:
        # Existing LRFD scripts pass An = 0.9 * main Ag. Interpret that as a
        # net-area ratio and apply it consistently to brace members as well.
        reference = max(float(np.max(Ag_vec)), 1e-12)
        ratio = float(arr[0]) / reference
        return ratio * Ag_vec
    raise ValueError(f"Expected scalar or {Ag_vec.size} member values, got {arr.size}.")


def member_value_vector(value, count, name):
    arr = np.asarray(value, dtype=float).reshape(-1)
    if arr.size == count:
        return arr
    if arr.size == 1:
        return np.full(count, float(arr[0]), dtype=float)
    raise ValueError(f"Expected scalar or {count} values for {name}, got {arr.size}.")


def check_bar_members_arema(
    bar,
    node,
    U_end,
    An,
    r_val,
    Fy,
    Fu,
    Ae=None,
    shear_lag_factor=1.0,
    allowable_stress_factor=1.0,
    effective_length_factor=AREMA_K_BOLTED_OR_WELDED,
    compression_slenderness_limit=100.0,
    tension_slenderness_limit=200.0,
    enforce_local_screen=False,
    bt=10.7,
    ht=24.5,
):
    truss_strain = bar.solve_strain(node, U_end)
    internal_force = truss_strain * bar.E_vec * bar.A_vec
    Lc = bar.L0_vec.reshape(-1)
    Ag_vec = np.asarray(bar.A_vec, dtype=float).reshape(-1)
    An_vec = member_area_vector(An, Ag_vec, default_ratio=0.9)
    if Ae is None:
        Ae_vec = float(shear_lag_factor) * An_vec
    else:
        Ae_vec = member_area_vector(Ae, Ag_vec)
    r_vec = member_value_vector(r_val, internal_force.size, "r_val")
    local_ok, local_limit = arema_outstanding_element_pass(float(np.max(bar.E_vec)), Fy, bt=bt, ht=ht)
    local_dcr = max(float(bt), float(ht)) / max(float(local_limit), 1e-12)
    pass_yn = np.zeros(internal_force.size, dtype=bool)
    dcr = np.full(internal_force.size, np.nan, dtype=float)
    for j, Pu in enumerate(internal_force):
        passed, _, _, dcr_j = check_truss_arema(
            Pu,
            Ag_vec[j],
            An_vec[j],
            bar.E_vec[j],
            Lc[j],
            r_vec[j],
            Fy,
            Fu,
            Ae=Ae_vec[j],
            allowable_stress_factor=allowable_stress_factor,
            effective_length_factor=effective_length_factor,
            slenderness_limit=compression_slenderness_limit,
            tension_slenderness_limit=tension_slenderness_limit,
        )
        if enforce_local_screen and not local_ok:
            passed = False
            dcr_j = max(float(dcr_j), float(local_dcr))
        pass_yn[j] = passed
        dcr[j] = dcr_j
    return truss_strain, pass_yn, dcr


def check_scissor_members_arema(
    model,
    U_end,
    An,
    r_val,
    Fy,
    Fu,
    Ae=None,
    shear_lag_factor=1.0,
    effective_length_factor=AREMA_K_BOLTED_OR_WELDED,
):
    return check_bar_members_arema(
        model.bar,
        model.node,
        U_end,
        An,
        r_val,
        Fy,
        Fu,
        Ae=Ae,
        shear_lag_factor=shear_lag_factor,
        effective_length_factor=effective_length_factor,
    )


def print_arema_local_screen(E, Fy):
    local_ok, limit = arema_outstanding_element_pass(E, Fy)
    print("--- AREMA Outstanding Element Screen (MRE Chapter 15) ---")
    print("  Section satisfies the default outstanding-element screen" if local_ok else "  WARNING: Section exceeds the default outstanding-element screen")
    print(f"  b/t limit = {limit:.2f}")
    return local_ok, limit


def deck_weight():
    return 2.0 * (0.03 + 10.0 / 50.0 * 0.2) * 16.0 * 1000.0 * 9.8


def npy_deployment_offset(name, node_count, dep_rate, keep_nodes=None):
    Uhis = np.load(history_path(name))
    if keep_nodes is not None:
        Uhis = Uhis[:, :keep_nodes, :]
    if Uhis.shape[1:] != (node_count, 3):
        raise ValueError(f"{name} shape {Uhis.shape} does not match node_count={node_count}")
    dep_step = max(1, int((1.0 - dep_rate) * Uhis.shape[0]))
    idx = min(Uhis.shape[0], dep_step) - 1
    print(f"Using {name} deployment history step {idx + 1}/{Uhis.shape[0]}")
    return Uhis[idx]


def rolling_deployment_offset(node_count, dep_rate, N):
    Uhis = np.load(history_path("RollingUhis.npz"))["Uhis"]
    Uhis = Uhis[:, :N * 6, :]
    if Uhis.shape[1:] != (node_count, 3):
        raise ValueError(f"RollingUhis shape {Uhis.shape} does not match node_count={node_count}")
    dep_step = max(1, int((1.0 - dep_rate) * Uhis.shape[0]))
    idx = min(Uhis.shape[0], dep_step) - 1
    print(f"Using RollingUhis deployment history step {idx + 1}/{Uhis.shape[0]}")
    return Uhis[idx]


def standard_scissor_deployment_coordinates(model, dep_rate):
    N = model.settings["N"]
    L = model.settings["L"]
    theta = np.pi / 4.0 * dep_rate
    dL = L * np.sqrt(2.0) * np.cos(theta) - L
    L2 = L / np.sqrt(2.0) * np.sin(theta)
    L3 = np.sqrt(max((L / 2.0) ** 2 - L2 ** 2, 0.0))
    coords = []
    for i in range(1, N + 1):
        x0 = 2.0 * L2 * (i - 1)
        xm = x0 + L2
        coords += [
            [x0, 0.0, 0.0], [x0, L, 0.0], [x0, 0.0, L + dL], [x0, L, L + dL],
            [xm, 0.0, (L + dL) / 2.0], [xm, L, (L + dL) / 2.0],
            [xm, 0.0, L3], [xm, L, L3],
        ]
    coords += [[2.0 * L2 * N, 0.0, 0.0], [2.0 * L2 * N, L, 0.0], [2.0 * L2 * N, 0.0, L + dL], [2.0 * L2 * N, L, L + dL]]
    model.node.coordinates_mat = np.asarray(coords, dtype=float)


def improved_scissor_deployment_coordinates(model, dep_rate):
    N = model.settings["N"]
    L = model.settings["L"]
    theta = np.pi / 4.0 * dep_rate
    dL = L * np.sqrt(2.0) * np.cos(theta) - L
    L2 = L / np.sqrt(2.0) * np.sin(theta)
    L3 = np.sqrt(max((L / 2.0) ** 2 - L2 ** 2, 0.0))
    coords = []
    for i in range(1, N + 1):
        x0 = 2.0 * L2 * (i - 1)
        xm = x0 + L2
        coords += [
            [x0, 0.0, 0.0], [x0, L, 0.0], [x0, 0.0, L + dL], [x0, L, L + dL],
            [xm, 0.0, (L + dL) / 2.0], [xm, L, (L + dL) / 2.0],
            [xm, 0.0, L3], [xm, L, L3],
            [xm, 0.0, L + dL - L3], [xm, L, L + dL - L3],
        ]
    coords += [[2.0 * L2 * N, 0.0, 0.0], [2.0 * L2 * N, L, 0.0], [2.0 * L2 * N, 0.0, L + dL], [2.0 * L2 * N, L, L + dL]]
    model.node.coordinates_mat = np.asarray(coords, dtype=float)
