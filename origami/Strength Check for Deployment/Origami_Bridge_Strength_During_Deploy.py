import os
import time
import zlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from LRFD_Checks import check_truss_lrfd, local_buckling_pass
from Origami_Bridge_common import build_origami_bridge
from Solver_NR_Loading import Solver_NR_Loading


OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def save_figure(fig, name):
    path = os.path.join(OUT_DIR, name)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def bar_length_and_weight(node, bar, rho_steel=7850.0, g=9.81):
    total_length = 0.0
    total_weight = 0.0
    for idx, (n1, n2) in enumerate(bar.node_ij_mat):
        length = np.linalg.norm(node.coordinates_mat[n1 - 1] - node.coordinates_mat[n2 - 1])
        total_length += length
        total_weight += length * bar.A_vec[idx] * rho_steel * g
    return total_length, total_weight


def _read_mat_tag(data, offset):
    raw = data[offset:offset + 8]
    tag0 = int.from_bytes(raw[:4], "little")
    tag1 = int.from_bytes(raw[4:], "little")
    small_type = tag0 & 0xFFFF
    small_nbytes = tag0 >> 16
    if small_nbytes:
        payload = raw[4:4 + small_nbytes]
        return small_type, small_nbytes, payload, offset + 8
    nbytes = tag1
    start = offset + 8
    end = start + nbytes
    next_offset = end + ((8 - nbytes % 8) % 8)
    return tag0, nbytes, data[start:end], next_offset


def _read_mat_v5_array(path, variable_name):
    with open(path, "rb") as f:
        data = f.read()
    offset = 128
    while offset < len(data):
        data_type, _, payload, offset = _read_mat_tag(data, offset)
        if data_type == 15:  # miCOMPRESSED
            inner = zlib.decompress(payload)
            inner_type, _, matrix_payload, _ = _read_mat_tag(inner, 0)
            if inner_type != 14:
                continue
            result = _parse_mat_matrix(matrix_payload, variable_name)
            if result is not None:
                return result
        elif data_type == 14:  # miMATRIX
            result = _parse_mat_matrix(payload, variable_name)
            if result is not None:
                return result
    raise KeyError(f"Variable {variable_name!r} not found in {path}")


def _parse_mat_matrix(payload, variable_name):
    offset = 0
    _, _, _, offset = _read_mat_tag(payload, offset)  # array flags
    _, _, dim_payload, offset = _read_mat_tag(payload, offset)
    dims = tuple(np.frombuffer(dim_payload, dtype="<i4").astype(int))
    _, _, name_payload, offset = _read_mat_tag(payload, offset)
    name = name_payload.decode("latin1")
    data_type, _, real_payload, offset = _read_mat_tag(payload, offset)
    if name != variable_name:
        return None
    if data_type != 9:
        raise TypeError(f"Expected miDOUBLE for {variable_name!r}, got type {data_type}")
    values = np.frombuffer(real_payload, dtype="<f8")
    return values.reshape(dims, order="F").copy()


def deployment_offset(node_count, dep_rate):
    deploy_dir = os.path.abspath(os.path.join(OUT_DIR, "..", "Deploy"))
    npy_path = os.path.join(deploy_dir, "OrigamiUhis.npy")
    mat_path = os.path.abspath(os.path.join(
        OUT_DIR, "..", "..", "..", "2026-DeployableBridges", "OrigamiUhis.mat"
    ))

    if os.path.exists(npy_path):
        Uhis = np.load(npy_path)
    elif os.path.exists(mat_path):
        Uhis = _read_mat_v5_array(mat_path, "Uhis")
        os.makedirs(deploy_dir, exist_ok=True)
        np.save(npy_path, Uhis)
        print(f"Cached deployment history: {npy_path}")
    else:
        print("WARNING: Origami deployment history not found; using zero deployment offset.")
        return np.zeros((node_count, 3), dtype=float), 0, 0

    if Uhis.shape[1:] != (node_count, 3):
        raise ValueError(f"OrigamiUhis shape {Uhis.shape} does not match node_count={node_count}")
    dep_step = max(1, int((1.0 - dep_rate) * Uhis.shape[0]))
    idx = min(Uhis.shape[0], dep_step) - 1
    print(f"Using origami deployment history step {idx + 1}/{Uhis.shape[0]}")
    return Uhis[idx], idx + 1, Uhis.shape[0]


def check_members(bar, node, U_end, An, r_val, Fy, Fu, Rp):
    truss_strain = bar.solve_strain(node, U_end)
    internal_force = truss_strain * bar.E_vec * bar.A_vec
    Lc = bar.L0_vec.reshape(-1)
    pass_yn = np.zeros(internal_force.size, dtype=bool)
    dcr = np.full(internal_force.size, np.nan, dtype=float)
    for j, Pu in enumerate(1.5 * internal_force):
        passed, _, _, _, _, dcr_j = check_truss_lrfd(
            Pu, bar.A_vec[j], An, bar.E_vec[j], Lc[j], r_val, Fy, Fu, Rp
        )
        pass_yn[j] = passed
        dcr[j] = dcr_j
    return truss_strain, pass_yn, dcr


def write_summary(name, lines):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {path}")


def main():
    start = time.time()

    L = 2.0
    W = 4.0
    H = 2.0
    N = 4
    dep_rate = 0.3
    barA = 0.00415
    barE = 2.0e11
    Ix = 7.16e-6
    Fy = 345e6
    Fu = 427e6
    An = barA * 0.9
    Rp = 1.0
    r_val = np.sqrt(Ix / barA)

    local_ok, lambda_r = local_buckling_pass(barE, Fy)
    print("--- Local Buckling Check (AASHTO LRFD Art. 6.9.4.2) ---")
    print("  Section is non-slender (local buckling OK)" if local_ok else "  WARNING: Section fails local buckling slenderness limit")
    print(f"  lambda_r = {lambda_r:.2f}")

    assembly, node, bar, cst, rot_spr_4N, plots = build_origami_bridge(
        L=L, W=W, H=H, N=N, barA=barA, barE=barE,
        panel_E=2.0e8, panel_t=0.01, panel_v=0.3, rotK=1.0e8,
    )
    offset, dep_step, dep_steps = deployment_offset(node.coordinates_mat.shape[0], dep_rate)
    node.coordinates_mat = node.coordinates_mat + offset
    assembly.Initialize_Assembly()
    L_total, W_bar = bar_length_and_weight(node, bar)
    W_deck = 2.0 * (0.03 + 10.0 / 50.0 * 0.2) * 16.0 * 1000.0 * 9.8

    nr = Solver_NR_Loading()
    nr.assembly = assembly
    nr.supp = np.asarray([[0, 1, 1, 1], [1, 1, 1, 1], [2, 1, 1, 1], [3, 1, 1, 1]], dtype=float)
    node_num = node.coordinates_mat.shape[0]

    history = []
    Uhis = U_end = truss_strain = pass_yn = dcr = None
    total_F = 0.0
    final_step = 5

    for step in range(1, 6):
        force = (W_bar + W_deck) / node_num / 5.0 * step
        nr.load = np.column_stack([
            np.arange(node_num), np.zeros(node_num), np.zeros(node_num), -force * np.ones(node_num),
        ])
        nr.increStep = 1
        nr.iterMax = 50
        nr.tol = 1.0e-5
        Uhis = nr.Solve()
        U_end = Uhis[-1]
        truss_strain, pass_yn, dcr = check_members(bar, node, U_end, An, r_val, Fy, Fu, Rp)
        total_F = node_num * force
        safe = bool(np.all(pass_yn))
        history.append([step, total_F, float(np.nanmax(dcr)), 1.0 if safe else 0.0])
        print(f"Step {step:2d} : {'All Truss Members Safe' if safe else 'Member Failure Detected'} (AASHTO LRFD)")
        if not safe:
            final_step = step
            break

    Uaverage = -float(np.mean(U_end[[38 - 1, 39 - 1], 2]))
    np.savetxt(
        os.path.join(OUT_DIR, "Origami_Bridge_Strength_During_Deploy_Step_History.csv"),
        np.asarray(history), delimiter=",", header="step,total_load_N,max_DCR,all_members_safe", comments="",
    )
    summary = [
        "Origami_Bridge_Strength_During_Deploy",
        f"Deployment rate: {dep_rate:.3f}",
        f"Deployment step: {dep_step}/{dep_steps}",
        f"Final checked step: {final_step}",
        f"Total length of all bars: {L_total:.2f} m",
        f"Total bar weight: {W_bar:.2f} N",
        f"Deck weight: {W_deck:.2f} N",
        f"Maximum stress ratio: {np.nanmax(dcr):.3f}",
        f"Tip deflection: {Uaverage:.6f} m",
        f"Execution time: {time.time() - start:.2f} s",
    ]
    write_summary("Origami_Bridge_Strength_During_Deploy_Summary.txt", summary)

    truss_stress = truss_strain * bar.E_vec
    save_figure(plots.Plot_Shape_Bar_Stress(truss_stress), "Origami_Bridge_Strength_During_Deploy_Bar_Stress.png")
    save_figure(plots.Plot_Shape_Bar_Failure(pass_yn), "Origami_Bridge_Strength_During_Deploy_Bar_Failure.png")
    save_figure(plots.Plot_Deformed_Shape(U_end), "Origami_Bridge_Strength_During_Deploy_Deformed.png")


if __name__ == "__main__":
    main()
