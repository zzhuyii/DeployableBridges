import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from KirigamiTruss_common import build_kirigami_truss
from LRFD_Checks import check_truss_lrfd, local_buckling_pass
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


def deployment_offset(node_count, dep_rate):
    deploy_path = os.path.abspath(os.path.join(OUT_DIR, "..", "Deploy", "KirigamiUhis.npy"))
    if not os.path.exists(deploy_path):
        return np.zeros((node_count, 3), dtype=float)
    Uhis = np.load(deploy_path)
    dep_step = max(1, int((1.0 - dep_rate) * Uhis.shape[0]))
    idx = min(Uhis.shape[0], dep_step) - 1
    print(f"Using deployment history step {idx + 1}/{Uhis.shape[0]} from {deploy_path}")
    return Uhis[idx]


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
    N = 8
    dep_rate = 0.0
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

    assembly, node, bar, cst, rot_spr_4N, rot_spr_3N, plots = build_kirigami_truss(
        L=L, gap=0.0, N=N, barA=barA, barE=barE,
        panel_E=2.0e8, panel_t=0.01, panel_v=0.3,
        rot4K=1.0e5, rot3K=1.0e8,
    )
    node.coordinates_mat = node.coordinates_mat + deployment_offset(node.coordinates_mat.shape[0], dep_rate)
    assembly.Initialize_Assembly()
    L_total, W_bar = bar_length_and_weight(node, bar)
    W_deck = 2.0 * (0.03 + 10.0 / 50.0 * 0.2) * 16.0 * 1000.0 * 9.8

    nr = Solver_NR_Loading()
    nr.assembly = assembly
    nr.supp = np.asarray([
        [0, 1, 1, 1],
        [1, 1, 1, 1],
        [2, 1, 1, 1],
        [3, 1, 1, 1],
    ], dtype=float)

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

    Uaverage = -float(np.mean(U_end[[65 - 1, 66 - 1], 2]))
    np.savetxt(
        os.path.join(OUT_DIR, "Kirigami_Truss_Strength_During_Deploy_Step_History.csv"),
        np.asarray(history), delimiter=",", header="step,total_load_N,max_DCR,all_members_safe", comments="",
    )
    summary = [
        "Kirigami_Truss_Strength_During_Deploy",
        f"Deployment rate: {dep_rate:.3f}",
        f"Final checked step: {final_step}",
        f"Total length of all bars: {L_total:.2f} m",
        f"Total bar weight: {W_bar:.2f} N",
        f"Deck weight: {W_deck:.2f} N",
        f"Maximum stress ratio: {np.nanmax(dcr):.3f}",
        f"Tip deflection: {Uaverage:.6f} m",
        f"Execution time: {time.time() - start:.2f} s",
    ]
    write_summary("Kirigami_Truss_Strength_During_Deploy_Summary.txt", summary)

    truss_stress = truss_strain * bar.E_vec
    save_figure(plots.Plot_Shape_Bar_Stress(truss_stress), "Kirigami_Truss_Strength_During_Deploy_Bar_Stress.png")
    save_figure(plots.Plot_Shape_Bar_Failure(pass_yn), "Kirigami_Truss_Strength_During_Deploy_Bar_Failure.png")
    save_figure(plots.Plot_Deformed_Shape(U_end), "Kirigami_Truss_Strength_During_Deploy_Deformed.png")


if __name__ == "__main__":
    main()
