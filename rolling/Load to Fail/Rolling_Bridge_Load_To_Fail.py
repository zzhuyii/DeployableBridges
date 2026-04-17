import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from LRFD_Checks import check_truss_lrfd, local_buckling_pass
from Rolling_Bridge_common import build_rolling_bridge
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

    H = 2.0
    W = 2.0
    L = 2.0
    N = 8
    barA = 0.00415
    barE = 2.0e11
    barA_brace = 0.0019
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

    assembly, node, bar, actBar, cst, rot_spr_4N, plots = build_rolling_bridge(
        H=H, W=W, L=L, N=N, barA=barA, barE=barE,
        panel_E=2.0e8, panel_t=0.01, panel_v=0.3,
        activeBarE=2.0e11, rotK=1.0e6, barA_brace=barA_brace,
    )
    assembly.Initialize_Assembly()
    L_total, W_bar = bar_length_and_weight(node, bar)

    nr = Solver_NR_Loading()
    nr.assembly = assembly
    node_num = node.coordinates_mat.shape[0]
    nr.supp = np.column_stack([np.arange(node_num), np.zeros(node_num), np.zeros(node_num), np.zeros(node_num)])
    for support_node in [1, 4, 45, 47]:
        nr.supp[support_node - 1, 1:4] = 1

    force = 2000.0
    history = []
    Uhis = U_end = truss_strain = pass_yn = dcr = None
    total_F = 0.0
    failed_step = 100

    for step in range(1, 101):
        loads = []
        total_F = 0.0
        for k in range(1, N):
            for node_id in [6 * (k - 1) + 3, 6 * (k - 1) + 5]:
                loads.append([node_id - 1, 0.0, 0.0, -force * step])
            total_F += force * 2.0 * step

        nr.load = np.asarray(loads, dtype=float)
        nr.increStep = 1
        nr.iterMax = 50
        nr.tol = 1.0e-5
        Uhis = nr.Solve()
        U_end = Uhis[-1]
        truss_strain, pass_yn, dcr = check_members(bar, node, U_end, An, r_val, Fy, Fu, Rp)
        safe = bool(np.all(pass_yn))
        history.append([step, total_F, float(np.nanmax(dcr)), 1.0 if safe else 0.0])
        print(f"Step {step:2d} : {'All Truss Members Safe' if safe else 'Member Failure Detected'} (AASHTO LRFD)")
        if not safe:
            failed_step = step
            break

    mid_nodes = [3 * N - 3 - 1, 3 * N - 1 - 1]
    Uaverage = -float(np.mean(U_end[mid_nodes, 2]))
    Kstiff = total_F / Uaverage if abs(Uaverage) > 1.0e-12 else np.inf

    np.savetxt(
        os.path.join(OUT_DIR, "Rolling_Bridge_Load_To_Fail_Step_History.csv"),
        np.asarray(history), delimiter=",", header="step,total_load_N,max_DCR,all_members_safe", comments="",
    )
    summary = [
        "Rolling_Bridge_Load_To_Fail",
        f"Failed or final step: {failed_step}",
        f"Total length of all bars: {L_total:.2f} m",
        f"Total bar weight: {W_bar:.2f} N",
        f"Failure load: {total_F:.2f} N",
        f"Mid-span deflection at failure: {Uaverage:.6f} m",
        f"Stiffness: {Kstiff:.2f} N/m",
        f"span/disp at failure: {16.0 / Uaverage:.2f}",
        f"capacity/weight: {total_F / W_bar:.2f}",
        f"Maximum DCR: {np.nanmax(dcr):.3f}",
        f"Execution time: {time.time() - start:.2f} s",
    ]
    write_summary("Rolling_Bridge_Load_To_Fail_Summary.txt", summary)

    truss_stress = truss_strain * bar.E_vec
    save_figure(plots.Plot_Shape_Bar_Stress(truss_stress), "Rolling_Bridge_Load_To_Fail_Bar_Stress.png")
    save_figure(plots.Plot_Shape_Bar_Failure(pass_yn), "Rolling_Bridge_Load_To_Fail_Bar_Failure.png")
    save_figure(plots.Plot_Deformed_Shape(U_end), "Rolling_Bridge_Load_To_Fail_Deformed.png")


if __name__ == "__main__":
    main()
