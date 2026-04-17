import os
import sys
import time
import numpy as np
import matplotlib.pyplot as plt

"""Python conversion of Rolling_Bridge_Pedestrain_Load_LRFD.m.

Source MATLAB path:
D:\\PAPER\\1st paper\\2026-DeployableBridges\\Rolling_Bridge_Pedestrain_Load_LRFD.m

The MATLAB source defines N=8, and this file intentionally keeps N fixed at 8.
"""

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from LRFD_Checks import check_truss_lrfd, local_buckling_pass
from Solver_NR_Loading import Solver_NR_Loading
from Rolling_Bridge_common import build_rolling_bridge


def main():
    print("RUNNING FILE:", __file__)
    start = time.time()
    out_dir = os.path.dirname(os.path.abspath(__file__))

    H = 2.0
    W = 2.0
    L = 2.0
    N = 8
    assert N == 8, "Rolling_Bridge_Pedestrain_Load_LRFD.m uses N=8."
    barA = 0.00415
    barE = 2.0e11
    barA_brace = 0.0019
    Ix = 7.16e-6

    Fy = 345e6
    Fu = 427e6
    An = barA * 0.9
    Rp = 1.0
    r_val = np.sqrt(Ix / barA)
    K_eff = 1.0

    local_ok, lambda_r = local_buckling_pass(barE, Fy)
    print("--- Local Buckling Check (AASHTO LRFD Art. 6.9.4.2) ---")
    print("  Section is non-slender (local buckling OK)" if local_ok else "  WARNING: Section fails local buckling slenderness limit")
    print(f"  lambda_r = {lambda_r:.2f}")

    assembly, node, bar, actBar, cst, rot_spr_4N, plots = build_rolling_bridge(
        H=H,
        W=W,
        L=L,
        N=N,
        barA=barA,
        barE=barE,
        panel_E=2.0e8,
        panel_t=0.01,
        panel_v=0.3,
        activeBarE=2.0e11,
        rotK=1.0e6,
        barA_brace=barA_brace,
    )
    assembly.Initialize_Assembly()

    rho_steel = 7850.0
    g = 9.81
    L_total = 0.0
    W_bar = 0.0
    for idx, (n1, n2) in enumerate(bar.node_ij_mat):
        length = np.linalg.norm(node.coordinates_mat[n1 - 1, :] - node.coordinates_mat[n2 - 1, :])
        L_total += length
        W_bar += length * bar.A_vec[idx] * rho_steel * g

    W_deck = 2.0 * (0.03 + 10.0 / 50.0 * 0.2) * 16.0 * 1000.0 * 9.8
    qPL = 3.6e3
    W_LL = qPL * 16.0 * 2.0
    W_factored = 1.25 * (W_bar + W_deck) + 1.75 * W_LL
    force = W_factored / 14.0 / 5.0

    nr = Solver_NR_Loading()
    nr.assembly = assembly
    node_num = node.coordinates_mat.shape[0]
    nr.supp = np.column_stack([
        np.arange(node_num),
        np.zeros(node_num),
        np.zeros(node_num),
        np.zeros(node_num),
    ])
    for support_node in [1, 4, 45, 47]:
        nr.supp[support_node - 1, 1:4] = 1

    Uhis = None
    total_F = 0.0
    pass_yn = None
    DCR = None
    truss_strain = None

    for step in range(1, 6):
        loads = []
        total_F = 0.0
        for k in range(1, N):
            for node_id in [6 * (k - 1) + 3, 6 * (k - 1) + 5]:
                loads.append([node_id - 1, 0.0, 0.0, -force * step])
            total_F += force * 2.0 * step

        nr.load = np.array(loads, dtype=float)
        nr.increStep = 1
        nr.iterMax = 50
        nr.tol = 1.0e-5
        Uhis = nr.Solve()

        U_end = Uhis[-1, :, :]
        truss_strain = bar.solve_strain(node, U_end)
        internal_force = truss_strain * bar.E_vec * bar.A_vec
        Lc = K_eff * bar.L0_vec.reshape(-1)

        pass_yn = np.zeros(internal_force.size, dtype=bool)
        DCR = np.zeros(internal_force.size, dtype=float)
        for j, Pu in enumerate(1.5 * internal_force):
            passed, _, _, _, _, dcr = check_truss_lrfd(
                Pu, bar.A_vec[j], An, bar.E_vec[j], Lc[j], r_val, Fy, Fu, Rp
            )
            pass_yn[j] = passed
            DCR[j] = dcr

        if np.all(pass_yn):
            print(f"Step {step:2d} : All Truss Members Safe (AASHTO LRFD)")
        else:
            print(f"Step {step:2d} : Member Failure Detected (AASHTO LRFD)")
            break

    U_end = Uhis[-1, :, :]
    mid_nodes = [3 * N - 3 - 1, 3 * N - 1 - 1]
    Uaverage = -float(np.mean(U_end[mid_nodes, 2]))
    Kstiff = total_F / Uaverage if abs(Uaverage) > 1e-12 else np.inf

    print("-----------------------------")
    print(f"Total length of all bars: {L_total:.2f} m")
    print(f"Total bar weight: {W_bar:.2f} N")
    print(f"Total load is: {total_F:.2f} N")
    print(f"Mid-span deflection at Strength limit state is: {Uaverage:.3f} m")
    print(f"Stiffness is: {Kstiff:.2f} N/m")
    print(f"span/disp at Strength limit state is: {16.0 / Uaverage:.2f}")
    print(f"Maximum DCR: {np.max(DCR):.2f}")
    print("-----------------------------")

    truss_stress = truss_strain * bar.E_vec
    fig_stress = plots.Plot_Shape_Bar_Stress(truss_stress)
    fig_stress.savefig(os.path.join(out_dir, "Rolling_Bridge_Load_Bar_Stress.png"), dpi=200, bbox_inches="tight")
    plt.close(fig_stress)

    fig_fail = plots.Plot_Shape_Bar_Failure(pass_yn)
    fig_fail.savefig(os.path.join(out_dir, "Rolling_Bridge_Load_Bar_Failure.png"), dpi=200, bbox_inches="tight")
    plt.close(fig_fail)

    fig_def = plots.Plot_Deformed_Shape(U_end)
    fig_def.savefig(os.path.join(out_dir, "Rolling_Bridge_Load_Deformed.png"), dpi=200, bbox_inches="tight")
    plt.close(fig_def)

    print("Execution Time:", time.time() - start)


if __name__ == "__main__":
    main()
