import os
import time
import numpy as np
import matplotlib.pyplot as plt

from KirigamiTruss_common import build_kirigami_truss
from LRFD_Checks import check_truss_lrfd, local_buckling_pass
from Solver_NR_Loading import Solver_NR_Loading


OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    start = time.time()

    L = 2.0
    N = 8
    barA = 0.00415
    barE = 2.0e11
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

    assembly, node, bar, cst, rot_spr_4N, rot_spr_3N, plots = build_kirigami_truss(
        L=L,
        gap=0.0,
        N=N,
        barA=barA,
        barE=barE,
        panel_E=2.0e8,
        panel_t=0.01,
        panel_v=0.3,
        rot4K=1.0e8,
        rot3K=1.0e8,
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
    nr.supp = np.array([
        [1 - 1, 1, 1, 1],
        [2 - 1, 1, 1, 1],
        [16 * N + 1 - 1, 1, 1, 1],
        [16 * N + 2 - 1, 1, 1, 1],
    ], dtype=float)

    Uhis = None
    total_F = 0.0
    pass_yn = None
    DCR = None
    truss_strain = None
    internal_force = None

    for step in range(1, 6):
        loads = []
        total_F = 0.0
        for k in range(1, N):
            loads += [
                [17 + (k - 1) * 16 - 1, 0.0, 0.0, -force * step],
                [18 + (k - 1) * 16 - 1, 0.0, 0.0, -force * step],
            ]
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
    fig_stress.savefig(os.path.join(OUT_DIR, "Kirigami_Truss_Load_Bar_Stress.png"), dpi=200, bbox_inches="tight")
    plt.close(fig_stress)

    fig_fail = plots.Plot_Shape_Bar_Failure(pass_yn)
    fig_fail.savefig(os.path.join(OUT_DIR, "Kirigami_Truss_Load_Bar_Failure.png"), dpi=200, bbox_inches="tight")
    plt.close(fig_fail)

    fig_def = plots.Plot_Deformed_Shape(U_end)
    fig_def.savefig(os.path.join(OUT_DIR, "Kirigami_Truss_Load_Deformed.png"), dpi=200, bbox_inches="tight")
    plt.close(fig_def)

    print("Execution Time:", time.time() - start)


if __name__ == "__main__":
    main()
