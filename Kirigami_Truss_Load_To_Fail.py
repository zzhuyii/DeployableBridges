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


def check_members(bar, node, U_end, An, r_val, Fy, Fu, Rp, K_eff=1.0):
    truss_strain = bar.solve_strain(node, U_end)
    internal_force = truss_strain * bar.E_vec * bar.A_vec
    Lc = K_eff * bar.L0_vec.reshape(-1)
    pass_yn = np.zeros(internal_force.size, dtype=bool)
    dcr = np.full(internal_force.size, np.nan, dtype=float)
    for j, Pu in enumerate(1.5 * internal_force):
        passed, _, _, _, _, dcr_j = check_truss_lrfd(
            Pu, bar.A_vec[j], An, bar.E_vec[j], Lc[j], r_val, Fy, Fu, Rp
        )
        pass_yn[j] = passed
        dcr[j] = dcr_j
    return truss_strain, internal_force, pass_yn, dcr


def kirigami_fail(L, N ):
    
    barA = 0.00415
    barE = 2.0e11
    Ix = 7.16e-6
    Fy = 345e6
    Fu = 427e6
    Rp = 1.0

    An = barA * 0.9  
    r_val = np.sqrt(Ix / barA)

    local_ok, lambda_r = local_buckling_pass(barE, Fy)
    print("--- Local Buckling Check (AASHTO LRFD Art. 6.9.4.2) ---")
    print("  Section is non-slender (local buckling OK)" if local_ok else "  WARNING: Section fails local buckling slenderness limit")

    assembly, node, bar, cst, rot_spr_4N, rot_spr_3N, plots = build_kirigami_truss(
        L=L, N=N, barA=barA, barE=barE,
        panel_E=2.0e8, panel_t=0.01, panel_v=0.3,
        rot4K=1.0e8, rot3K=1.0e8,
    )
    assembly.Initialize_Assembly()
    L_total, W_bar = bar_length_and_weight(node, bar)

    nr = Solver_NR_Loading()
    nr.assembly = assembly
    nr.supp = np.array([
        [1 - 1, 1, 1, 1],
        [2 - 1, 1, 1, 1],
        [16 * N + 1 - 1, 1, 1, 1],
        [16 * N + 2 - 1, 1, 1, 1],
    ], dtype=float)

    force = 40000.0
    history = []
    Uhis = U_end = truss_strain = pass_yn = dcr = None
    total_F = 0.0

    for step in range(1, 101):
        loads = []
        total_F = 0.0
        for k in range(1, N):
            loads += [
                [17 + (k - 1) * 16 - 1, 0.0, 0.0, -force * step],
                [18 + (k - 1) * 16 - 1, 0.0, 0.0, -force * step],
            ]
            total_F += force * 2.0 * step

        nr.load = np.asarray(loads, dtype=float)
        nr.increStep = 1
        nr.iterMax = 50
        nr.tol = 1.0e-5
        Uhis = nr.Solve()
        U_end = Uhis[-1]
        truss_strain, _, pass_yn, dcr = check_members(bar, node, U_end, An, r_val, Fy, Fu, Rp)
        max_dcr = float(np.nanmax(dcr))
        safe = bool(np.all(pass_yn))
        history.append([step, total_F, max_dcr, 1.0 if safe else 0.0])
        print(f"Step {step:2d} : {'All Truss Members Safe' if safe else 'Member Failure Detected'} (AASHTO LRFD)")
        if not safe:
            break

    plots.view_angle1=10
    plots.view_angle2=-75 
    
    plots.height=4
    plots.width=8

    truss_stress = truss_strain * bar.E_vec
    # save_figure(plots.Plot_Shape_Bar_Stress(truss_stress, U_end), "Kirigami_Truss_Load_To_Fail_Bar_Stress.png")
    # save_figure(plots.Plot_Shape_Bar_Failure(pass_yn, U_end), "Kirigami_Truss_Load_To_Fail_Bar_Failure.png")
    
    fig1=plots.Plot_Shape_Bar_Stress(truss_stress, U_end)
    fig2=plots.Plot_Shape_Bar_Failure(pass_yn,U_end)

    return fig1, fig2

