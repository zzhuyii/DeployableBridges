import os
import matplotlib
matplotlib.use("Agg")
import numpy as np

from AASHTO_Checks import local_buckling_pass
from Origami_Bridge_common import build_origami_bridge,bar_length_and_weight,check_members
from Solver_NR_Loading import Solver_NR_Loading


OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def origami_fail(L, N, designCode):
    
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

    assembly, node, bar, cst, rot_spr_4N, plots = build_origami_bridge(
        L=L, W=2*L, H=L, N=N, barA=barA, barE=barE,
        panel_E=2.0e8, panel_t=0.01, panel_v=0.3, rotK=1.0e8,
    )
    assembly.Initialize_Assembly()
    L_total, W_bar = bar_length_and_weight(node, bar)

    nr = Solver_NR_Loading()
    nr.assembly = assembly
    nr.supp = np.asarray([
        [2 - 1, 1, 1, 1],
        [3 - 1, 1, 1, 1],
        [N * 9 + 2 - 1, 1, 1, 1],
        [N * 9 + 3 - 1, 1, 1, 1],
    ], dtype=float)

    force = 10000.0
    history = []
    Uhis = U_end = truss_strain = pass_yn = dcr = None
    total_F = 0.0

    for step in range(1, 101):
        loads = []
        total_F = 0.0
        for k in range(1, N + 1):
            b = 9 * (k - 1)
            node_ids = [b + 6, b + 8, b + 11, b + 12] if k < N else [b + 6, b + 8]
            for node_id in node_ids:
                loads.append([node_id - 1, 0.0, 0.0, -force * step])
            total_F += force * len(node_ids) * step

        nr.load = np.asarray(loads, dtype=float)
        nr.increStep = 1
        nr.iterMax = 20
        nr.tol = 1.0e-5
        Uhis = nr.Solve()
        U_end = Uhis[-1]
        truss_strain, pass_yn, dcr = check_members(bar, node, U_end, An, r_val, Fy, Fu, Rp, designCode)
        safe = bool(np.all(pass_yn))
        history.append([step, total_F, float(np.nanmax(dcr)), 1.0 if safe else 0.0])
        print(f"Step {step:2d} : {'All Truss Members Safe' if safe else 'Member Failure Detected'} (AASHTO LRFD)")
        if not safe:
            break

    plots.viewAngle1=10
    plots.viewAngle2=-75 

    truss_stress = truss_strain * bar.E_vec
    # save_figure(plots.Plot_Bar_Stress(truss_stress), "Origami_Bridge_Load_To_Fail_Bar_Stress.png")
    # save_figure(plots.Plot_Shape_Bar_Failure(pass_yn), "Origami_Bridge_Load_To_Fail_Bar_Failure.png")
    
    fig1=plots.Plot_Bar_Stress(truss_stress, U_end)
    fig2=plots.Plot_Shape_Bar_Failure(pass_yn,U_end)

    return fig1, fig2, total_F, W_bar

