import os
import matplotlib
matplotlib.use("Agg")
import numpy as np

from KirigamiTruss_common import build_kirigami_truss,bar_length_and_weight,check_members,deployment_offset
from AASHTO_Checks import local_buckling_pass
from Solver_NR_Loading import Solver_NR_Loading


OUT_DIR = os.path.dirname(os.path.abspath(__file__))
def kirigami_deploy(L, N, dep_rate, designCode):
    
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
    print(f"  lambda_r = {lambda_r:.2f}")

    assembly, node, bar, cst, rot_spr_4N, rot_spr_3N, plots = build_kirigami_truss(
        L=L, N=N, barA=barA, barE=barE,
        panel_E=2.0e8, panel_t=0.01, panel_v=0.3,
        rot4K=1.0e7, rot3K=1.0e8,
    )
    node.coordinates_mat = node.coordinates_mat + deployment_offset(node.coordinates_mat.shape[0], dep_rate, N) * L/2.0
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
        truss_strain, pass_yn, dcr = check_members(bar, node, U_end, An, r_val, Fy, Fu, Rp, designCode)
        total_F = node_num * force
        safe = bool(np.all(pass_yn))
        history.append([step, total_F, float(np.nanmax(dcr)), 1.0 if safe else 0.0])
        print(f"Step {step:2d} : {'All Truss Members Safe' if safe else 'Member Failure Detected'} (AASHTO LRFD)")
        if not safe:
            break
        
    plots.view_angle1=10
    plots.view_angle2=-75
    
    plots.height=4
    plots.width=8

    truss_stress = truss_strain * bar.E_vec
    
    fig1=plots.Plot_Shape_Bar_Stress(truss_stress, U_end)
    fig2=plots.Plot_Shape_Bar_Failure(pass_yn,U_end)
    
    U1=U_end[16 * N + 1 - 1, 2]
    U2=U_end[16 * N + 2 - 1, 2]
    
    tipDeflection=0.5*(U1+U2)

    return fig1, fig2, tipDeflection