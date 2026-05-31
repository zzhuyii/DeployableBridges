import os
import matplotlib
matplotlib.use("Agg")
import numpy as np

from AASHTO_Checks import local_buckling_pass
from Rolling_Bridge_common import build_rolling_bridge,rolling_deploy_offset,bar_length_and_weight,check_members
from Solver_NR_Loading import Solver_NR_Loading


OUT_DIR = os.path.dirname(os.path.abspath(__file__))

def rolling_deploy(N,dep_rate,L,designCode):
    H = L
    W = 2.0

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
    plots.displayRange = np.array([-2.0, N*L+2.0, -1.0, L+2.0, -1.0, N*L+2.0], dtype=float)
    node.coordinates_mat = node.coordinates_mat + rolling_deploy_offset(node.coordinates_mat.shape[0], dep_rate, N)*L/2.0
    assembly.Initialize_Assembly()
    L_total, W_bar = bar_length_and_weight(node, bar)
    W_deck = 2.0 * (0.03 + 10.0 / 50.0 * 0.2) * 16.0 * 1000.0 * 9.8

    nr = Solver_NR_Loading()
    nr.assembly = assembly
    node_num = node.coordinates_mat.shape[0]
    nr.supp = np.column_stack([np.arange(node_num), np.zeros(node_num), np.zeros(node_num), np.zeros(node_num)])
    for support_node in [1, 4, 3, 5]:
        nr.supp[support_node - 1, 1:4] = 1

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

    truss_stress = truss_strain * bar.E_vec

    fig1=plots.Plot_Shape_Bar_Stress(truss_stress,U_end)
    fig2=plots.Plot_Shape_Bar_Failure(pass_yn,U_end)
    
    U1=U_end[N*6-3, 2]
    U2=U_end[N*6-1, 2]
    
    tipDeflection=0.5*(U1+U2)
        
    return fig1, fig2, tipDeflection