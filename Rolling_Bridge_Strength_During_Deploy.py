import os
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


def rolling_deploy_offset(node_count, dep_rate,N):
    # deploy_path = os.path.abspath("RollingUhis.npy")

    Uhis = np.load("RollingUhis.npz")["Uhis"]
    print(Uhis.shape)
    
    dep_step = max(1, int((1.0 - dep_rate) * Uhis.shape[0]))
    idx = min(Uhis.shape[0], dep_step) - 1
    
    Uhis = Uhis[:,0:(N*6),:]
    
    print(f"Using rolling deployment step {idx + 1}/{Uhis.shape[0]}")
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


def rolling_deploy(N,dep_rate):
    H = 2.0
    W = 2.0
    L = 2.0

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
    plots.displayRange = np.array([-2.0, 18.0, -1.0, 3.0, -1.0, 14.0], dtype=float)
    node.coordinates_mat = node.coordinates_mat + rolling_deploy_offset(node.coordinates_mat.shape[0], dep_rate, N)
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
        truss_strain, pass_yn, dcr = check_members(bar, node, U_end, An, r_val, Fy, Fu, Rp)
        total_F = node_num * force
        safe = bool(np.all(pass_yn))
        history.append([step, total_F, float(np.nanmax(dcr)), 1.0 if safe else 0.0])
        print(f"Step {step:2d} : {'All Truss Members Safe' if safe else 'Member Failure Detected'} (AASHTO LRFD)")
        if not safe:
            break
        
    plots.view_angle1=10
    plots.view_angle2=-75

    truss_stress = truss_strain * bar.E_vec
    save_figure(plots.Plot_Shape_Bar_Stress(truss_stress,U_end), "Rolling_Bridge_Strength_During_Deploy_Bar_Stress.png")
    save_figure(plots.Plot_Shape_Bar_Failure(pass_yn,U_end), "Rolling_Bridge_Strength_During_Deploy_Bar_Failure.png")

    fig1=plots.Plot_Shape_Bar_Stress(truss_stress,U_end)
    fig2=plots.Plot_Shape_Bar_Failure(pass_yn,U_end)
    
    return fig1, fig2