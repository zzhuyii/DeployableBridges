import matplotlib
matplotlib.use("Agg")
import numpy as np

from AREMA_common import (
    bar_length_and_weight,
    check_bar_members_arema,
    deck_weight,
    print_arema_local_screen,
    rolling_deployment_offset,
)
from Rolling_Bridge_common import build_rolling_bridge
from Solver_NR_Loading import Solver_NR_Loading


def rolling_deploy(N, dep_rate):
    H = W = L = 2.0
    barA = 0.00415
    barE = 2.0e11
    barA_brace = 0.0019
    Ix = 7.16e-6
    Fy = 345e6
    Fu = 427e6
    An = barA * 0.9
    r_val = np.sqrt(Ix / barA)
    print_arema_local_screen(barE, Fy)

    assembly, node, bar, actBar, cst, rot_spr_4N, plots = build_rolling_bridge(
        H=H, W=W, L=L, N=N, barA=barA, barE=barE,
        panel_E=2.0e8, panel_t=0.01, panel_v=0.3,
        activeBarE=2.0e11, rotK=1.0e6, barA_brace=barA_brace,
    )
    plots.displayRange = np.array([-2.0, 18.0, -1.0, 3.0, -1.0, 14.0], dtype=float)
    node.coordinates_mat = node.coordinates_mat + rolling_deployment_offset(node.coordinates_mat.shape[0], dep_rate, N)
    assembly.Initialize_Assembly()
    _, W_bar = bar_length_and_weight(node, bar)

    nr = Solver_NR_Loading()
    nr.assembly = assembly
    node_num = node.coordinates_mat.shape[0]
    nr.supp = np.column_stack([np.arange(node_num), np.zeros(node_num), np.zeros(node_num), np.zeros(node_num)])
    for support_node in [1, 4, 3, 5]:
        nr.supp[support_node - 1, 1:4] = 1

    U_end = truss_strain = pass_yn = None
    for step in range(1, 6):
        force = (W_bar + deck_weight()) / node_num / 5.0 * step
        nr.load = np.column_stack([np.arange(node_num), np.zeros(node_num), np.zeros(node_num), -force * np.ones(node_num)])
        nr.increStep = 1
        nr.iterMax = 50
        nr.tol = 1.0e-5
        U_end = nr.Solve()[-1]
        truss_strain, pass_yn, _ = check_bar_members_arema(bar, node, U_end, An, r_val, Fy, Fu)
        if not bool(np.all(pass_yn)):
            break

    plots.view_angle1 = 10
    plots.view_angle2 = -75
    truss_stress = truss_strain * bar.E_vec
    fig1 = plots.Plot_Shape_Bar_Stress(truss_stress, U_end)
    fig2 = plots.Plot_Shape_Bar_Failure(pass_yn, U_end)
    tip_deflection = 0.5 * (U_end[N * 6 - 3, 2] + U_end[N * 6 - 1, 2])
    return fig1, fig2, tip_deflection


rolling_deploy_arema = rolling_deploy
