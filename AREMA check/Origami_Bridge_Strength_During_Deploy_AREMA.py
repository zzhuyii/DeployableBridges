import matplotlib
matplotlib.use("Agg")
import numpy as np

from AREMA_common import (
    bar_length_and_weight,
    check_bar_members_arema,
    deck_weight,
    npy_deployment_offset,
    print_arema_local_screen,
)
from Origami_Bridge_common import build_origami_bridge
from Solver_NR_Loading import Solver_NR_Loading


def origami_deploy(L, N, dep_rate):
    barA = 0.00415
    barE = 2.0e11
    Ix = 7.16e-6
    Fy = 345e6
    Fu = 427e6
    An = barA * 0.9
    r_val = np.sqrt(Ix / barA)
    print_arema_local_screen(barE, Fy)

    assembly, node, bar, cst, rot_spr_4N, plots = build_origami_bridge(
        L=L, W=2 * L, H=L, N=N, barA=barA, barE=barE,
        panel_E=2.0e8, panel_t=0.01, panel_v=0.3, rotK=1.0e8,
    )
    node.coordinates_mat = node.coordinates_mat + npy_deployment_offset(
        "OrigamiUhis.npy", node.coordinates_mat.shape[0], dep_rate, N * 9 + 4
    )
    assembly.Initialize_Assembly()
    _, W_bar = bar_length_and_weight(node, bar)

    nr = Solver_NR_Loading()
    nr.assembly = assembly
    nr.supp = np.asarray([[0, 1, 1, 1], [1, 1, 1, 1], [2, 1, 1, 1], [3, 1, 1, 1]], dtype=float)
    node_num = node.coordinates_mat.shape[0]
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

    plots.viewAngle1 = 10
    plots.viewAngle2 = -75
    truss_stress = truss_strain * bar.E_vec
    fig1 = plots.Plot_Bar_Stress(truss_stress, U_end)
    fig2 = plots.Plot_Shape_Bar_Failure(pass_yn, U_end)
    tip_deflection = 0.5 * (U_end[N * 9 + 1, 2] + U_end[N * 9 + 2, 2])
    return fig1, fig2, tip_deflection


origami_deploy_arema = origami_deploy
