import matplotlib
matplotlib.use("Agg")
import numpy as np

from AREMA_common import bar_length_and_weight, check_bar_members_arema, print_arema_local_screen
from Origami_Bridge_common import build_origami_bridge
from Solver_NR_Loading import Solver_NR_Loading


def origami_fail(L, N):
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
    assembly.Initialize_Assembly()
    _, W_bar = bar_length_and_weight(node, bar)

    nr = Solver_NR_Loading()
    nr.assembly = assembly
    nr.supp = np.asarray([[1, 1, 1, 1], [2, 1, 1, 1], [N * 9 + 1, 1, 1, 1], [N * 9 + 2, 1, 1, 1]], dtype=float)

    force = 10000.0
    U_end = truss_strain = pass_yn = None
    total_F = 0.0
    capacity_F = 0.0
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
        U_end = nr.Solve()[-1]
        truss_strain, pass_yn, _ = check_bar_members_arema(bar, node, U_end, An, r_val, Fy, Fu)
        if bool(np.all(pass_yn)):
            capacity_F = total_F
        else:
            break

    plots.viewAngle1 = 10
    plots.viewAngle2 = -75
    truss_stress = truss_strain * bar.E_vec
    fig1 = plots.Plot_Bar_Stress(truss_stress, U_end)
    fig2 = plots.Plot_Shape_Bar_Failure(pass_yn, U_end)
    return fig1, fig2, capacity_F, W_bar


origami_fail_arema = origami_fail
