import matplotlib

matplotlib.use("Agg")
import numpy as np

from AREMA_E80_common import run_e80_service_check, station_groups_from_nodes
from AREMA_common import bar_length_and_weight, print_arema_local_screen
from Origami_Bridge_common import build_origami_bridge


def origami_e80_service_arema(L, N, include_impact=True, train_step_m=None):
    barA = 0.00415
    barE = 2.0e11
    Ix = 7.16e-6
    Fy = 345e6
    Fu = 427e6
    An = barA * 0.9
    r_val = np.sqrt(Ix / barA)
    print_arema_local_screen(barE, Fy)

    assembly, node, bar, cst, rot_spr_4N, plots = build_origami_bridge(
        L=L,
        W=2 * L,
        H=L,
        N=N,
        barA=barA,
        barE=barE,
        panel_E=2.0e8,
        panel_t=0.01,
        panel_v=0.3,
        rotK=1.0e8,
    )
    assembly.Initialize_Assembly()
    _, W_bar = bar_length_and_weight(node, bar)

    load_nodes = []
    for k in range(1, N + 1):
        b = 9 * (k - 1)
        node_ids = [b + 6, b + 8, b + 11, b + 12] if k < N else [b + 6, b + 8]
        load_nodes += [node_id - 1 for node_id in node_ids]
    load_station_groups = station_groups_from_nodes(node, load_nodes)
    supports = np.asarray([[1, 1, 1, 1], [2, 1, 1, 1], [N * 9 + 1, 1, 1, 1], [N * 9 + 2, 1, 1, 1]], dtype=float)

    result = run_e80_service_check(
        assembly,
        node,
        bar,
        load_station_groups,
        supports,
        An,
        r_val,
        Fy,
        Fu,
        W_bar,
        include_impact=include_impact,
        train_step_m=train_step_m,
    )

    plots.viewAngle1 = 10
    plots.viewAngle2 = -75
    truss_stress = result["truss_strain"] * bar.E_vec
    fig1 = plots.Plot_Bar_Stress(truss_stress, result["U_end"])
    fig2 = plots.Plot_Shape_Bar_Failure(result["pass_yn"], result["U_end"])
    return fig1, fig2, result


origami_e80_service = origami_e80_service_arema
