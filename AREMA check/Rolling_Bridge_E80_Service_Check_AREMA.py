import matplotlib

matplotlib.use("Agg")
import numpy as np

from AREMA_E80_common import run_e80_service_check, station_groups_from_nodes
from AREMA_common import bar_length_and_weight, print_arema_local_screen
from Rolling_Bridge_common import build_rolling_bridge


def rolling_e80_service_arema(N, include_impact=True, train_step_m=None):
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
        H=H,
        W=W,
        L=L,
        N=N,
        barA=barA,
        barE=barE,
        panel_E=2.0e8,
        panel_t=0.01,
        panel_v=0.3,
        activeBarE=2.0e11,
        rotK=1.0e6,
        barA_brace=barA_brace,
    )
    assembly.Initialize_Assembly()
    _, W_bar = bar_length_and_weight(node, bar)

    load_nodes = []
    for k in range(1, N):
        load_nodes += [6 * (k - 1) + 3 - 1, 6 * (k - 1) + 5 - 1]
    load_station_groups = station_groups_from_nodes(node, load_nodes)
    node_num = node.coordinates_mat.shape[0]
    supports = np.column_stack([np.arange(node_num), np.zeros(node_num), np.zeros(node_num), np.zeros(node_num)])
    for support_node in [1, 4, N * 6 - 3, N * 6 - 1]:
        supports[support_node - 1, 1:4] = 1

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

    plots.view_angle1 = 10
    plots.view_angle2 = -75
    truss_stress = result["truss_strain"] * bar.E_vec
    fig1 = plots.Plot_Shape_Bar_Stress(truss_stress, result["U_end"])
    fig2 = plots.Plot_Shape_Bar_Failure(result["pass_yn"], result["U_end"])
    return fig1, fig2, result


rolling_e80_service = rolling_e80_service_arema
