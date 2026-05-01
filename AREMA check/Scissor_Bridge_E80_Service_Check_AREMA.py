import numpy as np

from AREMA_E80_common import run_e80_service_check, station_groups_from_nodes
from AREMA_common import check_scissor_members_arema, print_arema_local_screen
from scissor_common import bridge_self_weight, build_scissor_model, load_supports


def scissor_e80_service_arema(secNum, include_impact=True, train_step_m=None):
    model = build_scissor_model(variant="standard", analysis="load", N=secNum)
    model.assembly.Initialize_Assembly()

    Fy = 345e6
    Fu = 427e6
    E = model.settings["barE"]
    Ix = model.settings["Ix"]
    Iy = model.settings["Iy"]
    barA = model.settings["barA"]
    An = barA * 0.9
    r_val = np.sqrt(Ix / barA)
    print_arema_local_screen(E, Fy)

    _, W_bar = bridge_self_weight(model)
    load_nodes = []
    for k in range(1, model.settings["N"]):
        load_nodes += [model.settings["stride"] * k, model.settings["stride"] * k + 1]
    load_station_groups = station_groups_from_nodes(model.node, load_nodes)
    supports = load_supports(model.settings["N"], model.settings["stride"])

    def extra_check(U_end):
        model.rot3.Solve_Global_Theta(model.node, U_end)
        moment_vec = np.abs(model.rot3.theta_current_vec - model.rot3.theta_stress_free_vec) * model.rot3.rot_spr_K_vec
        max_moment = float(np.max(moment_vec))
        moment_capacity = 0.55 * Fy * Iy / 0.0762
        dcr = max_moment / max(moment_capacity, 1e-12)
        return dcr <= 1.0, {"dcr": dcr, "max_moment": max_moment, "moment_capacity": moment_capacity}

    result = run_e80_service_check(
        model.assembly,
        model.node,
        model.bar,
        load_station_groups,
        supports,
        An,
        r_val,
        Fy,
        Fu,
        W_bar,
        include_impact=include_impact,
        train_step_m=train_step_m,
        extra_check=extra_check,
    )

    model.plots.viewAngle1 = 10
    model.plots.viewAngle2 = -75
    truss_stress = result["truss_strain"] * model.bar.E_vec
    fig1 = model.plots.Plot_Shape_Bar_Stress(truss_stress, result["U_end"])
    fig2 = model.plots.Plot_Shape_Bar_Failure(result["pass_yn"], result["U_end"])
    return fig1, fig2, result


scissor_e80_service = scissor_e80_service_arema
