import numpy as np

from AREMA_common import (
    check_scissor_members_arema,
    deck_weight,
    print_arema_local_screen,
    standard_scissor_deployment_coordinates,
)
from Solver_NR_Loading import Solver_NR_Loading
from scissor_common import bridge_self_weight, build_scissor_model


def scissor_deploy(secNum, dep_rate):
    model = build_scissor_model(variant="standard", analysis="load", N=secNum)
    standard_scissor_deployment_coordinates(model, dep_rate)
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
    node_num = model.node.coordinates_mat.shape[0]
    nr = Solver_NR_Loading()
    nr.assembly = model.assembly
    nr.supp = np.asarray([[0, 1, 1, 1], [1, 1, 1, 1], [2, 1, 1, 1], [3, 1, 1, 1]], dtype=float)

    U_end = truss_strain = pass_yn = None
    for step in range(1, 6):
        force = (W_bar + deck_weight()) / node_num / 5.0 * step
        nr.load = np.column_stack([np.arange(node_num), np.zeros(node_num), np.zeros(node_num), -force * np.ones(node_num)])
        nr.increStep = 1
        nr.iterMax = 50
        nr.tol = 1.0e-5
        U_end = nr.Solve()[-1]
        truss_strain, pass_yn, _ = check_scissor_members_arema(model, U_end, An, r_val, Fy, Fu)
        model.rot3.Solve_Global_Theta(model.node, U_end)
        moment_vec = np.abs(model.rot3.theta_current_vec - model.rot3.theta_stress_free_vec) * model.rot3.rot_spr_K_vec
        moment_capacity = 0.55 * Fy * Iy / 0.0762
        if not (bool(np.all(pass_yn)) and moment_capacity > float(np.max(moment_vec))):
            break

    model.plots.viewAngle1 = 10
    model.plots.viewAngle2 = -75
    truss_stress = truss_strain * model.bar.E_vec
    fig1 = model.plots.Plot_Shape_Bar_Stress(truss_stress, U_end)
    fig2 = model.plots.Plot_Shape_Bar_Failure(pass_yn, U_end)
    tip_deflection = 0.5 * (U_end[secNum * 8 + 1, 2] + U_end[secNum * 8 + 2, 2])
    return fig1, fig2, tip_deflection


scissor_deploy_arema = scissor_deploy
