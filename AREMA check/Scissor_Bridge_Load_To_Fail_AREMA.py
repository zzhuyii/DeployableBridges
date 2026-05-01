import numpy as np

from AREMA_common import check_scissor_members_arema, print_arema_local_screen
from Solver_NR_Loading import Solver_NR_Loading
from scissor_common import bridge_self_weight, build_scissor_model, load_supports


def scissor_fail(secNum):
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
    nr = Solver_NR_Loading()
    nr.assembly = model.assembly
    nr.supp = load_supports(model.settings["N"], model.settings["stride"])

    force = 40000.0
    U_end = truss_strain = pass_yn = None
    total_F = 0.0
    capacity_F = 0.0
    for step in range(1, 101):
        load_rows = []
        total_F = 0.0
        for k in range(1, model.settings["N"]):
            n1 = model.settings["stride"] * k
            n2 = model.settings["stride"] * k + 1
            load_rows += [[n1, 0.0, 0.0, -force * step], [n2, 0.0, 0.0, -force * step]]
            total_F += force * 2.0 * step
        nr.load = np.asarray(load_rows, dtype=float)
        nr.increStep = 1
        nr.iterMax = 50
        nr.tol = 1.0e-5
        U_end = nr.Solve()[-1]
        truss_strain, pass_yn, _ = check_scissor_members_arema(model, U_end, An, r_val, Fy, Fu)
        model.rot3.Solve_Global_Theta(model.node, U_end)
        moment_vec = np.abs(model.rot3.theta_current_vec - model.rot3.theta_stress_free_vec) * model.rot3.rot_spr_K_vec
        moment_capacity = 0.55 * Fy * Iy / 0.0762
        if bool(np.all(pass_yn)) and moment_capacity > float(np.max(moment_vec)):
            capacity_F = total_F
        else:
            break

    model.plots.viewAngle1 = 10
    model.plots.viewAngle2 = -75
    truss_stress = truss_strain * model.bar.E_vec
    fig1 = model.plots.Plot_Shape_Bar_Stress(truss_stress, U_end)
    fig2 = model.plots.Plot_Shape_Bar_Failure(pass_yn, U_end)
    return fig1, fig2, capacity_F, W_bar


scissor_fail_arema = scissor_fail
