import os
import numpy as np
from Solver_NR_Loading import Solver_NR_Loading
from scissor_common import (
    bridge_self_weight,
    build_scissor_model,
    check_truss_lrfd,
    load_supports,
    local_buckling_message,
    save_figure,
)


OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def check_members(model, U_end, An, r_val, Fy, Fu, Rp):
    truss_strain = model.bar.solve_strain(model.node, U_end)
    internal_force = truss_strain * model.bar.E_vec * model.bar.A_vec
    Lc = model.bar.L0_vec.reshape(-1)
    pass_yn = np.zeros(internal_force.size, dtype=bool)
    dcr = np.full(internal_force.size, np.nan, dtype=float)
    for j, Pu in enumerate(1.5 * internal_force):
        passed, _, _, _, _, dcr_j = check_truss_lrfd(
            Pu, model.bar.A_vec[j], An, model.bar.E_vec[j], Lc[j], r_val, Fy, Fu, Rp
        )
        pass_yn[j] = passed
        dcr[j] = dcr_j
    return truss_strain, pass_yn, dcr

def improvedScissor_fail(secNum):

    model = build_scissor_model(variant="improved", analysis="load", N=secNum)
    model.assembly.Initialize_Assembly()

    Fy = 345e6
    Fu = 427e6
    E = model.settings["barE"]
    Ix = model.settings["Ix"]
    Iy = model.settings["Iy"]
    barA = model.settings["barA"]
    An = barA * 0.9
    Rp = 1.0
    r_val = np.sqrt(Ix / barA)

    _, lambda_r, buckling_status = local_buckling_message(E, Fy)
    print("--- Local Buckling Check (AASHTO LRFD Art. 6.9.4.2) ---")
    print(f"  lambda_r = {lambda_r:.2f}")
    print(f"  {buckling_status}")

    L_total, W_bar = bridge_self_weight(model)
    nr = Solver_NR_Loading()
    nr.assembly = model.assembly
    nr.supp = load_supports(model.settings["N"], model.settings["stride"])
    nr.verbose = False

    force = 20000.0
    history = []
    Uhis = U_end = truss_strain = pass_yn = dcr = None
    total_F = 0.0
    failed_step = 100
    failure_reason = "No failure through final step"

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
        Uhis = nr.Solve()
        U_end = Uhis[-1]
        truss_strain, pass_yn, dcr = check_members(model, U_end, An, r_val, Fy, Fu, Rp)

        model.rot3.Solve_Global_Theta(model.node, U_end)
        moment_vec = np.abs(model.rot3.theta_current_vec - model.rot3.theta_stress_free_vec) * model.rot3.rot_spr_K_vec
        max_moment = 1.5 * float(np.max(moment_vec))
        moment_capacity = Fy * Iy / 0.0762
        axial_safe = bool(np.all(pass_yn))
        moment_safe = bool(moment_capacity > max_moment)
        safe = axial_safe and moment_safe
        history.append([step, total_F, float(np.nanmax(dcr)), max_moment, moment_capacity, 1.0 if safe else 0.0])

        if axial_safe:
            print(f"Step {step:2d} : All Truss Members Safe (AASHTO LRFD)")
        else:
            print(f"Step {step:2d} : Member Failure Detected (AASHTO LRFD)")
            failed_step = step
            failure_reason = "Axial member LRFD failure"
            break
        if moment_safe:
            print(f"Step {step:2d} : Max Moment {max_moment:.2f} < Capacity {moment_capacity:.2f}")
        else:
            print(f"Step {step:2d} : Max Moment {max_moment:.2f} > Capacity {moment_capacity:.2f}")
            failed_step = step
            failure_reason = "Scissor bending moment failure"
            break


    model.plots.viewAngle1=10
    model.plots.viewAngle2=-75 

    truss_stress = truss_strain * model.bar.E_vec
    save_figure(model.plots.Plot_Shape_Bar_Stress(truss_stress,U_end), os.path.join(OUT_DIR, "Scissor_Bridge_2_Load_To_Fail_Bar_Stress.png"))
    save_figure(model.plots.Plot_Shape_Bar_Failure(pass_yn,U_end), os.path.join(OUT_DIR, "Scissor_Bridge_2_Load_To_Fail_Bar_Failure.png"))

    fig1=model.plots.Plot_Shape_Bar_Stress(truss_stress,U_end)
    fig2=model.plots.Plot_Shape_Bar_Failure(pass_yn,U_end)

    return fig1, fig2
