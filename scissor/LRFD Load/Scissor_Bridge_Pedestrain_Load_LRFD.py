import os
import time

import numpy as np

from Solver_NR_Loading import Solver_NR_Loading
from scissor_common import (
    bridge_self_weight,
    build_scissor_model,
    check_truss_lrfd,
    load_supports,
    local_buckling_message,
    save_figure,
    save_geometry_diagnostics,
)


def main():
    print("RUNNING FILE:", __file__)
    t0 = time.time()
    out_dir = os.path.dirname(os.path.abspath(__file__))

    model = build_scissor_model(variant="standard", analysis="load", N=8)
    model.assembly.Initialize_Assembly()
    save_geometry_diagnostics(model, out_dir, "Scissor_Bridge_Load")

    Fy = 345e6
    Fu = 427e6
    E = model.settings["barE"]
    Ix = model.settings["Ix"]
    Iy = model.settings["Iy"]
    barA = model.settings["barA"]
    An = barA * 0.9
    Rp = 1.0
    r_val = np.sqrt(Ix / barA)
    K = 1.0

    _, lambda_r, buckling_status = local_buckling_message(E, Fy)
    print("--- Local Buckling Check (AASHTO LRFD Art. 6.9.4.2) ---")
    print(f"  lambda_r = {lambda_r:.2f}")
    print(f"  {buckling_status}")

    L_total, W_bar = bridge_self_weight(model)
    W_deck = 2 * (0.03 + 10 / 50 * 0.2) * 16 * 1000 * 9.8
    qPL = 3.6e3
    W_LL = qPL * 16 * 2
    W_factored = 1.25 * (W_bar + W_deck) + 1.75 * W_LL

    nr = Solver_NR_Loading()
    nr.assembly = model.assembly
    nr.supp = load_supports(model.settings["N"], model.settings["stride"])
    nr.verbose = False

    force = W_factored / 14 / 5
    Uhis = None
    U_end = None
    total_F = 0.0
    passYN = None
    DCR = None
    truss_strain = None

    for load_step in range(1, 6):
        nr.incre_step = 1
        nr.iter_max = 50
        nr.tol = 1e-5

        load_rows = []
        total_F = 0.0
        for k in range(1, model.settings["N"]):
            n1 = model.settings["stride"] * k
            n2 = model.settings["stride"] * k + 1
            load_rows += [[n1, 0.0, 0.0, -force * load_step], [n2, 0.0, 0.0, -force * load_step]]
            total_F += force * 2 * load_step
        nr.load = np.asarray(load_rows, dtype=float)

        Uhis = nr.Solve()
        U_end = Uhis[-1, :, :]
        truss_strain = model.bar.solve_strain(model.node, U_end)
        internal_force = truss_strain * model.bar.E_vec * model.bar.A_vec

        Lc = K * model.bar.L0_vec.reshape(-1)
        bar_num = internal_force.size
        passYN = np.zeros(bar_num, dtype=bool)
        DCR = np.full(bar_num, np.nan, dtype=float)
        mode_str = []

        for bar_id in range(bar_num):
            Pu = 1.5 * internal_force[bar_id]
            passed, mode, _, _, _, dcr = check_truss_lrfd(
                Pu, model.bar.A_vec[bar_id], An, model.bar.E_vec[bar_id],
                Lc[bar_id], r_val, Fy, Fu, Rp,
            )
            passYN[bar_id] = passed
            DCR[bar_id] = dcr
            mode_str.append(mode)

        if np.all(passYN):
            print(f"Step {load_step:2d}/5 : All Truss Members Safe (AASHTO LRFD)")
        else:
            print(f"Step {load_step:2d}/5 : Member Failure Detected (AASHTO LRFD)")
            break

        model.rot3.Solve_Global_Theta(model.node, U_end)
        moment_vec = np.abs(model.rot3.theta_current_vec - model.rot3.theta_stress_free_vec) * model.rot3.rot_spr_K_vec
        max_moment = 1.5 * float(np.max(moment_vec))
        moment_capacity = Fy * Iy / 0.0762
        if moment_capacity > max_moment:
            print(f"Step {load_step:2d} : Max Moment {max_moment:.2f} < Capacity {moment_capacity:.2f}")
        else:
            print(f"Step {load_step:2d} : Max Moment {max_moment:.2f} > Capacity {moment_capacity:.2f}")
            break

    if Uhis is None or U_end is None or passYN is None or DCR is None or truss_strain is None:
        raise RuntimeError("Loading analysis did not produce results.")

    mid_nodes = [3 * model.settings["N"] - 4, 3 * model.settings["N"] - 2]
    Uaverage = -float(np.mean(Uhis[-1, mid_nodes, 2]))
    Kstiff = total_F / Uaverage if abs(Uaverage) > 1e-12 else np.inf

    print("-----------------------------")
    print(f"Total length of all bars: {L_total:.2f} m")
    print(f"Total bar weight: {W_bar:.2f} N")
    print(f"Total load is: {total_F:.2f} N")
    print(f"Mid-span deflection at Strength limit state is: {Uaverage:.3f} m")
    print(f"Stiffness is: {Kstiff:.2f} N/m")
    print(f"span/disp at Strength limit state is: {16 / Uaverage:.2f}")
    print(f"Maximum DCR: {np.nanmax(DCR):.2f}")
    print("-----------------------------")

    truss_stress = truss_strain * model.bar.E_vec
    save_figure(model.plots.Plot_Shape_Bar_Stress(truss_stress),
                os.path.join(out_dir, "Scissor_Bridge_Load_Bar_Stress.png"))
    save_figure(model.plots.Plot_Shape_Bar_Failure(passYN),
                os.path.join(out_dir, "Scissor_Bridge_Load_Bar_Failure.png"))
    save_figure(model.plots.Plot_Deformed_Shape(U_end),
                os.path.join(out_dir, "Scissor_Bridge_Load_Deformed.png"))
    save_figure(model.plots.Plot_Deformed_Shape(U_end),
                os.path.join(out_dir, "Scissor_Bridge_Fully_Deployed_Load_Deformed.png"))

    print("Execution Time:", time.time() - t0)


if __name__ == "__main__":
    main()
