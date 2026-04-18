import os
import time

import numpy as np

from Solver_NR_Loading import Solver_NR_Loading
from scissor_common import (
    bridge_self_weight,
    build_scissor_model,
    check_truss_lrfd,
    local_buckling_message,
    save_figure,
)


OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def set_standard_deployment_coordinates(model, dep_rate):
    N = model.settings["N"]
    L = model.settings["L"]
    theta = np.pi / 4.0 * dep_rate
    dL = L * np.sqrt(2.0) * np.cos(theta) - L
    L2 = L / np.sqrt(2.0) * np.sin(theta)
    L3 = np.sqrt(max((L / 2.0) ** 2 - L2 ** 2, 0.0))
    coords = []
    for i in range(1, N + 1):
        x0 = 2.0 * L2 * (i - 1)
        xm = x0 + L2
        coords += [
            [x0, 0.0, 0.0], [x0, L, 0.0], [x0, 0.0, L + dL], [x0, L, L + dL],
            [xm, 0.0, (L + dL) / 2.0], [xm, L, (L + dL) / 2.0],
            [xm, 0.0, L3], [xm, L, L3],
        ]
    coords += [[2.0 * L2 * N, 0.0, 0.0], [2.0 * L2 * N, L, 0.0], [2.0 * L2 * N, 0.0, L + dL], [2.0 * L2 * N, L, L + dL]]
    model.node.coordinates_mat = np.asarray(coords, dtype=float)
    return dL


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


def write_summary(name, lines):
    path = os.path.join(OUT_DIR, name)
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Saved: {path}")


def main():
    start = time.time()
    dep_rate = 1.0

    model = build_scissor_model(variant="standard", analysis="load", N=8)
    dL = set_standard_deployment_coordinates(model, dep_rate)
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
    W_deck = 2.0 * (0.03 + 10.0 / 50.0 * 0.2) * 16.0 * 1000.0 * 9.8
    node_num = model.node.coordinates_mat.shape[0]

    nr = Solver_NR_Loading()
    nr.assembly = model.assembly
    nr.supp = np.asarray([[0, 1, 1, 1], [1, 1, 1, 1], [2, 1, 1, 1], [3, 1, 1, 1]], dtype=float)

    history = []
    Uhis = U_end = truss_strain = pass_yn = dcr = None
    total_F = 0.0
    final_step = 5
    for step in range(1, 6):
        force = (W_bar + W_deck) / node_num / 5.0 * step
        nr.load = np.column_stack([np.arange(node_num), np.zeros(node_num), np.zeros(node_num), -force * np.ones(node_num)])
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
        total_F = node_num * force
        safe = bool(np.all(pass_yn)) and moment_capacity > max_moment
        history.append([step, total_F, float(np.nanmax(dcr)), max_moment, moment_capacity, 1.0 if safe else 0.0])
        print(f"Step {step:2d} : {'All checks safe' if safe else 'Failure detected'}")
        if not safe:
            final_step = step
            break

    Uaverage = -float(np.mean(U_end[[67 - 1, 68 - 1], 2]))
    np.savetxt(
        os.path.join(OUT_DIR, "Scissor_Bridge_Strength_During_Deploy_Step_History.csv"),
        np.asarray(history), delimiter=",",
        header="step,total_load_N,max_DCR,max_moment_Nm,moment_capacity_Nm,all_checks_safe", comments="",
    )
    summary = [
        "Scissor_Bridge_Strength_During_Deploy",
        f"Deployment rate: {dep_rate:.3f}",
        f"dL: {dL:.6f} m",
        f"Final checked step: {final_step}",
        f"Total length of all bars: {L_total:.2f} m",
        f"Total bar weight: {W_bar:.2f} N",
        f"Deck weight: {W_deck:.2f} N",
        f"Maximum stress ratio: {np.nanmax(dcr):.3f}",
        f"Tip deflection: {Uaverage:.6f} m",
        f"Execution time: {time.time() - start:.2f} s",
    ]
    write_summary("Scissor_Bridge_Strength_During_Deploy_Summary.txt", summary)

    truss_stress = truss_strain * model.bar.E_vec
    save_figure(model.plots.Plot_Shape_Bar_Stress(truss_stress), os.path.join(OUT_DIR, "Scissor_Bridge_Strength_During_Deploy_Bar_Stress.png"))
    save_figure(model.plots.Plot_Shape_Bar_Failure(pass_yn), os.path.join(OUT_DIR, "Scissor_Bridge_Strength_During_Deploy_Bar_Failure.png"))
    save_figure(model.plots.Plot_Deformed_Shape(U_end), os.path.join(OUT_DIR, "Scissor_Bridge_Strength_During_Deploy_Deformed.png"))


if __name__ == "__main__":
    main()
