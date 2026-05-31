import os
import numpy as np
from Solver_NR_Loading import Solver_NR_Loading
from scissor_common import bridge_self_weight,build_scissor1_model,check_members

OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def scissor_fail(secNum, Lb, designCode):

    assembly, node, bar, act_bar, cst, rot3, rot4, plots = build_scissor1_model(N=secNum,L=Lb)
    assembly.Initialize_Assembly()
    
    Fy = 345e6
    Fu = 427e6
    E = 2.0e11
    Ix = 7.16e-6
    Iy = 7.16e-6
    barA = 0.00415
    An = barA * 0.9
    Rp = 1.0
    r_val = np.sqrt(Ix / barA)


    L_total, W_bar = bridge_self_weight(node, bar)
    nr = Solver_NR_Loading()
    nr.assembly = assembly
    
    endNode= secNum * 8
    
    nr.supp = np.asarray([
        [0,     1, 1, 1],
        [1,     1, 1, 1],
        [endNode,   0, 1, 1],
        [endNode+1, 0, 1, 1],
    ], dtype=float)
    
    
    nr.verbose = False

    force = 100000.0
    history = []
    Uhis = U_end = truss_strain = pass_yn = dcr = None
    total_F = 0.0

    for step in range(1, 101):
        load_rows = []
        total_F = 0.0
        for k in range(1, secNum):
            n1 = 8 * k
            n2 = 8 * k + 1
            load_rows += [[n1, 0.0, 0.0, -force * step], [n2, 0.0, 0.0, -force * step]]
            total_F += force * 2.0 * step

        nr.load = np.asarray(load_rows, dtype=float)
        nr.increStep = 1
        nr.iterMax = 50
        nr.tol = 1.0e-5
        Uhis = nr.Solve()
        U_end = Uhis[-1]
        truss_strain, pass_yn, dcr = check_members(bar, node, U_end, An, r_val, Fy, Fu, Rp, designCode)

        rot3.Solve_Global_Theta(node, U_end)
        moment_vec = np.abs(rot3.theta_current_vec - rot3.theta_stress_free_vec) * rot3.rot_spr_K_vec
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
            break



    plots.viewAngle1=10
    plots.viewAngle2=-75 

    truss_stress = truss_strain * bar.E_vec

    fig1=plots.Plot_Shape_Bar_Stress(truss_stress,U_end)
    fig2=plots.Plot_Shape_Bar_Failure(pass_yn,U_end)
    
    print(pass_yn)
    print(dcr)
    
    return fig1,fig2, total_F, W_bar
