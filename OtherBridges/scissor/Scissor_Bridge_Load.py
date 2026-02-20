import os
import sys
import time
import numpy as np

from Elements_Nodes import Elements_Nodes
from Vec_Elements_CST import Vec_Elements_CST
from Vec_Elements_RotSprings_4N import Vec_Elements_RotSprings_4N
from Std_Elements_Bars import Std_Elements_Bars
from CD_Elements_RotSprings_3N import CD_Elements_RotSprings_3N

from Assembly_Scissor_Bridge import Assembly_Scissor_Bridge
from Plot_Scissor_Bridge import Plot_Scissor_Bridge


# Ensure project root is on sys.path for Solver_NR_Loading
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from Solver_NR_Loading import Solver_NR_Loading


def main():
    print("RUNNING FILE:", __file__)

    out_dir = os.path.dirname(os.path.abspath(__file__))

    # -----------------------------
    # Timing
    # -----------------------------
    t0 = time.time()

    # -----------------------------
    # Initialize the scissor
    # -----------------------------
    N = 2
    H = 2.0
    L = 2.0

    barA = 0.0023
    barE = 2.0e11

    panel_E = 2.0e11
    panel_t = 0.05
    panel_v = 0.3

    I = (1.0 / 12.0) * (0.01 ** 4)
    barL = np.sqrt(H ** 2 + L ** 2)
    kspr = 3.0 * barE * I / barL

    node = Elements_Nodes()

    # -----------------------------
    # Define nodal coordinates
    # -----------------------------
    coords = []
    for i in range(1, N + 1):
        x0 = L * (i - 1)

        coords += [
            [x0, 0, 0],
            [x0, L, 0],
            [x0, 0, L],
            [x0, L, L],

            [0.25 * L + x0, 0, 0.25 * L],
            [0.25 * L + x0, L, 0.25 * L],
            [0.25 * L + x0, 0, 0.75 * L],
            [0.25 * L + x0, L, 0.75 * L],

            [0.75 * L + x0, 0, 0.25 * L],
            [0.75 * L + x0, L, 0.25 * L],
            [0.75 * L + x0, 0, 0.75 * L],
            [0.75 * L + x0, L, 0.75 * L],

            [x0 + L / 2, 0, L / 2],
            [x0 + L / 2, L, L / 2],
            [x0 + L / 2, 0, 0],
            [x0 + L / 2, L, 0],

            [x0 + L / 2, 0, L],
            [x0 + L / 2, L, L],
            [x0 + L / 2, L / 2, L],
        ]

    coords += [
        [L * N, 0, 0],
        [L * N, L, 0],
        [L * N, 0, L],
        [L * N, L, L],
    ]

    node.coordinates_mat = np.array(coords, dtype=float)

    # -----------------------------
    # Define assembly and elements
    # -----------------------------
    assembly = Assembly_Scissor_Bridge()

    cst = Vec_Elements_CST()
    rotSpr3N = CD_Elements_RotSprings_3N()
    rotSpr4N = Vec_Elements_RotSprings_4N()
    bar = Std_Elements_Bars()

    assembly.node = node
    assembly.cst = cst
    assembly.bar = bar
    assembly.rotSpr3 = rotSpr3N
    assembly.rotSpr = rotSpr4N

    # -----------------------------
    # Plot object (optional)
    # -----------------------------
    plots = Plot_Scissor_Bridge()
    plots.assembly = assembly
    plots.displayRange = np.array([-0.5, 2 * N + 0.5, -0.5, 2.5, -0.5, 2.5], dtype=float)
    plots.viewAngle1 = 20
    plots.viewAngle2 = 20

    # -----------------------------
    # Define CST triangles
    # -----------------------------
    tri = []
    for i in range(1, N + 1):
        idx = 19 * (i - 1)
        tri += [
            [idx + 1, idx + 2, idx + 15],
            [idx + 2, idx + 15, idx + 16],
            [idx + 16, idx + 15, idx + 20],
            [idx + 20, idx + 16, idx + 21],
        ]

    cst.node_ijk_mat = np.array(tri, dtype=int)
    cst_num = cst.node_ijk_mat.shape[0]
    cst.t_vec = panel_t * np.ones(cst_num, dtype=float)
    cst.E_vec = panel_E * np.ones(cst_num, dtype=float)
    cst.v_vec = panel_v * np.ones(cst_num, dtype=float)

    # -----------------------------
    # Define bars
    # -----------------------------
    bars = []
    for i in range(1, N + 1):
        idx = 19 * (i - 1)
        bars += [
            [idx + 1, idx + 5],
            [idx + 5, idx + 13],
            [idx + 13, idx + 11],
            [idx + 11, idx + 22],

            [idx + 3, idx + 7],
            [idx + 7, idx + 13],
            [idx + 13, idx + 9],
            [idx + 9, idx + 20],

            [idx + 2, idx + 6],
            [idx + 6, idx + 14],
            [idx + 14, idx + 12],
            [idx + 12, idx + 23],

            [idx + 4, idx + 8],
            [idx + 8, idx + 14],
            [idx + 14, idx + 10],
            [idx + 10, idx + 21],

            [idx + 3, idx + 4],
            [idx + 3, idx + 17],
            [idx + 3, idx + 19],
            [idx + 4, idx + 19],
            [idx + 4, idx + 18],

            [idx + 17, idx + 19],
            [idx + 18, idx + 19],

            [idx + 18, idx + 23],
            [idx + 19, idx + 23],
            [idx + 17, idx + 22],
            [idx + 19, idx + 22],

            [idx + 1, idx + 2],
            [idx + 15, idx + 16],
            [idx + 1, idx + 15],
            [idx + 15, idx + 20],
            [idx + 2, idx + 16],
            [idx + 16, idx + 21],

            [idx + 2, idx + 15],
            [idx + 16, idx + 20],
        ]

    bars += [
        [19 * N + 3, 19 * N + 4],
        [19 * N + 1, 19 * N + 2],
    ]

    bar.node_ij_mat = np.array(bars, dtype=int)
    bar_num = bar.node_ij_mat.shape[0]
    bar.A_vec = barA * np.ones(bar_num, dtype=float)
    bar.E_vec = barE * np.ones(bar_num, dtype=float)

    # -----------------------------
    # Define 3-node rotational springs
    # -----------------------------
    spr3 = []
    for i in range(1, N + 1):
        idx = 19 * (i - 1)
        spr3 += [
            [idx + 1, idx + 5, idx + 13],
            [idx + 5, idx + 13, idx + 11],
            [idx + 13, idx + 11, idx + 22],

            [idx + 3, idx + 7, idx + 13],
            [idx + 7, idx + 13, idx + 9],
            [idx + 13, idx + 9, idx + 20],

            [idx + 2, idx + 6, idx + 14],
            [idx + 6, idx + 14, idx + 12],
            [idx + 14, idx + 12, idx + 23],

            [idx + 4, idx + 8, idx + 14],
            [idx + 8, idx + 14, idx + 10],
            [idx + 14, idx + 10, idx + 21],
        ]

    rotSpr3N.node_ijk_mat = np.array(spr3, dtype=int)
    rot_num_3 = rotSpr3N.node_ijk_mat.shape[0]
    rotSpr3N.rot_spr_K_vec = kspr * np.ones(rot_num_3, dtype=float) * 1000.0

    # -----------------------------
    # Define 4-node rotational springs
    # -----------------------------
    spr4 = []
    for i in range(1, N + 1):
        idx = 19 * (i - 1)
        spr4 += [
            [idx + 1, idx + 2, idx + 15, idx + 16],
            [idx + 2, idx + 15, idx + 16, idx + 20],
            [idx + 15, idx + 16, idx + 20, idx + 21],

            [idx + 3, idx + 17, idx + 19, idx + 22],
            [idx + 4, idx + 18, idx + 19, idx + 23],

            [idx + 3, idx + 4, idx + 19, idx + 18],
            [idx + 4, idx + 3, idx + 19, idx + 17],
            [idx + 18, idx + 19, idx + 23, idx + 22],
            [idx + 17, idx + 22, idx + 19, idx + 23],
        ]

    rotSpr4N.node_ijkl_mat = np.array(spr4, dtype=int)
    rot_num_4 = rotSpr4N.node_ijkl_mat.shape[0]
    rotSpr4N.rot_spr_K_vec = 100000.0 * np.ones(rot_num_4, dtype=float)

    # -----------------------------
    # Initialize assembly
    # -----------------------------
    assembly.Initialize_Assembly()

    # -----------------------------
    # Calculate self-weight
    # -----------------------------
    rho_steel = 7850.0
    g = 9.81

    # Bar elements
    A_bar = barA
    L_total = 0.0
    bar_node_mat = bar.node_ij_mat
    coords = node.coordinates_mat

    for i in range(bar_node_mat.shape[0]):
        n1 = int(bar_node_mat[i, 0]) - 1
        n2 = int(bar_node_mat[i, 1]) - 1
        p1 = coords[n1, :]
        p2 = coords[n2, :]
        length = np.linalg.norm(p1 - p2)
        L_total += length

    W_bar = A_bar * L_total * rho_steel * g

    # CST panels
    A_cst_total = 0.0
    cst_node_mat = cst.node_ijk_mat
    for i in range(cst_node_mat.shape[0]):
        n1 = int(cst_node_mat[i, 0]) - 1
        n2 = int(cst_node_mat[i, 1]) - 1
        n3 = int(cst_node_mat[i, 2]) - 1
        p1 = coords[n1, :]
        p2 = coords[n2, :]
        p3 = coords[n3, :]
        a = np.linalg.norm(p1 - p2)
        b = np.linalg.norm(p2 - p3)
        c = np.linalg.norm(p3 - p1)
        s = (a + b + c) / 2.0
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
        A_cst_total += area

    W_cst = A_cst_total * panel_t * rho_steel * g
    W_total = W_bar + W_cst

    print("-----------------------------")
    print(f"Total length of all bars: {L_total:.3f} m")
    print(f"Total area of all CST panels: {A_cst_total:.3f} m^2")
    print(f"Total bar weight: {W_bar:.2f} N")
    print(f"Total CST panel weight: {W_cst:.2f} N")
    print(f"Total self-weight of the bridge: {W_total:.2f} N")
    print("-----------------------------")

    # -----------------------------
    # Set up solver (distributed load)
    # -----------------------------
    nr = Solver_NR_Loading()
    nr.assembly = assembly

    nr.supp = [
        [1 - 1, 1, 1, 1],
        [2 - 1, 1, 1, 1],
        [3 - 1, 1, 1, 1],
        [4 - 1, 1, 1, 1],
        [19 * N + 1 - 1, 1, 1, 1],
        [19 * N + 2 - 1, 1, 1, 1],
        [19 * N + 3 - 1, 1, 1, 1],
        [19 * N + 4 - 1, 1, 1, 1],
    ]

    P_total = 10000.0
    step = 20

    load_list = []
    for i in range(1, N):
        n1 = 20 + (i - 1) * 19
        n2 = 21 + (i - 1) * 19
        load_list += [
            [n1 - 1, 0.0, 0.0, -P_total / 2.0 / (N - 1)],
            [n2 - 1, 0.0, 0.0, -P_total / 2.0 / (N - 1)],
        ]

    nr.load = np.array(load_list, dtype=float)
    nr.incre_step = step
    nr.iter_max = 30
    nr.tol = 1e-4

    Uhis = nr.Solve()

    # -----------------------------
    # Post-process
    # -----------------------------
    U_end = Uhis[-1, :, :]
    fig_def = plots.Plot_Deformed_Shape(U_end * 50.0)
    try:
        import matplotlib.pyplot as plt
        fig_def.savefig(os.path.join(out_dir, "Scissor_Bridge_Load_Deformed.png"),
                        dpi=200, bbox_inches="tight")
        plt.close(fig_def)
    except Exception as e:
        print("WARNING: failed to save deformed shape figure:", repr(e))

    truss_strain = bar.Solve_Strain(node, U_end)
    internal_force = truss_strain * bar.E_vec * bar.A_vec
    bar_stress = truss_strain * bar.E_vec

    max_bar_force = float(np.max(np.abs(internal_force)))

    sigma_u = 300.0e6
    bar_failure_force = sigma_u * barA

    bar_total_length = float(np.sum(bar.L0_vec))

    idx_base = int(38 * N / 4)
    u_avg = -float(np.mean(U_end[[idx_base, idx_base + 1], 2]))
    k_stiff = step * P_total / u_avg if abs(u_avg) > 1e-12 else np.inf

    load_at_fail = step * P_total * bar_failure_force / max_bar_force

    print(f"Failure load is {load_at_fail / 1000.0:.3f} kN")
    print(f"Total bar length is {bar_total_length:.3f} m")
    print(f"Stiffness is {k_stiff:.3f} N/m")
    print(f"Total bars: {bar_num}")

    load_eff = load_at_fail / W_bar if W_bar != 0 else np.inf
    print(f"Load efficiency (FailureLoad/SelfWeight) = {load_eff:.3f}")

    fig_stress = plots.Plot_Shape_Bar_Stress(bar_stress)
    try:
        import matplotlib.pyplot as plt
        fig_stress.savefig(os.path.join(out_dir, "Scissor_Bridge_Load_Bar_Stress.png"),
                           dpi=200, bbox_inches="tight")
        plt.close(fig_stress)
    except Exception as e:
        print("WARNING: failed to save bar stress figure:", repr(e))

    # -----------------------------
    # Evaluate members
    # -----------------------------
    axial_force = internal_force.reshape(-1)
    A = bar.A_vec.reshape(-1)
    E = bar.E_vec.reshape(-1)
    nb = A.size

    if not hasattr(bar, "L0_vec") or bar.L0_vec is None or bar.L0_vec.size == 0:
        L0_vec = np.zeros(nb, dtype=float)
        for k in range(nb):
            n1 = int(bar.node_ij_mat[k, 0]) - 1
            n2 = int(bar.node_ij_mat[k, 1]) - 1
            L0_vec[k] = np.linalg.norm(assembly.node.coordinates_mat[n1, :] -
                                       assembly.node.coordinates_mat[n2, :])
    else:
        L0_vec = bar.L0_vec.reshape(-1)

    K = 1.0
    Lc = K * L0_vec

    r = 0.5 * np.sqrt(A / np.pi)
    r = np.maximum(r, 1e-9)

    Fy = 345e6

    pass_yn = np.zeros(nb, dtype=bool)
    util = np.full(nb, np.nan, dtype=float)
    mode_str = [""] * nb
    Pn = np.full(nb, np.nan, dtype=float)

    for i in range(nb):
        Ni = axial_force[i]
        Ai = A[i]
        Ei = E[i]
        Lci = Lc[i]
        ri = r[i]

        if Ni > 0:
            Pn_i = Fy * Ai
            mode_str[i] = "Tension-Yield"
        else:
            slender = Lci / ri
            Fe = (np.pi ** 2 * Ei) / (slender ** 2)
            lambda_lim = 4.71 * np.sqrt(Ei / Fy)
            if slender <= lambda_lim:
                Fcr = (0.658 ** (Fy / Fe)) * Fy
            else:
                Fcr = 0.877 * Fe
            Pn_i = Fcr * Ai
            mode_str[i] = "Compression-Buckling"

        Pn[i] = Pn_i
        util[i] = abs(Ni) / max(Pn_i, 1e-12)
        pass_yn[i] = util[i] <= 1.0

    print(f"Bars passed: {int(np.sum(pass_yn))} / {nb}")

    idx_sorted = np.argsort(util)[::-1]
    topk = min(10, nb)
    print(f"Worst {topk} bars (by utilization):")
    for ii in range(topk):
        b = int(idx_sorted[ii])
        print(
            f"#{b + 1}: N={axial_force[b] / 1e3:.2f} kN, "
            f"util={util[b]:.3f}, mode={mode_str[b]}, "
            f"Pn={Pn[b] / 1e3:.2f} kN"
        )

    print("Execution Time:", time.time() - t0)


if __name__ == "__main__":
    main()
