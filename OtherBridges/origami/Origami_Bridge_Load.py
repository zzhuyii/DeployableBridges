import os
import time
import numpy as np
import matplotlib.pyplot as plt

from Elements_Nodes import Elements_Nodes
from Vec_Elements_CST import Vec_Elements_CST
from Vec_Elements_RotSprings_4N import Vec_Elements_RotSprings_4N
from Vec_Elements_Bars import Vec_Elements_Bars

from Plot_Origami import Plot_Origami
from Solver_NR_Loading import Solver_NR_Loading

from Assembly_Origami import Assembly_Origami


def check_truss_aisc(Ni, Ai, Ei, Lci, ri, Fy):
    if Ni > 0:
        Pn_i = Fy * Ai
        mode_str = "Tension-Yield"
    else:
        slender = Lci / ri
        Fe = (np.pi ** 2 * Ei) / (slender ** 2)
        lambda_lim = 4.71 * np.sqrt(Ei / Fy)
        if slender <= lambda_lim:
            Fcr = (0.658 ** (Fy / Fe)) * Fy
        else:
            Fcr = 0.877 * Fe
        Pn_i = Fcr * Ai
        mode_str = "Compression-Buckling"

    stress_ratio = abs(Ni) / abs(Pn_i) if Pn_i != 0 else np.inf
    passed = stress_ratio <= 1.0

    return passed, mode_str, Pn_i, stress_ratio


def plot_bar_failure(plots, pass_yn):
    assembly = plots.assembly
    node0 = assembly.node.coordinates_mat
    bar_conn = assembly.bar.node_ij_mat

    fig = plt.figure(figsize=(plots.width / plots.sizeFactor, plots.height / plots.sizeFactor))
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(plots.viewAngle1, plots.viewAngle2)
    ax.set_facecolor('white')

    vsize = plots.displayRange
    if isinstance(vsize, (list, tuple, np.ndarray)):
        ax.set_xlim(vsize[0], vsize[1])
        ax.set_ylim(vsize[2], vsize[3])
        ax.set_zlim(vsize[4], vsize[5])
    else:
        ax.set_xlim(-vsize * plots.displayRangeRatio, vsize)
        ax.set_ylim(-vsize * plots.displayRangeRatio, vsize)
        ax.set_zlim(-vsize * plots.displayRangeRatio, vsize)

    for j in range(bar_conn.shape[0]):
        n1, n2 = bar_conn[j]
        node1 = node0[n1 - 1]
        node2 = node0[n2 - 1]
        color = "green" if pass_yn[j] else "red"
        ax.plot([node1[0], node2[0]], [node1[1], node2[1]], [node1[2], node2[2]],
                color=color, linewidth=2)

    plt.gca().set_aspect('equal')
    plt.show()
    return fig


def main():
    print("RUNNING FILE:", __file__)

    # -----------------------------
    # Timing
    # -----------------------------
    t0 = time.time()

    # -----------------------------
    # Define Geometry
    # -----------------------------
    L = 2.0
    W = 4.0
    H = 2.0
    N = 4

    barA = 0.0023
    barE = 2.0e11
    I = 1.88e-6

    panel_E = 2.0e8
    panel_t = 0.01
    panel_v = 0.3

    # -----------------------------
    # Define assembly and elements
    # -----------------------------
    node = Elements_Nodes()
    assembly = Assembly_Origami()
    cst = Vec_Elements_CST()
    rot_spr = Vec_Elements_RotSprings_4N()
    bar = Vec_Elements_Bars()

    assembly.node = node
    assembly.cst = cst
    assembly.rotSpr = rot_spr
    assembly.bar = bar

    # -----------------------------
    # Define nodal coordinates
    # -----------------------------
    coords = []
    for i in range(1, N + 1):
        x0 = 2 * L * (i - 1)
        coords += [
            [x0, 0, H],
            [x0, 0, 0],
            [x0, W, 0],
            [x0, W, H],
            [x0 + L, 0, H],
            [x0 + L, 0, 0],
            [x0 + L, L, 0],
            [x0 + L, W, 0],
            [x0 + L, W, H],
        ]

    coords += [
        [2 * L * N, 0, H],
        [2 * L * N, 0, 0],
        [2 * L * N, W, 0],
        [2 * L * N, W, H],
    ]

    node.coordinates_mat = np.array(coords, dtype=float)

    # -----------------------------
    # Define plotting
    # -----------------------------
    plots = Plot_Origami()
    plots.assembly = assembly
    plots.displayRange = np.array([-1, 4 * (N + 1), -3, 7, -2, 3], dtype=float)
    plots.viewAngle1 = 20
    plots.viewAngle2 = 20

    # -----------------------------
    # Define CST triangles
    # -----------------------------
    tri = []
    for i in range(1, N + 1):
        base = 9 * (i - 1)
        tri += [
            [base + 2, base + 6, base + 7],
            [base + 2, base + 7, base + 3],
            [base + 3, base + 7, base + 8],
            [base + 6, base + 7, base + 11],
            [base + 7, base + 11, base + 12],
            [base + 7, base + 8, base + 12],
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
        base = 9 * (i - 1)
        bars += [
            [base + 2, base + 5],
            [base + 1, base + 2],
            [base + 1, base + 5],
            [base + 5, base + 11],
            [base + 5, base + 6],
            [base + 5, base + 10],
            [base + 10, base + 11],

            [base + 3, base + 9],
            [base + 4, base + 3],
            [base + 4, base + 9],
            [base + 9, base + 12],
            [base + 9, base + 8],
            [base + 9, base + 13],
            [base + 13, base + 12],

            [base + 2, base + 7],
            [base + 3, base + 7],
            [base + 6, base + 7],
            [base + 8, base + 7],
            [base + 11, base + 7],
            [base + 12, base + 7],
            [base + 2, base + 3],
            [base + 3, base + 8],
            [base + 8, base + 12],
            [base + 2, base + 6],
            [base + 6, base + 11],
        ]

    bars += [[9 * (N - 1) + 11, 9 * (N - 1) + 12]]

    bar.node_ij_mat = np.array(bars, dtype=int)
    bar_num = bar.node_ij_mat.shape[0]
    bar.A_vec = barA * np.ones(bar_num, dtype=float)
    bar.E_vec = barE * np.ones(bar_num, dtype=float)

    # -----------------------------
    # Define rotational springs (4-node)
    # -----------------------------
    spr4 = []
    for i in range(1, N + 1):
        base = 9 * (i - 1)
        spr4 += [
            [base + 5, base + 2, base + 6, base + 7],
            [base + 5, base + 6, base + 11, base + 7],
            [base + 2, base + 5, base + 6, base + 11],
            [base + 2, base + 6, base + 7, base + 11],

            [base + 9, base + 3, base + 8, base + 7],
            [base + 9, base + 8, base + 12, base + 7],
            [base + 3, base + 8, base + 9, base + 12],
            [base + 3, base + 7, base + 8, base + 12],

            [base + 2, base + 7, base + 3, base + 8],
            [base + 3, base + 7, base + 2, base + 6],
            [base + 6, base + 7, base + 11, base + 12],
            [base + 11, base + 7, base + 12, base + 8],

            [base + 1, base + 2, base + 5, base + 6],
            [base + 6, base + 5, base + 11, base + 10],
            [base + 4, base + 3, base + 9, base + 8],
            [base + 8, base + 9, base + 12, base + 13],
        ]

    for i in range(1, N):
        base = 9 * (i - 1)
        spr4 += [
            [base + 5, base + 10, base + 11, base + 14],
            [base + 9, base + 12, base + 13, base + 18],
            [base + 7, base + 11, base + 12, base + 16],
        ]

    rot_spr.node_ijkl_mat = np.array(spr4, dtype=int)
    rot_num = rot_spr.node_ijkl_mat.shape[0]
    rot_spr.rot_spr_K_vec = np.ones(rot_num, dtype=float) * 1.0e8

    # -----------------------------
    # Initialize assembly
    # -----------------------------
    assembly.Initialize_Assembly()

    # -----------------------------
    # Calculate self-weight of the bridge
    # -----------------------------
    rho_steel = 7850.0
    g = 9.81

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

    W_bar = barA * L_total * rho_steel * g

    # -----------------------------
    # Set up solver (distributed load)
    # -----------------------------
    nr = Solver_NR_Loading()
    nr.assembly = assembly

    nr.supp = [
        [2 - 1, 1, 1, 1],
        [3 - 1, 1, 1, 1],
        [N * 9 + 2 - 1, 1, 1, 1],
        [N * 9 + 3 - 1, 1, 1, 1],
    ]

    force = 10000.0

    Uhis = None
    total_F = 0.0
    pass_yn = None
    stress_ratio = None

    for i in range(1, 101):
        nr.load = []
        total_F = 0.0

        for k in range(1, N + 1):
            nr.load += [
                [6 + (k - 1) * 9 - 1, 0.0, 0.0, -force * i],
                [8 + (k - 1) * 9 - 1, 0.0, 0.0, -force * i],
                [11 + (k - 1) * 9 - 1, 0.0, 0.0, -force * i],
                [12 + (k - 1) * 9 - 1, 0.0, 0.0, -force * i],
            ]
            total_F += 4.0 * force * i

        nr.load = np.array(nr.load, dtype=float)
        nr.incre_step = 1
        nr.iter_max = 20
        nr.tol = 1e-5

        Uhis = nr.Solve()

        # Evaluate if member is failing
        U_end = Uhis[-1, :, :]

        truss_strain = bar.solve_strain(node, U_end)
        internal_force = truss_strain * bar.E_vec * bar.A_vec

        bar_num = internal_force.size
        L0_vec = bar.L0_vec.reshape(-1)

        K = 1.0
        Lc = K * L0_vec

        r = np.sqrt(I / barA) * np.ones(bar_num)
        r = np.maximum(r, 1e-9)

        Fy = 345e6

        pass_yn = np.zeros(bar_num, dtype=bool)
        stress_ratio = np.full(bar_num, np.nan, dtype=float)

        for k in range(bar_num):
            Ni = internal_force[k]
            Ai = bar.A_vec[k]
            Ei = bar.E_vec[k]
            Lci = Lc[k]
            ri = r[k]

            pass_i, _, _, stress_ratio_i = check_truss_aisc(Ni, Ai, Ei, Lci, ri, Fy)
            pass_yn[k] = pass_i
            stress_ratio[k] = stress_ratio_i

        if np.max(stress_ratio) < 1.0:
            print("All Truss Safe")
        else:
            print("Failure Detected")
            break

    # Find stiffness
    idx1 = 3 * N - 3
    idx2 = 3 * N - 1
    u_avg = -float(np.mean(Uhis[-1, [idx1 - 1, idx2 - 1], 2]))
    k_stiff = total_F / u_avg if abs(u_avg) > 1e-12 else np.inf

    print("-----------------------------")
    print(f"Total length of all bars: {L_total:.2f} m")
    print(f"Total bar weight: {W_bar:.2f} N")
    print(f"Failure load is: {total_F:.2f} N")
    print(f"Mid-span deflection at failure is: {u_avg:.3f} m")
    print(f"Stiffness is: {k_stiff:.2f} N/m")
    print(f"span/disp at failure is: {16 / u_avg:.2f}")
    print(f"capacity/weight: {total_F / W_bar:.2f}")
    print("-----------------------------")

    # Plot bar stress
    fig_stress = plots.Plot_Bar_Stress(Uhis[-1])
    try:
        out_dir = os.path.dirname(os.path.abspath(__file__))
        fig_stress.savefig(os.path.join(out_dir, "Origami_Bridge_Load_Bar_Stress.png"),
                           dpi=200, bbox_inches="tight")
        plt.close(fig_stress)
    except Exception as e:
        print("WARNING: failed to save bar stress figure:", repr(e))

    # Plot failed bar stress
    if pass_yn is not None:
        fig_fail = plot_bar_failure(plots, pass_yn)
        try:
            out_dir = os.path.dirname(os.path.abspath(__file__))
            fig_fail.savefig(os.path.join(out_dir, "Origami_Bridge_Load_Bar_Failure.png"),
                             dpi=200, bbox_inches="tight")
            plt.close(fig_fail)
        except Exception as e:
            print("WARNING: failed to save bar failure figure:", repr(e))

    print("Execution Time:", time.time() - t0)


if __name__ == "__main__":
    main()
