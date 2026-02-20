import os
import sys
import time
import numpy as np

# Ensure project root is on sys.path for shared element classes
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from Elements_Nodes import Elements_Nodes
from Vec_Elements_Bars import Vec_Elements_Bars
from Std_Elements_Bars import Std_Elements_Bars
from Vec_Elements_CST import Vec_Elements_CST
from Solver_NR_Loading import Solver_NR_Loading

from Assembly_Rolling_Bridge import Assembly_Rolling_Bridge
from Plot_Rolling_Bridge import Plot_Rolling_Bridge


def main():
    print("RUNNING FILE:", __file__)
    t0 = time.time()
    out_dir = os.path.dirname(os.path.abspath(__file__))

    # -----------------------------
    # Define Geometry (Rolling Bridge)
    # -----------------------------
    H = 2.0
    HA = 2.0
    W = 2.0
    L = 2.0
    l = 0.3
    N = 2

    barA = 0.0023
    barE = 2.0e11
    panel_E = 2.0e11
    panel_t = 0.05
    panel_v = 0.3
    activeBarE = 2.0e11

    node = Elements_Nodes()
    bar = Vec_Elements_Bars()
    actBar = Std_Elements_Bars()
    cst = Vec_Elements_CST()

    assembly = Assembly_Rolling_Bridge()
    assembly.node = node
    assembly.bar = bar
    assembly.actBar = actBar
    assembly.cst = cst

    # -----------------------------
    # Define nodal coordinates
    # -----------------------------
    coords = [
        [0, 0, 0],
        [L / 2, 0, H],
        [L, 0, 0],
        [0, W, 0],
        [L, W, 0],
        [L / 2, W, H],
        [L, 0, H],
        [L, W, H],
    ]

    for i in range(2, N):
        coords += [
            [L * i, 0, 0],
            [L * i - L / 2, 0, H],
            [L * i, W, 0],
            [L * i - L / 2, W, H],
            [L * i, 0, H],
            [L * i, W, H],
        ]

    coords += [
        [L * N, 0, 0],
        [L * N - L / 2, 0, H],
        [L * N, W, 0],
        [L * N - L / 2, W, H],
    ]

    node.coordinates_mat = np.array(coords, dtype=float)

    # -----------------------------
    # Plot setup (inspection)
    # -----------------------------
    plots = Plot_Rolling_Bridge()
    plots.assembly = assembly
    plots.displayRange = np.array([-0.5, 2 * N + 0.5, -0.5, 2.5, -0.5, 2.5], dtype=float)
    plots.viewAngle1 = 20
    plots.viewAngle2 = 20
    fig_node = plots.Plot_Shape_Node_Number()
    try:
        import matplotlib.pyplot as plt
        fig_node.savefig(os.path.join(out_dir, "Rolling_Bridge_Load_Node_Number.png"),
                         dpi=200, bbox_inches="tight")
        plt.close(fig_node)
    except Exception as e:
        print("WARNING: failed to save node number figure:", repr(e))

    # -----------------------------
    # Define CST panels
    # -----------------------------
    tri = [
        [1, 3, 4],
        [3, 4, 5],
    ]
    for i in range(2, N + 1):
        tri += [
            [3 + (i - 2) * 6, 5 + (i - 2) * 6, 9 + (i - 2) * 6],
            [5 + (i - 2) * 6, 9 + (i - 2) * 6, 11 + (i - 2) * 6],
        ]

    cst.node_ijk_mat = np.array(tri, dtype=int)
    cst_num = cst.node_ijk_mat.shape[0]
    cst.v_vec = panel_v * np.ones(cst_num, dtype=float)
    cst.E_vec = panel_E * np.ones(cst_num, dtype=float)
    cst.t_vec = panel_t * np.ones(cst_num, dtype=float)
    fig_cst = plots.Plot_Shape_CST_Number()
    try:
        import matplotlib.pyplot as plt
        fig_cst.savefig(os.path.join(out_dir, "Rolling_Bridge_Load_CST_Number.png"),
                        dpi=200, bbox_inches="tight")
        plt.close(fig_cst)
    except Exception as e:
        print("WARNING: failed to save CST number figure:", repr(e))

    # -----------------------------
    # Define normal bars
    # -----------------------------
    bars = [
        [1, 2],
        [1, 3],
        [2, 3],
        [4, 5],
        [4, 6],
        [5, 6],
        [2, 7],
        [6, 8],
        [1, 4],
        [3, 5],
        [3, 4],
    ]

    for i in range(2, N):
        bars += [
            [3 + (i - 2) * 6, 9 + (i - 2) * 6],
            [3 + (i - 2) * 6, 10 + (i - 2) * 6],
            [9 + (i - 2) * 6, 10 + (i - 2) * 6],
            [5 + (i - 2) * 6, 11 + (i - 2) * 6],
            [5 + (i - 2) * 6, 12 + (i - 2) * 6],
            [11 + (i - 2) * 6, 12 + (i - 2) * 6],
            [7 + (i - 2) * 6, 10 + (i - 2) * 6],
            [8 + (i - 2) * 6, 12 + (i - 2) * 6],
            [10 + (i - 2) * 6, 13 + (i - 2) * 6],
            [12 + (i - 2) * 6, 14 + (i - 2) * 6],
            [9 + (i - 2) * 6, 11 + (i - 2) * 6],
            [5 + (i - 2) * 6, 9 + (i - 2) * 6],
        ]

    bars += [
        [3 + (N - 2) * 6, 9 + (N - 2) * 6],
        [3 + (N - 2) * 6, 10 + (N - 2) * 6],
        [9 + (N - 2) * 6, 10 + (N - 2) * 6],
        [5 + (N - 2) * 6, 11 + (N - 2) * 6],
        [5 + (N - 2) * 6, 12 + (N - 2) * 6],
        [11 + (N - 2) * 6, 12 + (N - 2) * 6],
        [7 + (N - 2) * 6, 10 + (N - 2) * 6],
        [8 + (N - 2) * 6, 12 + (N - 2) * 6],
        [5 + (N - 2) * 6, 9 + (N - 2) * 6],
        [9 + (N - 2) * 6, 11 + (N - 2) * 6],
    ]

    bar.node_ij_mat = np.array(bars, dtype=int)
    bar_num = bar.node_ij_mat.shape[0]
    bar.A_vec = barA * np.ones(bar_num, dtype=float)
    bar.E_vec = barE * np.ones(bar_num, dtype=float)
    fig_bar = plots.Plot_Shape_Bar_Number()
    try:
        import matplotlib.pyplot as plt
        fig_bar.savefig(os.path.join(out_dir, "Rolling_Bridge_Load_Bar_Number.png"),
                        dpi=200, bbox_inches="tight")
        plt.close(fig_bar)
    except Exception as e:
        print("WARNING: failed to save bar number figure:", repr(e))

    # -----------------------------
    # Define active bars
    # -----------------------------
    act = []
    for i in range(1, N):
        act += [
            [3 + (i - 1) * 6, 7 + (i - 1) * 6],
            [5 + (i - 1) * 6, 8 + (i - 1) * 6],
        ]

    actBar.node_ij_mat = np.array(act, dtype=int)
    act_num = actBar.node_ij_mat.shape[0]
    actBar.A_vec = barA * np.ones(act_num, dtype=float)
    actBar.E_vec = activeBarE * np.ones(act_num, dtype=float)
    fig_act = plots.Plot_Shape_ActBar_Number()
    try:
        import matplotlib.pyplot as plt
        fig_act.savefig(os.path.join(out_dir, "Rolling_Bridge_Load_ActBar_Number.png"),
                        dpi=200, bbox_inches="tight")
        plt.close(fig_act)
    except Exception as e:
        print("WARNING: failed to save act bar number figure:", repr(e))

    # Initialize the entire assembly
    assembly.Initialize_Assembly()

    # -----------------------------
    # Calculate self-weight
    # -----------------------------
    rho_steel = 7850.0
    g = 9.81

    coords_np = node.coordinates_mat

    L_bar_total = 0.0
    for i in range(bar.node_ij_mat.shape[0]):
        n1 = int(bar.node_ij_mat[i, 0]) - 1
        n2 = int(bar.node_ij_mat[i, 1]) - 1
        L_bar_total += np.linalg.norm(coords_np[n1] - coords_np[n2])

    L_act_total = 0.0
    for i in range(actBar.node_ij_mat.shape[0]):
        n1 = int(actBar.node_ij_mat[i, 0]) - 1
        n2 = int(actBar.node_ij_mat[i, 1]) - 1
        L_act_total += np.linalg.norm(coords_np[n1] - coords_np[n2])

    L_total_bars = L_bar_total + L_act_total
    W_bar = barA * L_total_bars * rho_steel * g

    A_cst_total = 0.0
    for i in range(cst.node_ijk_mat.shape[0]):
        n1, n2, n3 = cst.node_ijk_mat[i] - 1
        p1 = coords_np[n1]
        p2 = coords_np[n2]
        p3 = coords_np[n3]
        a = np.linalg.norm(p1 - p2)
        b = np.linalg.norm(p2 - p3)
        c = np.linalg.norm(p3 - p1)
        s = (a + b + c) / 2.0
        area = np.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
        A_cst_total += area

    W_cst = A_cst_total * panel_t * rho_steel * g
    W_total = W_bar + W_cst

    print("-----------------------------")
    print(f"Total length of all bars: {L_total_bars:.2f} m")
    print(f"Total area of all CST panels: {A_cst_total:.2f} m^2")
    print(f"Total bar weight: {W_bar:.2f} N")
    print(f"Total CST panel weight: {W_cst:.2f} N")
    print(f"Total self-weight of the bridge: {W_total:.2f} N")
    print("-----------------------------")

    # -----------------------------
    # Distributed load along full length (bottom edge)
    # -----------------------------
    nr = Solver_NR_Loading()
    nr.assembly = assembly

    coords = node.coordinates_mat
    tol = 1e-9

    node_num = coords.shape[0]
    node_num_vec = np.arange(node_num)

    xmin = float(np.min(coords[:, 0]))
    xmax = float(np.max(coords[:, 0]))

    is_bottom = np.abs(coords[:, 2] - 0.0) < tol
    is_y0 = np.abs(coords[:, 1] - 0.0) < tol
    is_yw = np.abs(coords[:, 1] - W) < tol
    is_xmin = np.abs(coords[:, 0] - xmin) < tol
    is_xmax = np.abs(coords[:, 0] - xmax) < tol

    A = np.where(is_bottom & is_xmin & is_y0)[0]
    B = np.where(is_bottom & is_xmin & is_yw)[0]
    C = np.where(is_bottom & is_xmax & is_y0)[0]
    D = np.where(is_bottom & is_xmax & is_yw)[0]

    if len(A) == 0 or len(B) == 0 or len(C) == 0 or len(D) == 0:
        raise ValueError("Failed to find four bottom corner nodes. Check geometry/tol/W.")

    A = int(A[0])
    B = int(B[0])
    C = int(C[0])
    D = int(D[0])

    print(f"Corner supports: A={A+1}, B={B+1}, C={C+1}, D={D+1}")

    # lock uy for all nodes
    nr.supp = np.column_stack([
        node_num_vec,
        np.zeros(node_num),
        np.ones(node_num),
        np.zeros(node_num),
    ]).astype(int)
    # fully fix the 4 corners
    nr.supp[A, 1:4] = 1
    nr.supp[B, 1:4] = 1
    nr.supp[C, 1:4] = 1
    nr.supp[D, 1:4] = 1

    force = 1000.0
    step = 50

    L_total = L * N
    q = force / L_total

    is_bottom_z = np.abs(coords[:, 2] - 0.0) < tol
    is_edge_y0 = np.abs(coords[:, 1] - 0.0) < tol
    is_edge_yw = np.abs(coords[:, 1] - W) < tol
    bottom = np.where(is_bottom_z & (is_edge_y0 | is_edge_yw))[0]

    if bottom.size < 2:
        raise ValueError("Bottom node set too small. Check y/z criteria or geometry.")

    x_all = coords[bottom, 0]
    tol_x = 1e-9
    x_round = np.round(x_all / tol_x) * tol_x
    xu, inv = np.unique(x_round, return_inverse=True)
    order = np.argsort(xu)
    xu = xu[order]
    inv = order[inv]

    nsec = xu.size
    if nsec < 2:
        raise ValueError("Not enough x-sections to compute tributary lengths.")

    ell_sec = np.zeros(nsec, dtype=float)
    for s in range(nsec):
        if s == 0:
            ell_sec[s] = (xu[s + 1] - xu[s]) / 2.0
        elif s == nsec - 1:
            ell_sec[s] = (xu[s] - xu[s - 1]) / 2.0
        else:
            ell_sec[s] = (xu[s + 1] - xu[s - 1]) / 2.0

    Fz_base = np.zeros(bottom.size, dtype=float)
    for s in range(nsec):
        ids = np.where(inv == s)[0]
        Fsec = -q * ell_sec[s]
        Fz_base[ids] = Fsec / max(len(ids), 1)

    print(f"Check sum(Fz_base) = {np.sum(Fz_base):.6f} N (target = {-force:.6f} N)")

    nr.load = np.column_stack([
        bottom,
        np.zeros(bottom.size),
        np.zeros(bottom.size),
        Fz_base,
    ])

    nr.incre_step = step
    nr.iter_max = 50
    nr.tol = 1e-5

    Uhis = nr.Solve()

    # -----------------------------
    # Post-processing
    # -----------------------------
    P_final = force * step

    U_end = Uhis[-1, :, :]
    truss_strain = bar.solve_strain(node, U_end)
    internal_force = truss_strain * bar.E_vec * bar.A_vec

    max_bar_force = float(np.max(np.abs(internal_force)))
    idx_max = int(np.argmax(np.abs(internal_force))) + 1
    print(f"Max |bar force| = {max_bar_force:.6f} N at bar #{idx_max}")

    sigma_u = 300e6
    bar_failure_force = sigma_u * barA

    barLtotal = float(np.sum(bar.L0_vec))

    x_mid = 0.5 * L_total
    s_mid = int(np.argmin(np.abs(xu - x_mid)))
    mid_ids_local = np.where(inv == s_mid)[0]
    mid_nodes = bottom[mid_ids_local]
    Uaverage = -float(np.mean(U_end[mid_nodes, 2]))
    Kstiff = P_final / Uaverage if abs(Uaverage) > 1e-12 else np.inf

    print(f"Midspan nodes used for stiffness: {mid_nodes + 1}")
    print(f"Uaverage = {Uaverage:.6e} m, Kstiff = {Kstiff:.6e} N/m")

    bar_stress = truss_strain * bar.E_vec * (bar_failure_force / max_bar_force)
    fig_stress = plots.Plot_Shape_Bar_Stress(bar_stress)
    try:
        import matplotlib.pyplot as plt
        fig_stress.savefig(os.path.join(out_dir, "Rolling_Bridge_Load_Bar_Stress.png"),
                           dpi=200, bbox_inches="tight")
        plt.close(fig_stress)
    except Exception as e:
        print("WARNING: failed to save bar stress figure:", repr(e))

    loadatfail = P_final * (bar_failure_force / max_bar_force)
    print(f"Failure load is {loadatfail/1000:.3f} kN")
    print(f"Total bar length is {barLtotal:.3f} m")
    print(f"Stiffness is {Kstiff:.3e} N/m")

    if W_bar > 0:
        load_eff = loadatfail / W_bar
        print(f"Load efficiency (loadatfail/W_bar) = {load_eff:.6f}")

    nb = internal_force.size
    topk = min(10, nb)
    ord_idx = np.argsort(np.abs(internal_force))[::-1]
    print(f"Worst {topk} bars by |axial force|:")
    for ii in range(topk):
        b = int(ord_idx[ii])
        print(f"#{ii + 1}: Bar {b + 1}, N = {internal_force[b]/1e3:+.3f} kN")

    bar_sigma = internal_force / bar.A_vec
    sigma_max = float(np.max(np.abs(bar_sigma)))
    print(f"Max |axial stress| = {sigma_max/1e6:.3f} MPa")

    print("Execution Time:", time.time() - t0)


if __name__ == "__main__":
    main()
