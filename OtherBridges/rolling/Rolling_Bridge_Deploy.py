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
from CD_Elements_RotSprings_3N import CD_Elements_RotSprings_3N
from Vec_Elements_CST import Vec_Elements_CST

from Assembly_Rolling_Bridge import Assembly_Rolling_Bridge
from Plot_Rolling_Bridge import Plot_Rolling_Bridge
from Solver_NR_TrussAction import Solver_NR_TrussAction


def main():
    print("RUNNING FILE:", __file__)
    t0 = time.time()

    # -----------------------------
    # Define Rolling Bridge Geometry
    # -----------------------------
    H = 1100 / 1000
    HA = 1100 / 1000
    L = 1600 / 1000
    l = 530 / 1000
    W = 1600 / 1000

    barA = 0.0001
    barE = 80e9
    panelE = 10e6
    panelv = 0.3
    panelt = 100 / 1000

    activeBarE = 80e9
    N = 2

    I = (1.0 / 12.0) * (0.01 ** 4)
    kspr = 3 * barE * I / L

    node = Elements_Nodes()
    bar = Vec_Elements_Bars()
    actBar = Std_Elements_Bars()
    rot_spr_3N = CD_Elements_RotSprings_3N()
    cst = Vec_Elements_CST()

    assembly = Assembly_Rolling_Bridge()
    assembly.node = node
    assembly.bar = bar
    assembly.actBar = actBar
    assembly.rot_spr_3N = rot_spr_3N
    assembly.cst = cst

    # -----------------------------
    # Define nodal coordinates
    # -----------------------------
    coords = [
        [0, 0, 0],
        [0, W, 0],
    ]

    for i in range(1, N + 1):
        x0 = (i - 1) * L
        xL = x0 + L
        xl = x0 + l
        xL_l = x0 + L - l

        if i < N:
            coords += [
                [xL, 0, 0],
                [xL, W, 0],
                [xl, 0, H],
                [xL_l, 0, H],
                [xl, W, H],
                [xL_l, W, H],
                [xL, 0, HA],
                [xL, W, HA],
            ]
        else:
            coords += [
                [xL, 0, 0],
                [xL, W, 0],
                [xl, 0, H],
                [xL_l, 0, H],
                [xl, W, H],
                [xL_l, W, H],
            ]

    node.coordinates_mat = np.array(coords, dtype=float)

    # -----------------------------
    # Plot settings (inspection)
    # -----------------------------
    plots = Plot_Rolling_Bridge()
    plots.assembly = assembly
    plots.displayRange = np.array([-2, 14, -1, 2, -1, 10], dtype=float)
    plots.Plot_Shape_Node_Number()

    # -----------------------------
    # Define CST panels
    # -----------------------------
    tri = []
    for i in range(1, N + 1):
        if i == 1:
            tri += [
                [(i - 1) * 8 + 1, (i - 1) * 8 + 2, (i - 1) * 8 + 3],
                [(i - 1) * 8 + 2, (i - 1) * 8 + 3, (i - 1) * 8 + 4],
            ]
        else:
            tri += [
                [(i - 2) * 8 + 3, (i - 2) * 8 + 4, (i - 2) * 8 + 11],
                [(i - 2) * 8 + 4, (i - 2) * 8 + 11, (i - 2) * 8 + 12],
            ]

    cst.node_ijk_mat = np.array(tri, dtype=int)
    cst_num = cst.node_ijk_mat.shape[0]
    cst.v_vec = panelv * np.ones(cst_num, dtype=float)
    cst.E_vec = panelE * np.ones(cst_num, dtype=float)
    cst.t_vec = panelt * np.ones(cst_num, dtype=float)
    plots.Plot_Shape_CST_Number()

    # -----------------------------
    # Define normal bars
    # -----------------------------
    bars = []
    for i in range(1, N + 1):
        if i == 1:
            bars += [
                [1, 3], [3, 6], [6, 5], [5, 1],
                [2, 4], [4, 8], [8, 7], [7, 2],
                [6, 9], [8, 10],
                [1, 2],
            ]
        elif i == N:
            bars += [
                [(i - 2) * 8 + 3, (i - 2) * 8 + 11],
                [(i - 2) * 8 + 11, (i - 2) * 8 + 14],
                [(i - 2) * 8 + 13, (i - 2) * 8 + 14],
                [(i - 2) * 8 + 13, (i - 2) * 8 + 3],

                [(i - 2) * 8 + 4, (i - 2) * 8 + 12],
                [(i - 2) * 8 + 12, (i - 2) * 8 + 16],
                [(i - 2) * 8 + 15, (i - 2) * 8 + 16],
                [(i - 2) * 8 + 15, (i - 2) * 8 + 4],

                [(i - 2) * 8 + 9, (i - 2) * 8 + 13],
                [(i - 2) * 8 + 10, (i - 2) * 8 + 15],

                [(i - 2) * 8 + 3, (i - 2) * 8 + 4],
                [(i - 2) * 8 + 11, (i - 2) * 8 + 12],
            ]
        else:
            bars += [
                [(i - 2) * 8 + 3, (i - 2) * 8 + 11],
                [(i - 2) * 8 + 11, (i - 2) * 8 + 14],
                [(i - 2) * 8 + 13, (i - 2) * 8 + 14],
                [(i - 2) * 8 + 13, (i - 2) * 8 + 3],

                [(i - 2) * 8 + 4, (i - 2) * 8 + 12],
                [(i - 2) * 8 + 12, (i - 2) * 8 + 16],
                [(i - 2) * 8 + 15, (i - 2) * 8 + 16],
                [(i - 2) * 8 + 15, (i - 2) * 8 + 4],

                [(i - 2) * 8 + 9, (i - 2) * 8 + 13],
                [(i - 2) * 8 + 14, (i - 2) * 8 + 17],
                [(i - 2) * 8 + 10, (i - 2) * 8 + 15],
                [(i - 2) * 8 + 16, (i - 2) * 8 + 18],

                [(i - 2) * 8 + 3, (i - 2) * 8 + 4],
            ]

    bar.node_ij_mat = np.array(bars, dtype=int)
    bar_num = bar.node_ij_mat.shape[0]
    bar.A_vec = barA * np.ones(bar_num, dtype=float)
    bar.E_vec = barE * np.ones(bar_num, dtype=float)
    plots.Plot_Shape_Bar_Number()

    # -----------------------------
    # Define active bars
    # -----------------------------
    act = []
    for i in range(1, N):
        act += [
            [3 + (i - 1) * 8, 9 + (i - 1) * 8],
            [4 + (i - 1) * 8, 10 + (i - 1) * 8],
        ]

    actBar.node_ij_mat = np.array(act, dtype=int)
    act_num = actBar.node_ij_mat.shape[0]
    actBar.A_vec = barA * np.ones(act_num, dtype=float)
    actBar.E_vec = activeBarE * np.ones(act_num, dtype=float)
    plots.Plot_Shape_ActBar_Number()

    assembly.Initialize_Assembly()

    # -----------------------------
    # Define rotational springs
    # -----------------------------
    spr3 = []
    for i in range(1, N + 1):
        if i == 1:
            spr3 += [
                [5, 1, 3],
                [1, 3, 6],
                [3, 6, 5],
                [6, 5, 1],

                [7, 2, 4],
                [2, 4, 8],
                [4, 8, 7],
                [8, 7, 2],
            ]
        else:
            spr3 += [
                [13 + (i - 2) * 8, 3 + (i - 2) * 8, 11 + (i - 2) * 8],
                [3 + (i - 2) * 8, 11 + (i - 2) * 8, 14 + (i - 2) * 8],
                [11 + (i - 2) * 8, 14 + (i - 2) * 8, 13 + (i - 2) * 8],
                [14 + (i - 2) * 8, 13 + (i - 2) * 8, 3 + (i - 2) * 8],

                [15 + (i - 2) * 8, 4 + (i - 2) * 8, 12 + (i - 2) * 8],
                [4 + (i - 2) * 8, 12 + (i - 2) * 8, 16 + (i - 2) * 8],
                [12 + (i - 2) * 8, 16 + (i - 2) * 8, 15 + (i - 2) * 8],
                [16 + (i - 2) * 8, 15 + (i - 2) * 8, 4 + (i - 2) * 8],
            ]

    rot_spr_3N.node_ijk_mat = np.array(spr3, dtype=int)
    rot_num = rot_spr_3N.node_ijk_mat.shape[0]
    rot_spr_3N.rot_spr_K_vec = kspr * np.ones(rot_num, dtype=float)

    plots.Plot_Shape_Spr_Number()
    assembly.Initialize_Assembly()

    # -----------------------------
    # Solver setup
    # -----------------------------
    ta = Solver_NR_TrussAction()
    ta.assembly = assembly

    node_num = node.coordinates_mat.shape[0]
    node_num_vec = np.arange(node_num)
    ta.supp = np.column_stack([node_num_vec, np.zeros(node_num), np.ones(node_num), np.zeros(node_num)])
    ta.supp[0, 1:4] = 1
    ta.supp[1, 1:4] = 1
    ta.supp[2, 1:4] = 1
    ta.supp[3, 1:4] = 1

    ta.increStep = 400
    ta.iterMax = 30
    ta.tol = 1e-4

    dL = 0.4
    ta.targetL0 = actBar.L0_vec.copy() + dL

    Uhis = ta.Solve()

    plots.Plot_Deformed_Shape(Uhis[-1])

    plots.fileName = os.path.join(_THIS_DIR, "Rolling_Bridge_Deploy.gif")
    plots.Plot_Deformed_His(Uhis[::10])

    print("Execution Time:", time.time() - t0)


if __name__ == "__main__":
    main()
