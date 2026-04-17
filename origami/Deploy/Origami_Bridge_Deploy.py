import os
import time
import numpy as np

from Elements_Nodes import Elements_Nodes
from Vec_Elements_CST import Vec_Elements_CST
from Vec_Elements_RotSprings_4N import Vec_Elements_RotSprings_4N
from Vec_Elements_Bars import Vec_Elements_Bars

from Plot_Origami import Plot_Origami
from Solver_NR_Folding_4N import Solver_NR_Folding_4N

from Assembly_Origami import Assembly_Origami


def main():
    print("RUNNING FILE:", __file__)

    # -----------------------------
    # Timing
    # -----------------------------
    t0 = time.time()

    # -----------------------------
    # Define Geometry
    # -----------------------------
    N = 4
    L = 2.0

    barA = 0.0023
    barE = 2.0e11

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
            [x0, 0, L],
            [x0, 0, 0],
            [x0, 2 * L, 0],
            [x0, 2 * L, L],
            [x0 + L, 0, L],
            [x0 + L, 0, 0],
            [x0 + L, L, 0],
            [x0 + L, 2 * L, 0],
            [x0 + L, 2 * L, L],
        ]

    coords += [
        [2 * L * N, 0, L],
        [2 * L * N, 0, 0],
        [2 * L * N, 2 * L, 0],
        [2 * L * N, 2 * L, L],
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
    rot_spr.rot_spr_K_vec = np.ones(rot_num, dtype=float)

    # -----------------------------
    # Initialize assembly
    # -----------------------------
    assembly.Initialize_Assembly()

    # -----------------------------
    # Set up solver
    # -----------------------------
    sf = Solver_NR_Folding_4N()
    sf.assembly = assembly

    supp = []
    for i in range(1, N + 2):
        supp += [
            [9 * (i - 1) + 2, 0, 1, 1],
            [9 * (i - 1) + 3, 0, 1, 1],
        ]

    # Fix x-direction for the first two support rows
    supp[0][1] = 1
    supp[1][1] = 1
    sf.supp = supp

    sf.targetRot = np.pi * np.ones_like(rot_spr.theta_stress_free_vec)
    sf.increStep = 100
    sf.iterMax = 20
    sf.tol = 1e-5

    Uhis1 = sf.Solve()
    plots.Plot_Deformed_Shape(Uhis1[-1])

    sf.increStep = 2000
    targetFold = 0.9 * np.pi

    tr = np.array(sf.targetRot, dtype=float)
    for i in range(N):
        base = 16 * i
        for idx in [2, 3, 6, 7]:
            tr[base + idx] = np.pi - targetFold

        for idx in [12, 13, 9, 10]:
            tr[base + idx] = np.pi + targetFold

        for idx in [14, 15]:
            tr[base + idx] = np.pi - targetFold

        for idx in [8, 11]:
            tr[base + idx] = np.pi + targetFold

    for i in range(N - 1):
        base = 16 * N + 3 * i
        tr[base + 0] = np.pi - targetFold
        tr[base + 1] = np.pi + targetFold
        tr[base + 2] = np.pi - targetFold

    sf.targetRot = tr

    Uhis2 = sf.Solve()

    Uhis = np.concatenate([Uhis1, Uhis2[0::10, :, :]], axis=0)

    out_dir = os.path.dirname(os.path.abspath(__file__))
    plots.fileName = os.path.join(out_dir, "Origami_Bridge_Deploy.gif")
    plots.Plot_Deformed_His(Uhis[0::4, :, :])

    print("Execution Time:", time.time() - t0)


if __name__ == "__main__":
    main()
