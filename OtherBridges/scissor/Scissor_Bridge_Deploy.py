import os
import numpy as np
import time

from Elements_Nodes import Elements_Nodes
from Vec_Elements_CST import Vec_Elements_CST
from Vec_Elements_RotSprings_4N import Vec_Elements_RotSprings_4N
from Std_Elements_Bars import Std_Elements_Bars
from CD_Elements_RotSprings_3N import CD_Elements_RotSprings_3N

from Assembly_Scissor_Bridge import Assembly_Scissor_Bridge
from Plot_Scissor_Bridge import Plot_Scissor_Bridge
from Solver_NR_Folding_4N import Solver_NR_Folding_4N


def main():
    print("RUNNING FILE:", __file__)

    # -----------------------------
    # Settings
    # -----------------------------
    DO_PLOT_DEBUG = False
    DO_GIF = True
    SAVE_FINAL_PNG = True
    GIF_STRIDE = 10  # match MATLAB Uhis(1:10:end,:,:)

    out_dir = os.path.dirname(os.path.abspath(__file__))

    # -----------------------------
    # Timing
    # -----------------------------
    t0 = time.time()

    # -----------------------------
    # Initialize the scissor
    # -----------------------------
    N = 2
    H = 1.0
    L = 1.0
    barA = 0.0063 * 0.01
    barE = 2.0e9

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
    # Define CST triangles
    # -----------------------------
    tri = []
    for i in range(1, N + 1):
        idx = 19 * (i - 1)
        tri += [
            [idx + 1,  idx + 2,  idx + 15],
            [idx + 2,  idx + 15, idx + 16],
            [idx + 16, idx + 15, idx + 20],
            [idx + 20, idx + 16, idx + 21],
        ]

    cst.node_ijk_mat = np.array(tri, dtype=int)
    cstNum = cst.node_ijk_mat.shape[0]
    cst.t_vec = 0.0063 * np.ones(cstNum, dtype=float)
    cst.E_vec = 2.0e9 * np.ones(cstNum, dtype=float)
    cst.v_vec = 0.25 * np.ones(cstNum, dtype=float)

    # -----------------------------
    # Define bars
    # -----------------------------
    bars = []

    for i in range(1, N + 1):
        idx = 19 * (i - 1)

        bars += [
            [idx + 1,  idx + 5],
            [idx + 5,  idx + 13],
            [idx + 13, idx + 11],
            [idx + 11, idx + 22],

            [idx + 3,  idx + 7],
            [idx + 7,  idx + 13],
            [idx + 13, idx + 9],
            [idx + 9,  idx + 20],

            [idx + 2,  idx + 6],
            [idx + 6,  idx + 14],
            [idx + 14, idx + 12],
            [idx + 12, idx + 23],

            [idx + 4,  idx + 8],
            [idx + 8,  idx + 14],
            [idx + 14, idx + 10],
            [idx + 10, idx + 21],

            [idx + 3,  idx + 4],
            [idx + 3,  idx + 17],
            [idx + 3,  idx + 19],
            [idx + 4,  idx + 19],
            [idx + 4,  idx + 18],

            [idx + 17, idx + 19],
            [idx + 18, idx + 19],

            [idx + 18, idx + 23],
            [idx + 19, idx + 23],
            [idx + 17, idx + 22],
            [idx + 19, idx + 22],

            [idx + 1,  idx + 2],
            [idx + 15, idx + 16],
            [idx + 1,  idx + 15],
            [idx + 15, idx + 20],
            [idx + 2,  idx + 16],
            [idx + 16, idx + 21],

            [idx + 2,  idx + 15],
            [idx + 16, idx + 20],
        ]

    bars += [
        [19 * N + 3, 19 * N + 4],
        [19 * N + 1, 19 * N + 2],
    ]

    bar.node_ij_mat = np.array(bars, dtype=int)
    barNum = bar.node_ij_mat.shape[0]
    bar.A_vec = barA * np.ones(barNum, dtype=float)
    bar.E_vec = barE * np.ones(barNum, dtype=float)

    # -----------------------------
    # Define 3-node rotational springs
    # -----------------------------
    spr3 = []
    for i in range(1, N + 1):
        idx = 19 * (i - 1)
        spr3 += [
            [idx + 1,  idx + 5,  idx + 13],
            [idx + 5,  idx + 13, idx + 11],
            [idx + 13, idx + 11, idx + 22],

            [idx + 3,  idx + 7,  idx + 13],
            [idx + 7,  idx + 13, idx + 9],
            [idx + 13, idx + 9,  idx + 20],

            [idx + 2,  idx + 6,  idx + 14],
            [idx + 6,  idx + 14, idx + 12],
            [idx + 14, idx + 12, idx + 23],

            [idx + 4,  idx + 8,  idx + 14],
            [idx + 8,  idx + 14, idx + 10],
            [idx + 14, idx + 10, idx + 21],
        ]

    rotSpr3N.node_ijk_mat = np.array(spr3, dtype=int)
    rotNum3N = rotSpr3N.node_ijk_mat.shape[0]
    rotSpr3N.rot_spr_K_vec = kspr * np.ones(rotNum3N, dtype=float)

    # -----------------------------
    # Define 4-node rotational springs
    # -----------------------------
    spr4 = []
    for i in range(1, N + 1):
        idx = 19 * (i - 1)
        spr4 += [
            [idx + 1,  idx + 2,  idx + 15, idx + 16],
            [idx + 2,  idx + 15, idx + 16, idx + 20],
            [idx + 15, idx + 16, idx + 20, idx + 21],

            [idx + 3,  idx + 17, idx + 19, idx + 22],
            [idx + 4,  idx + 18, idx + 19, idx + 23],

            [idx + 3,  idx + 4,  idx + 19, idx + 18],
            [idx + 4,  idx + 3,  idx + 19, idx + 17],
            [idx + 18, idx + 19, idx + 23, idx + 22],
            [idx + 17, idx + 22, idx + 19, idx + 23],
        ]

    rotSpr4N.node_ijkl_mat = np.array(spr4, dtype=int)
    rotNum4N = rotSpr4N.node_ijkl_mat.shape[0]
    rotSpr4N.rot_spr_K_vec = 0.01 * np.ones(rotNum4N, dtype=float)

    # -----------------------------
    # Plot object
    # -----------------------------
    plots = Plot_Scissor_Bridge()
    plots.assembly = assembly
    plots.displayRange = np.array([-0.3 * L, L * (N + 1), -0.3 * L, 1.3 * L, -0.3 * L, 1.5 * L], dtype=float)
    plots.viewAngle1 = 20
    plots.viewAngle2 = 20
    plots.holdTime = 0.1

    if DO_PLOT_DEBUG:
        plots.Plot_Shape_Node_Number()
        plots.Plot_Shape_CST_Number()
        plots.Plot_Shape_Bar_Number()
        plots.Plot_Shape_RotSpr_3N_Number()
        plots.Plot_Shape_RotSpr_4N_Number()

    # -----------------------------
    # Initialize assembly
    # -----------------------------
    assembly.Initialize_Assembly()

    # -----------------------------
    # Set up solver
    # -----------------------------
    sf = Solver_NR_Folding_4N()
    sf.assembly = assembly

    sf.supp = [
        [1, 1, 1, 1],
        [2, 1, 1, 1],
        [3, 1, 1, 0],
        [4, 1, 1, 0],
    ]

    sf.increStep = 50
    sf.iterMax = 20
    sf.tol = 1e-5

    sf.targetRot = np.array(assembly.rotSpr.theta_current_vec, dtype=float).copy()
    targetAngle = 0.9 * np.pi

    for i in range(1, N + 1):
        base = 9 * (i - 1)
        sf.targetRot[base + 1] = np.pi - targetAngle
        sf.targetRot[base + 3] = np.pi + targetAngle
        sf.targetRot[base + 4] = np.pi + targetAngle

    # -----------------------------
    # Solve
    # -----------------------------
    Uhis = sf.Solve()

    # -----------------------------
    # Post-process / Output
    # -----------------------------
    print("Execution Time:", time.time() - t0)
    print("Uhis shape:", Uhis.shape)

    if DO_GIF:
        try:
            plots.fileName = os.path.join(out_dir, "Scissor_Bridge_Deploy.gif")
            plots.Plot_Deformed_His(Uhis[0::GIF_STRIDE, :, :])
            print("GIF saved:", plots.fileName)
        except Exception as e:
            print("WARNING: GIF export failed:", repr(e))

    U_end = Uhis[-1, :, :]
    try:
        plots.Plot_Deformed_Shape(U_end)
    except Exception as e:
        print("WARNING: final shape plot failed:", repr(e))

    if SAVE_FINAL_PNG:
        try:
            import matplotlib.pyplot as plt
            fig = plots.Plot_Deformed_Shape(U_end)
            png_path = os.path.join(out_dir, "Scissor_Bridge_Final.png")
            fig.savefig(png_path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            print("Final PNG saved:", png_path)
        except Exception as e:
            print("WARNING: final PNG export failed:", repr(e))


if __name__ == "__main__":
    main()
