import os
import time
import numpy as np
import matplotlib.pyplot as plt

"""Python conversion of the updated Kirigami_Truss_Deploy.m.

Source MATLAB path:
D:\\PAPER\\1st paper\\2026-DeployableBridges\\Kirigami_Truss_Deploy.m

The MATLAB deploy source defines N=8 and increStep=1000.
"""

from KirigamiTruss_common import build_kirigami_truss
from Solver_NR_Folding_4N import Solver_NR_Folding_4N


OUT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    start = time.time()

    L = 2.0
    N = 8
    assert N == 8, "Kirigami_Truss_Deploy.m uses N=8."
    assembly, node, bar, cst, rot_spr_4N, rot_spr_3N, plots = build_kirigami_truss(
        L=L,
        gap=0.0,
        N=N,
        barA=0.0023,
        barE=2.0e11,
        panel_E=2.0e8,
        panel_t=0.01,
        panel_v=0.3,
        rot4K=10.0,
        rot3K=1.0e8,
    )
    assembly.Initialize_Assembly()

    sf = Solver_NR_Folding_4N()
    sf.assembly = assembly
    sf.supp = [[1, 1, 1, 1], [2, 1, 1, 1], [3, 1, 1, 1], [4, 1, 1, 1]]
    sf.targetRot = rot_spr_4N.theta_stress_free_vec.copy()
    sf.increStep = int(os.environ.get("KIRIGAMI_DEPLOY_STEPS", "1000"))
    sf.iterMax = 30
    sf.tol = 1.0e-4

    rate = 0.9
    for i in range(N):
        b = 24 * i
        plus = [1, 2, 9, 10, 11, 12, 15, 16, 17, 18, 19, 20]
        minus = [3, 4, 5, 6, 7, 8, 13, 14, 21, 22, 23, 24]
        for idx in plus:
            sf.targetRot[b + idx - 1] = np.pi + rate * np.pi
        for idx in minus:
            sf.targetRot[b + idx - 1] = np.pi - rate * np.pi

    Uhis = sf.Solve()

    final_U = Uhis[-1, :, :]
    fig = plots.Plot_Deformed_Shape(final_U)
    fig.savefig(os.path.join(OUT_DIR, "Kirigami_Truss_Deploy_Final.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    np.save(os.path.join(OUT_DIR, "KirigamiUhis.npy"), Uhis)

    plots.file_name = os.path.join(OUT_DIR, "Kirigami_Truss_Deploy.gif")
    plots.Plot_Deformed_His(Uhis[::20, :, :])

    print("Kirigami truss deploy complete")
    print(f"Nodes: {node.coordinates_mat.shape[0]}, bars: {bar.node_ij_mat.shape[0]}, CST: {cst.node_ijk_mat.shape[0]}")
    print(f"4N springs: {rot_spr_4N.node_ijkl_mat.shape[0]}, 3N springs: {rot_spr_3N.node_ijk_mat.shape[0]}")
    print("Execution Time:", time.time() - start)


if __name__ == "__main__":
    main()
