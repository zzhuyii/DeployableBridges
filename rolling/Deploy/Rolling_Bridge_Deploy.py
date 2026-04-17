import os
import time
import numpy as np
import matplotlib.pyplot as plt

"""Python conversion of Rolling_Bridge_Deploy.m.

Source MATLAB path:
D:\\PAPER\\1st paper\\2026-DeployableBridges\\Rolling_Bridge_Deploy.m

The MATLAB source defines N=8, and this file intentionally keeps N fixed at 8.
"""

from Rolling_Bridge_common import build_rolling_bridge
from Solver_NR_TrussAction import Solver_NR_TrussAction


def main():
    print("RUNNING FILE:", __file__)
    start = time.time()
    out_dir = os.path.dirname(os.path.abspath(__file__))

    H = 2.0
    W = 2.0
    L = 2.0
    N = 8
    assert N == 8, "Rolling_Bridge_Deploy.m uses N=8."

    assembly, node, bar, actBar, cst, rot_spr_4N, plots = build_rolling_bridge(
        H=H,
        W=W,
        L=L,
        N=N,
        barA=0.0023,
        barE=2.0e11,
        panel_E=2.0e8,
        panel_t=0.01,
        panel_v=0.3,
        activeBarE=2.0e11,
        rotK=1.0e6,
    )
    plots.displayRange = np.array([-2.0, 18.0, -1.0, 3.0, -1.0, 14.0], dtype=float)
    assembly.Initialize_Assembly()

    ta = Solver_NR_TrussAction()
    ta.assembly = assembly
    node_num = node.coordinates_mat.shape[0]
    node_ids = np.arange(node_num)
    ta.supp = np.column_stack([node_ids, np.zeros(node_num), np.zeros(node_num), np.zeros(node_num)])
    ta.supp[0:4, 1:4] = 1
    ta.increStep = int(os.environ.get("ROLLING_DEPLOY_STEPS", "800"))
    ta.iterMax = 30
    ta.tol = 1.0e-1
    ta.targetL0 = actBar.L0_vec.copy() + 1.1

    Uhis = ta.Solve()
    final_U = Uhis[-1, :, :]

    fig = plots.Plot_Deformed_Shape(final_U)
    fig.savefig(os.path.join(out_dir, "Rolling_Bridge_Deploy_Final.png"), dpi=200, bbox_inches="tight")
    plt.close(fig)

    plots.fileName = os.path.join(out_dir, "Rolling_Bridge_Deploy.gif")
    plots.Plot_Deformed_His(Uhis[::10, :, :])

    print(f"Nodes: {node.coordinates_mat.shape[0]}, bars: {bar.node_ij_mat.shape[0]}, actuator bars: {actBar.node_ij_mat.shape[0]}")
    print(f"CST: {cst.node_ijk_mat.shape[0]}, 4N springs: {rot_spr_4N.node_ijkl_mat.shape[0]}")
    print("Execution Time:", time.time() - start)


if __name__ == "__main__":
    main()
