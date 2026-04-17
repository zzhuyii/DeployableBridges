import numpy as np

from Elements_Nodes import Elements_Nodes
from Vec_Elements_Bars import Vec_Elements_Bars
from Vec_Elements_CST import Vec_Elements_CST
from Vec_Elements_RotSprings_4N import Vec_Elements_RotSprings_4N

from Assembly_Origami import Assembly_Origami
from Plot_Origami import Plot_Origami


def build_origami_bridge(
    L=2.0,
    W=4.0,
    H=2.0,
    N=4,
    barA=0.0023,
    barE=2.0e11,
    panel_E=2.0e8,
    panel_t=0.01,
    panel_v=0.3,
    rotK=1.0e8,
):
    node = Elements_Nodes()
    assembly = Assembly_Origami()
    cst = Vec_Elements_CST()
    rot_spr_4N = Vec_Elements_RotSprings_4N()
    bar = Vec_Elements_Bars()

    assembly.node = node
    assembly.cst = cst
    assembly.bar = bar
    assembly.rot_spr_4N = rot_spr_4N

    coords = []
    for i in range(1, N + 1):
        x0 = 2.0 * L * (i - 1)
        coords += [
            [x0, 0.0, H],
            [x0, 0.0, 0.0],
            [x0, W, 0.0],
            [x0, W, H],
            [x0 + L, 0.0, H],
            [x0 + L, 0.0, 0.0],
            [x0 + L, L, 0.0],
            [x0 + L, W, 0.0],
            [x0 + L, W, H],
        ]
    coords += [
        [2.0 * L * N, 0.0, H],
        [2.0 * L * N, 0.0, 0.0],
        [2.0 * L * N, W, 0.0],
        [2.0 * L * N, W, H],
    ]
    node.coordinates_mat = np.array(coords, dtype=float)

    tris = []
    for i in range(1, N + 1):
        b = 9 * (i - 1)
        tris += [
            [b + 2, b + 6, b + 7],
            [b + 2, b + 7, b + 3],
            [b + 3, b + 7, b + 8],
            [b + 6, b + 7, b + 11],
            [b + 7, b + 11, b + 12],
            [b + 7, b + 8, b + 12],
        ]
    cst.node_ijk_mat = np.array(tris, dtype=int)
    cst.t_vec = panel_t * np.ones(len(tris), dtype=float)
    cst.E_vec = panel_E * np.ones(len(tris), dtype=float)
    cst.v_vec = panel_v * np.ones(len(tris), dtype=float)

    bars = []
    for i in range(1, N + 1):
        b = 9 * (i - 1)
        bars += [
            [b + 2, b + 5], [b + 1, b + 2], [b + 1, b + 5],
            [b + 5, b + 11], [b + 5, b + 6], [b + 5, b + 10], [b + 10, b + 11],
            [b + 3, b + 9], [b + 4, b + 3], [b + 4, b + 9],
            [b + 9, b + 12], [b + 9, b + 8], [b + 9, b + 13], [b + 13, b + 12],
            [b + 2, b + 7], [b + 3, b + 7], [b + 6, b + 7], [b + 8, b + 7],
            [b + 11, b + 7], [b + 12, b + 7], [b + 2, b + 3], [b + 3, b + 8],
            [b + 8, b + 12], [b + 2, b + 6], [b + 6, b + 11],
        ]
    bars += [[9 * (N - 1) + 11, 9 * (N - 1) + 12]]
    bar.node_ij_mat = np.array(bars, dtype=int)
    bar.A_vec = barA * np.ones(len(bars), dtype=float)
    bar.E_vec = barE * np.ones(len(bars), dtype=float)

    spr4 = []
    for i in range(1, N + 1):
        b = 9 * (i - 1)
        spr4 += [
            [b + 5, b + 2, b + 6, b + 7],
            [b + 5, b + 6, b + 11, b + 7],
            [b + 2, b + 5, b + 6, b + 11],
            [b + 2, b + 6, b + 7, b + 11],
            [b + 9, b + 3, b + 8, b + 7],
            [b + 9, b + 8, b + 12, b + 7],
            [b + 3, b + 8, b + 9, b + 12],
            [b + 3, b + 7, b + 8, b + 12],
            [b + 2, b + 7, b + 3, b + 8],
            [b + 3, b + 7, b + 2, b + 6],
            [b + 6, b + 7, b + 11, b + 12],
            [b + 11, b + 7, b + 12, b + 8],
            [b + 1, b + 2, b + 5, b + 6],
            [b + 6, b + 5, b + 11, b + 10],
            [b + 4, b + 3, b + 9, b + 8],
            [b + 8, b + 9, b + 12, b + 13],
        ]
    for i in range(1, N):
        b = 9 * (i - 1)
        spr4 += [
            [b + 5, b + 10, b + 11, b + 14],
            [b + 9, b + 12, b + 13, b + 18],
            [b + 7, b + 11, b + 12, b + 16],
        ]
    rot_spr_4N.node_ijkl_mat = np.array(spr4, dtype=int)
    rot_spr_4N.rot_spr_K_vec = rotK * np.ones(len(spr4), dtype=float)

    plots = Plot_Origami()
    plots.assembly = assembly
    plots.displayRange = np.array([-1.0, 4.0 * (N + 1), -1.0, 5.0, -1.0, 3.0], dtype=float)
    plots.viewAngle1 = 20
    plots.viewAngle2 = 20

    return assembly, node, bar, cst, rot_spr_4N, plots
