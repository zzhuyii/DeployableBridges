import numpy as np

r"""Kirigami truss geometry translated from:
D:\PAPER\1st paper\2026-DeployableBridges\Kirigami_Truss_Deploy.m

The updated MATLAB deploy source uses L=2, gap=0, N=8, 4-node rotational
springs, and 3-node rotational springs. This module intentionally rejects
legacy N values so the deploy script cannot fall back to an old N=2 model.
"""

from Elements_Nodes import Elements_Nodes
from Vec_Elements_Bars import Vec_Elements_Bars
from Vec_Elements_CST import Vec_Elements_CST
from Vec_Elements_RotSprings_4N import Vec_Elements_RotSprings_4N
from CD_Elements_RotSprings_3N import CD_Elements_RotSprings_3N
from Assembly_KirigamiTruss import Assembly_KirigamiTruss
from Plot_KirigamiTruss import Plot_KirigamiTruss


def build_kirigami_truss(
    L=2.0,
    gap=0.0,
    N=8,
    barA=0.0023,
    barE=2.0e11,
    panel_E=2.0e8,
    panel_t=0.01,
    panel_v=0.3,
    rot4K=10.0,
    rot3K=1.0e8,
):
    if N != 8:
        raise ValueError("Kirigami_Truss_Deploy.m uses N=8; refusing to build a legacy N!=8 model.")

    node = Elements_Nodes()
    cst = Vec_Elements_CST()
    rot_spr_4N = Vec_Elements_RotSprings_4N()
    rot_spr_3N = CD_Elements_RotSprings_3N()
    bar = Vec_Elements_Bars()

    assembly = Assembly_KirigamiTruss()
    assembly.node = node
    assembly.cst = cst
    assembly.bar = bar
    assembly.rot_spr_4N = rot_spr_4N
    assembly.rot_spr_3N = rot_spr_3N

    coords = [
        [0.0, 0.0, 0.0],
        [0.0, L, 0.0],
        [0.0, 0.0, L],
        [0.0, L, L],
    ]
    for i in range(1, N + 1):
        x_mid = L * (i - 1) + L / 2.0
        x_end = L * (i - 1) + L
        coords += [
            [x_mid, 0.0, 0.0],
            [x_mid, 0.0, gap],
            [x_mid, L, 0.0],
            [x_mid, L, gap],
            [x_mid, 0.0, L],
            [x_mid, gap, L],
            [x_mid, L, L],
            [x_mid, L, L - gap],
            [x_mid, L / 2.0, 0.0],
            [x_mid, L / 2.0, L],
            [x_mid, 0.0, L / 2.0],
            [x_mid, L, L / 2.0],
            [x_end, 0.0, 0.0],
            [x_end, L, 0.0],
            [x_end, 0.0, L],
            [x_end, L, L],
        ]
    node.coordinates_mat = np.array(coords, dtype=float)

    tris = []
    for i in range(1, N + 1):
        b = 16 * (i - 1)
        tris += [
            [b + 1, b + 5, b + 13],
            [b + 1, b + 13, b + 2],
            [b + 2, b + 7, b + 13],
            [b + 7, b + 13, b + 18],
            [b + 13, b + 17, b + 18],
            [b + 13, b + 5, b + 17],
        ]
    cst.node_ijk_mat = np.array(tris, dtype=int)
    cst.t_vec = panel_t * np.ones(len(tris), dtype=float)
    cst.E_vec = panel_E * np.ones(len(tris), dtype=float)
    cst.v_vec = panel_v * np.ones(len(tris), dtype=float)

    bars = []
    for i in range(1, N + 1):
        b = 16 * (i - 1)
        bars += [
            [b + 1, b + 2], [b + 1, b + 3], [b + 2, b + 4], [b + 3, b + 4],
            [b + 1, b + 5], [b + 5, b + 17], [b + 2, b + 7], [b + 7, b + 18],
            [b + 3, b + 9], [b + 9, b + 19], [b + 4, b + 11], [b + 11, b + 20],
            [b + 4, b + 12], [b + 12, b + 20], [b + 3, b + 10], [b + 10, b + 19],
            [b + 1, b + 6], [b + 6, b + 17], [b + 2, b + 8], [b + 8, b + 18],
            [b + 1, b + 15], [b + 3, b + 15], [b + 15, b + 17], [b + 15, b + 19],
            [b + 2, b + 16], [b + 16, b + 20], [b + 4, b + 16], [b + 16, b + 18],
            [b + 3, b + 14], [b + 4, b + 14], [b + 14, b + 20], [b + 14, b + 19],
            [b + 1, b + 13], [b + 2, b + 13], [b + 13, b + 18], [b + 13, b + 17],
            [b + 10, b + 14], [b + 11, b + 14], [b + 12, b + 16], [b + 8, b + 16],
            [b + 7, b + 13], [b + 5, b + 13], [b + 6, b + 15], [b + 15, b + 9],
        ]
    b = 16 * N
    bars += [[b + 1, b + 2], [b + 1, b + 3], [b + 2, b + 4], [b + 3, b + 4]]
    bar.node_ij_mat = np.array(bars, dtype=int)
    bar.A_vec = barA * np.ones(len(bars), dtype=float)
    bar.E_vec = barE * np.ones(len(bars), dtype=float)

    spr4 = []
    for i in range(1, N + 1):
        b = 16 * (i - 1)
        spr4 += [
            [b + 1, b + 6, b + 15, b + 17],
            [b + 3, b + 9, b + 15, b + 19],
            [b + 1, b + 3, b + 15, b + 9],
            [b + 3, b + 1, b + 15, b + 6],
            [b + 6, b + 15, b + 17, b + 19],
            [b + 9, b + 15, b + 19, b + 17],
            [b + 3, b + 10, b + 14, b + 19],
            [b + 4, b + 11, b + 14, b + 20],
            [b + 11, b + 14, b + 4, b + 3],
            [b + 4, b + 14, b + 3, b + 10],
            [b + 10, b + 14, b + 19, b + 20],
            [b + 11, b + 14, b + 20, b + 19],
            [b + 2, b + 8, b + 16, b + 18],
            [b + 4, b + 12, b + 16, b + 20],
            [b + 2, b + 16, b + 4, b + 12],
            [b + 4, b + 16, b + 2, b + 8],
            [b + 8, b + 16, b + 18, b + 20],
            [b + 18, b + 16, b + 20, b + 12],
            [b + 2, b + 7, b + 13, b + 18],
            [b + 1, b + 5, b + 13, b + 17],
            [b + 1, b + 13, b + 2, b + 7],
            [b + 2, b + 13, b + 1, b + 5],
            [b + 5, b + 13, b + 17, b + 18],
            [b + 7, b + 13, b + 18, b + 17],
        ]
    rot_spr_4N.node_ijkl_mat = np.array(spr4, dtype=int)
    rot_spr_4N.rot_spr_K_vec = rot4K * np.ones(len(spr4), dtype=float)

    spr3 = []
    for i in range(1, N + 2):
        b = 16 * (i - 1)
        spr3 += [
            [b + 1, b + 2, b + 4],
            [b + 2, b + 4, b + 3],
            [b + 4, b + 3, b + 1],
            [b + 3, b + 1, b + 2],
        ]
    rot_spr_3N.node_ijk_mat = np.array(spr3, dtype=int)
    rot_spr_3N.rot_spr_K_vec = rot3K * np.ones(len(spr3), dtype=float)

    plots = Plot_KirigamiTruss()
    plots.assembly = assembly
    plots.display_range = np.array([-1.0, L * (N + 1), -1.0, 3.0, -1.0, 3.0], dtype=float)
    plots.view_angle1 = 20
    plots.view_angle2 = 20

    return assembly, node, bar, cst, rot_spr_4N, rot_spr_3N, plots
