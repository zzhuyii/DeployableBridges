import os
import sys
import numpy as np

r"""Rolling bridge geometry translated from:
D:\PAPER\1st paper\2026-DeployableBridges\Rolling_Bridge_Deploy.m
D:\PAPER\1st paper\2026-DeployableBridges\Rolling_Bridge_Pedestrain_Load_LRFD.m

The MATLAB source uses N=8 sections. This module intentionally has no N=2
fallback or legacy geometry.
"""

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
if _PROJECT_ROOT not in sys.path:
    sys.path.append(_PROJECT_ROOT)

from Elements_Nodes import Elements_Nodes
from Vec_Elements_Bars import Vec_Elements_Bars
from Std_Elements_Bars import Std_Elements_Bars
from Vec_Elements_CST import Vec_Elements_CST
from Vec_Elements_RotSprings_4N import Vec_Elements_RotSprings_4N

from Assembly_Rolling_Bridge import Assembly_Rolling_Bridge
from Plot_Rolling_Bridge import Plot_Rolling_Bridge


def build_rolling_bridge(
    H=2.0,
    W=2.0,
    L=2.0,
    N=8,
    barA=0.0023,
    barE=2.0e11,
    panel_E=2.0e8,
    panel_t=0.01,
    panel_v=0.3,
    activeBarE=2.0e11,
    rotK=1.0e6,
    barA_brace=None,
):
    if N != 8:
        raise ValueError("Rolling MATLAB source uses N=8; refusing to build a legacy N!=8 model.")

    node = Elements_Nodes()
    bar = Vec_Elements_Bars()
    actBar = Std_Elements_Bars()
    cst = Vec_Elements_CST()
    rot_spr_4N = Vec_Elements_RotSprings_4N()

    assembly = Assembly_Rolling_Bridge()
    assembly.node = node
    assembly.bar = bar
    assembly.actBar = actBar
    assembly.cst = cst
    assembly.rot_spr_4N = rot_spr_4N

    coords = [
        [0.0, 0.0, 0.0],
        [L / 2.0, 0.0, H],
        [L, 0.0, 0.0],
        [0.0, W, 0.0],
        [L, W, 0.0],
        [L / 2.0, W, H],
        [L, 0.0, H],
        [L, W, H],
    ]
    for i in range(2, N):
        coords += [
            [L * i, 0.0, 0.0],
            [L * i - L / 2.0, 0.0, H],
            [L * i, W, 0.0],
            [L * i - L / 2.0, W, H],
            [L * i, 0.0, H],
            [L * i, W, H],
        ]
    coords += [
        [L * N, 0.0, 0.0],
        [L * N - L / 2.0, 0.0, H],
        [L * N, W, 0.0],
        [L * N - L / 2.0, W, H],
    ]
    node.coordinates_mat = np.array(coords, dtype=float)

    tris = [[1, 3, 4], [3, 4, 5]]
    for i in range(2, N + 1):
        b = 6 * (i - 2)
        tris += [[b + 3, b + 5, b + 9], [b + 5, b + 9, b + 11]]
    cst.node_ijk_mat = np.array(tris, dtype=int)
    cst.v_vec = panel_v * np.ones(len(tris), dtype=float)
    cst.E_vec = panel_E * np.ones(len(tris), dtype=float)
    cst.t_vec = panel_t * np.ones(len(tris), dtype=float)

    bars = [
        [1, 2], [1, 3], [2, 3],
        [4, 5], [4, 6], [5, 6],
        [2, 7], [6, 8],
        [1, 4], [3, 5], [3, 4],
    ]
    for i in range(2, N):
        b = 6 * (i - 2)
        bars += [
            [b + 3, b + 9], [b + 3, b + 10], [b + 9, b + 10],
            [b + 5, b + 11], [b + 5, b + 12], [b + 11, b + 12],
            [b + 7, b + 10], [b + 8, b + 12],
            [b + 10, b + 13], [b + 12, b + 14],
            [b + 9, b + 11], [b + 5, b + 9],
        ]
    b = 6 * (N - 2)
    bars += [
        [b + 3, b + 9], [b + 3, b + 10], [b + 9, b + 10],
        [b + 5, b + 11], [b + 5, b + 12], [b + 11, b + 12],
        [b + 7, b + 10], [b + 8, b + 12],
        [b + 5, b + 9], [b + 9, b + 11],
    ]
    primary_count = len(bars)

    bars += [[2, 6], [7, 8]]
    for i in range(1, N):
        b = 6 * (i - 2)
        bars += [[b + 10, b + 12], [b + 13, b + 14], [b + 16, b + 18]]

    bar.node_ij_mat = np.array(bars, dtype=int)
    bar.A_vec = barA * np.ones(len(bars), dtype=float)
    if barA_brace is not None:
        bar.A_vec[primary_count:] = barA_brace
    bar.E_vec = barE * np.ones(len(bars), dtype=float)

    act = []
    for i in range(1, N):
        b = 6 * (i - 1)
        act += [[b + 3, b + 7], [b + 5, b + 8]]
    actBar.node_ij_mat = np.array(act, dtype=int)
    actBar.A_vec = barA * np.ones(len(act), dtype=float)
    actBar.E_vec = activeBarE * np.ones(len(act), dtype=float)

    spr4 = [[4, 1, 3, 2], [3, 4, 5, 6]]
    for i in range(1, N):
        b = 6 * (i - 1)
        spr4 += [[b + 5, b + 3, b + 9, b + 10], [b + 9, b + 5, b + 11, b + 12]]

    spr4 += [[1, 3, 4, 5]]
    for i in range(1, N):
        b = 6 * (i - 1)
        spr4 += [[b + 3, b + 9, b + 5, b + 11]]

    spr4 += [[1, 2, 3, 7], [4, 5, 6, 8]]
    for i in range(1, N - 1):
        b = 6 * (i - 1)
        spr4 += [
            [b + 7, b + 3, b + 10, b + 9],
            [b + 3, b + 9, b + 10, b + 13],
            [b + 8, b + 5, b + 12, b + 11],
            [b + 5, b + 11, b + 12, b + 14],
        ]
    i = N - 1
    b = 6 * (i - 1)
    spr4 += [[b + 7, b + 3, b + 10, b + 9], [b + 8, b + 5, b + 12, b + 11]]

    rot_spr_4N.node_ijkl_mat = np.array(spr4, dtype=int)
    rot_spr_4N.rot_spr_K_vec = rotK * np.ones(len(spr4), dtype=float)

    plots = Plot_Rolling_Bridge()
    plots.assembly = assembly
    plots.displayRange = np.array([-0.5, 2.0 * N + 0.5, -0.5, 2.5, -0.5, 2.5], dtype=float)
    plots.viewAngle1 = 20
    plots.viewAngle2 = 20

    return assembly, node, bar, actBar, cst, rot_spr_4N, plots
