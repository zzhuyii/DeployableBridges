import numpy as np

from Elements_Nodes import Elements_Nodes
from Vec_Elements_Bars import Vec_Elements_Bars
from Std_Elements_Bars import Std_Elements_Bars
from Vec_Elements_CST import Vec_Elements_CST
from Vec_Elements_RotSprings_4N import Vec_Elements_RotSprings_4N

from Assembly_Rolling_Bridge import Assembly_Rolling_Bridge
from Plot_Rolling_Bridge import Plot_Rolling_Bridge

from AASHTO_Checks import check_truss_lrfd
from AREMA_Checks import arema_member_check


def bar_length_and_weight(node, bar, rho_steel=7850.0, g=9.81):
    total_length = 0.0
    total_weight = 0.0
    for idx, (n1, n2) in enumerate(bar.node_ij_mat):
        length = np.linalg.norm(node.coordinates_mat[n1 - 1] - node.coordinates_mat[n2 - 1])
        total_length += length
        total_weight += length * bar.A_vec[idx] * rho_steel * g
    return total_length, total_weight


def rolling_deploy_offset(node_count, dep_rate,N):
    # deploy_path = os.path.abspath("RollingUhis.npy")

    Uhis = np.load("RollingUhis.npz")["Uhis"]
    print(Uhis.shape)
    
    dep_step = max(1, int((1.0 - dep_rate) * Uhis.shape[0]))
    idx = min(Uhis.shape[0], dep_step) - 1
    
    Uhis = Uhis[:,0:(N*6),:]
    
    print(f"Using rolling deployment step {idx + 1}/{Uhis.shape[0]}")
    return Uhis[idx] 


def check_members(bar, node, U_end, An, r_val, Fy, Fu, Rp, designCode):
    truss_strain = bar.solve_strain(node, U_end)
    internal_force = truss_strain * bar.E_vec * bar.A_vec
    Lc = bar.L0_vec.reshape(-1)
    pass_yn = np.zeros(internal_force.size, dtype=bool)
    dcr = np.full(internal_force.size, np.nan, dtype=float)
    
    if designCode=='AASHTO':
        for j, Pu in enumerate(1.5 * internal_force):
            passed, _, _, _, _, dcr_j = check_truss_lrfd(
                Pu, bar.A_vec[j], An, bar.E_vec[j], Lc[j], r_val, Fy, Fu, Rp
            )
            pass_yn[j] = passed
            dcr[j] = dcr_j
    else:
        for j, Pu in enumerate(internal_force):
            passed, dcr_j = arema_member_check(
                Pu, bar.A_vec[j], An, bar.E_vec[j], Lc[j], r_val, Fy, Fu, Rp
            )
            pass_yn[j] = passed
            dcr[j] = dcr_j
    return truss_strain, pass_yn, dcr



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
    plots.displayRange = np.array([-0.5, L * N + 0.5, -0.5, L+0.5, -0.5, L+0.5], dtype=float)
    plots.viewAngle1 = 20
    plots.viewAngle2 = 20

    return assembly, node, bar, actBar, cst, rot_spr_4N, plots
