import numpy as np
import os

from Elements_Nodes import Elements_Nodes
from Vec_Elements_Bars import Vec_Elements_Bars
from Vec_Elements_CST import Vec_Elements_CST
from Vec_Elements_RotSprings_4N import Vec_Elements_RotSprings_4N
from CD_Elements_RotSprings_3N import CD_Elements_RotSprings_3N
from Assembly_KirigamiTruss import Assembly_KirigamiTruss
from Plot_KirigamiTruss import Plot_KirigamiTruss

from AASHTO_Checks import check_truss_lrfd
from AREMA_Checks import arema_member_check


# This code generate the kirigami bridge assembly
def build_kirigami_truss(L, N,
    barA=0.0023,
    barE=2.0e11,
    panel_E=2.0e8,
    panel_t=0.01,
    panel_v=0.3,
    rot4K=10.0,
    rot3K=1.0e8,
):

    gap=0.0

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
    plots.display_range = np.array([-1.0, L * (N + 1), -1.0, L+1.0, -1.0, L+1.0], dtype=float)
    plots.view_angle1 = 20
    plots.view_angle2 = 20

    return assembly, node, bar, cst, rot_spr_4N, rot_spr_3N, plots


# compute the total bar length and self weight of the bridge
def bar_length_and_weight(node, bar, rho_steel=7850.0, g=9.81):
    total_length = 0.0
    total_weight = 0.0
    for idx, (n1, n2) in enumerate(bar.node_ij_mat):
        length = np.linalg.norm(node.coordinates_mat[n1 - 1] - node.coordinates_mat[n2 - 1])
        total_length += length
        total_weight += length * bar.A_vec[idx] * rho_steel * g
    return total_length, total_weight

# compute deformation matrix during deployment
def deployment_offset(node_count, dep_rate, N):
    deploy_path = os.path.abspath("KirigamiUhis.npy")

    Uhis = np.load(deploy_path)
    print(Uhis.shape)
    Uhis = Uhis[:,0:(N*16+4),:]
    
    dep_step = max(1, int((1.0 - dep_rate) * Uhis.shape[0]))
    idx = min(Uhis.shape[0], dep_step) - 1
    print(f"Using deployment history step {idx + 1}/{Uhis.shape[0]} from {deploy_path}")
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