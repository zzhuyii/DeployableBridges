import numpy as np

from Assembly_Scissor_Bridge import Assembly_Scissor_Bridge
from CD_Elements_RotSprings_3N import CD_Elements_RotSprings_3N
from Elements_Nodes import Elements_Nodes
from Plot_Scissor_Bridge import Plot_Scissor_Bridge
from Std_Elements_Bars import Std_Elements_Bars
from Vec_Elements_Bars import Vec_Elements_Bars
from Vec_Elements_CST import Vec_Elements_CST
from Vec_Elements_RotSprings_4N import Vec_Elements_RotSprings_4N

from AASHTO_Checks import check_truss_lrfd
from AREMA_Checks import arema_member_check


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


def bridge_self_weight(node,bar):
    rho_steel = 7850.0
    g = 9.81
    L_total = 0.0
    W_bar = 0.0
    coords = node.coordinates_mat
    for i, (n1, n2) in enumerate(bar.node_ij_mat):
        length = np.linalg.norm(coords[n1 - 1, :] - coords[n2 - 1, :])
        L_total += length
        W_bar += length * bar.A_vec[i] * rho_steel * g
    return L_total, W_bar



# This code generates the improved scissor bridge assembly
def build_scissor2_model(N, L=2.0,
    barA=0.0023,
    barA_brace=0.0023,
    barE=2.0e11,
    panel_E=2.0e8,
    panel_t=0.01,
    panel_v=0.3,
    Iy=1.88e-6,
    rot3_factor=10.0,
    rot4K=100000.0,
    load_case=False,
):
    H = L
    kspr = barE * Iy / np.sqrt(H**2 + L**2)

    node = Elements_Nodes()
    cst = Vec_Elements_CST()
    bar = Vec_Elements_Bars()
    act_bar = Std_Elements_Bars()
    rot3 = CD_Elements_RotSprings_3N()
    rot4 = Vec_Elements_RotSprings_4N()

    assembly = Assembly_Scissor_Bridge()
    assembly.node = node
    assembly.cst = cst
    assembly.bar = bar
    assembly.actBar = act_bar
    assembly.rot_spr_3N = rot3
    assembly.rot_spr_4N = rot4

    # Coordinates: 10 nodes per panel, 4 end nodes
    coords = []
    for i in range(1, N + 1):
        x0 = L * (i - 1)
        coords += [
            [x0,       0, 0],   [x0,       L, 0],
            [x0,       0, L],   [x0,       L, L],
            [x0+L/2,   0, L/2], [x0+L/2,   L, L/2],
            [x0+L/2,   0, 0],   [x0+L/2,   L, 0],
            [x0+L/2,   0, L],   [x0+L/2,   L, L],
        ]
    coords += [
        [L*N, 0, 0], [L*N, L, 0],
        [L*N, 0, L], [L*N, L, L],
    ]
    node.coordinates_mat = np.array(coords, dtype=float)

    # CST panels: 4 per panel
    tris = []
    for i in range(1, N + 1):
        b = 10 * (i - 1)
        tris += [
            [b+1,  b+2,  b+7],
            [b+2,  b+7,  b+8],
            [b+7,  b+8,  b+11],
            [b+8,  b+12, b+11],
        ]
    cst.node_ijk_mat = np.array(tris, dtype=int)
    cst.t_vec = panel_t * np.ones(len(tris), dtype=float)
    cst.E_vec = panel_E * np.ones(len(tris), dtype=float)
    cst.v_vec = panel_v * np.ones(len(tris), dtype=float)

    # Structural bars
    bars = []
    areas = []
    for i in range(1, N + 1):
        b = 10 * (i - 1)
        base = [
            [b+1,  b+7],  [b+7,  b+11], [b+2,  b+8],  [b+8,  b+12],
            [b+3,  b+9],  [b+9,  b+13], [b+4,  b+10], [b+10, b+14],
            [b+3,  b+4],  [b+9,  b+10],
            [b+1,  b+2],  [b+7,  b+8],
            [b+1,  b+5],  [b+3,  b+5],  [b+2,  b+6],  [b+4,  b+6],
            [b+5,  b+13], [b+5,  b+11], [b+6,  b+12], [b+6,  b+14],
            [b+3,  b+10],
        ]
        if load_case:
            base += [
                [b+4,  b+9],  [b+13, b+10], [b+9,  b+14],
                [b+2,  b+7],  [b+1,  b+8],  [b+8,  b+11], [b+7,  b+12],
            ]
            bars += base
            areas += [barA] * 8 + [barA_brace] * 4 + [barA] * 8 + [barA_brace] * 8
        else:
            base += [[b+13, b+10], [b+2, b+7], [b+8, b+11]]
            bars += base
    end = 10 * N
    bars += [[end+1, end+2], [end+3, end+4]]
    if load_case:
        areas += [barA_brace, barA_brace]
    else:
        areas = [barA] * len(bars)
    bar.node_ij_mat = np.array(bars, dtype=int)
    bar.A_vec = np.array(areas, dtype=float)
    bar.E_vec = barE * np.ones(len(bars), dtype=float)

    # Actuator bars
    act_rows = []
    for i in range(1, N + 1):
        b = 10 * (i - 1)
        act_rows += [[b+1, b+3], [b+2, b+4]]
    end = 10 * N
    act_rows += [[end+1, end+3], [end+2, end+4]]
    for i in range(1, N + 1):
        b = 10 * (i - 1)
        act_rows += [[b+7, b+5], [b+5, b+9], [b+6, b+10], [b+8, b+6]]
    act_bar.node_ij_mat = np.array(act_rows, dtype=int)
    act_bar.A_vec = barA * np.ones(len(act_rows), dtype=float)
    act_bar.E_vec = barE * np.ones(len(act_rows), dtype=float)

    # Rotational springs 3N: 16 per panel
    spr3 = []
    for i in range(1, N + 1):
        b = 10 * (i - 1)
        spr3 += [
            [b+1,  b+5,  b+13], [b+3,  b+5,  b+11],
            [b+2,  b+6,  b+14], [b+4,  b+6,  b+12],
            [b+4,  b+3,  b+5],  [b+3,  b+4,  b+6],
            [b+2,  b+1,  b+5],  [b+6,  b+2,  b+1],
            [b+5,  b+11, b+12], [b+11, b+12, b+6],
            [b+5,  b+13, b+14], [b+13, b+14, b+6],
            [b+11, b+13, b+14], [b+13, b+14, b+12],
            [b+14, b+12, b+11], [b+12, b+11, b+13],
        ]
    rot3.node_ijk_mat = np.array(spr3, dtype=int)
    rot3.rot_spr_K_vec = kspr * rot3_factor * np.ones(len(spr3), dtype=float)
    rot3.delta = 1e-3

    # Rotational springs 4N: 4 per panel
    spr4 = []
    for i in range(1, N + 1):
        b = 10 * (i - 1)
        spr4 += [
            [b+1,  b+2,  b+7,  b+8],
            [b+7,  b+8,  b+11, b+12],
            [b+4,  b+3,  b+10, b+9],
            [b+10, b+9,  b+14, b+13],
        ]
    rot4.node_ijkl_mat = np.array(spr4, dtype=int)
    rot4.rot_spr_K_vec = rot4K * np.ones(len(spr4), dtype=float)

    plots = Plot_Scissor_Bridge()
    plots.assembly = assembly
    plots.displayRange = np.array([-1, L*N+1, -1, L+1, -1, L+1], dtype=float)
    plots.viewAngle1 = 20
    plots.viewAngle2 = 20

    return assembly, node, bar, act_bar, cst, rot3, rot4, plots


# Deploy supports: pin first two corner nodes, roller on the other two
def improved_deploy_supports(node_num):
    supp = np.column_stack([
        np.arange(node_num),
        np.zeros(node_num),
        np.ones(node_num),
        np.zeros(node_num),
    ])
    supp[0, :] = [0, 1, 1, 1]
    supp[1, :] = [1, 1, 1, 1]
    supp[2, :] = [2, 1, 1, 0]
    supp[3, :] = [3, 1, 1, 0]
    return supp


# Actuator displacement magnitude for a given integer source step
def improved_deploy_delta(source_step):
    return 0.001 * source_step


# ---------------------------------------------------------------------------
# Backward-compatibility helpers — imported by scissor_common.py
# ---------------------------------------------------------------------------

def _improved_coordinates(N, L):
    coords = []
    for i in range(1, N + 1):
        x0 = L * (i - 1)
        coords += [
            [x0,     0, 0],   [x0,     L, 0],
            [x0,     0, L],   [x0,     L, L],
            [x0+L/2, 0, L/2], [x0+L/2, L, L/2],
            [x0+L/2, 0, 0],   [x0+L/2, L, 0],
            [x0+L/2, 0, L],   [x0+L/2, L, L],
        ]
    coords += [
        [L*N, 0, 0], [L*N, L, 0],
        [L*N, 0, L], [L*N, L, L],
    ]
    return np.asarray(coords, dtype=float)


def _improved_cst(N):
    rows = []
    for i in range(1, N + 1):
        b = 10 * (i - 1)
        rows += [
            [b+1, b+2,  b+7],
            [b+2, b+7,  b+8],
            [b+7, b+8,  b+11],
            [b+8, b+12, b+11],
        ]
    return rows


def _improved_bars(N, load_case, barA, barA_brace):
    rows = []
    areas = []
    for i in range(1, N + 1):
        b = 10 * (i - 1)
        base_rows = [
            [b+1,  b+7],  [b+7,  b+11], [b+2,  b+8],  [b+8,  b+12],
            [b+3,  b+9],  [b+9,  b+13], [b+4,  b+10], [b+10, b+14],
            [b+3,  b+4],  [b+9,  b+10],
            [b+1,  b+2],  [b+7,  b+8],
            [b+1,  b+5],  [b+3,  b+5],  [b+2,  b+6],  [b+4,  b+6],
            [b+5,  b+13], [b+5,  b+11], [b+6,  b+12], [b+6,  b+14],
            [b+3,  b+10],
        ]
        if load_case:
            base_rows += [
                [b+4,  b+9],  [b+13, b+10], [b+9,  b+14],
                [b+2,  b+7],  [b+1,  b+8],  [b+8,  b+11], [b+7,  b+12],
            ]
            rows += base_rows
            areas += [barA] * 8 + [barA_brace] * 4 + [barA] * 8 + [barA_brace] * 8
        else:
            base_rows += [[b+13, b+10], [b+2, b+7], [b+8, b+11]]
            rows += base_rows
    end = 10 * N
    rows += [[end+1, end+2], [end+3, end+4]]
    if load_case:
        areas += [barA_brace, barA_brace]
    else:
        areas = [barA] * len(rows)
    return rows, areas


def _improved_rot3(N):
    rows = []
    for i in range(1, N + 1):
        b = 10 * (i - 1)
        rows += [
            [b+1,  b+5,  b+13], [b+3,  b+5,  b+11],
            [b+2,  b+6,  b+14], [b+4,  b+6,  b+12],
            [b+4,  b+3,  b+5],  [b+3,  b+4,  b+6],
            [b+2,  b+1,  b+5],  [b+6,  b+2,  b+1],
            [b+5,  b+11, b+12], [b+11, b+12, b+6],
            [b+5,  b+13, b+14], [b+13, b+14, b+6],
            [b+11, b+13, b+14], [b+13, b+14, b+12],
            [b+14, b+12, b+11], [b+12, b+11, b+13],
        ]
    return rows


def _improved_rot4(N):
    rows = []
    for i in range(1, N + 1):
        b = 10 * (i - 1)
        rows += [
            [b+1,  b+2,  b+7,  b+8],
            [b+7,  b+8,  b+11, b+12],
            [b+4,  b+3,  b+10, b+9],
            [b+10, b+9,  b+14, b+13],
        ]
    return rows


def _improved_act_bars(N):
    rows = []
    for i in range(1, N + 1):
        b = 10 * (i - 1)
        rows += [[b+1, b+3], [b+2, b+4]]
    end = 10 * N
    rows += [[end+1, end+3], [end+2, end+4]]
    act_bar_num_1 = len(rows)
    for i in range(1, N + 1):
        b = 10 * (i - 1)
        rows += [[b+7, b+5], [b+5, b+9], [b+6, b+10], [b+8, b+6]]
    return rows, act_bar_num_1
