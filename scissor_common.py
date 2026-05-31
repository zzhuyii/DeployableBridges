import os
from dataclasses import dataclass
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

    if designCode == 'AASHTO':
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


def bridge_self_weight(node, bar=None):
    if bar is None:
        # Legacy signature: bridge_self_weight(model)
        model = node
        node = model.node
        bar = model.bar
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


# This code generates the standard scissor bridge assembly
def build_scissor1_model(N, L=2.0,
    barA=0.0023,
    barA_brace=0.0023,
    barE=2.0e11,
    panel_E=2.0e8,
    panel_t=0.01,
    panel_v=0.3,
    Iy=1.88e-6,
    rot3_factor=100.0,
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

    # Coordinates: 8 nodes per panel, 4 end nodes
    coords = []
    for i in range(1, N + 1):
        x0 = L * (i - 1)
        coords += [
            [x0,     0,   0], [x0,     L,   0],
            [x0,     0,   L], [x0,     L,   L],
            [x0+L/2, 0, L/2], [x0+L/2, L, L/2],
            [x0+L/2, 0,   0], [x0+L/2, L,   0],
        ]
    coords += [
        [L*N, 0, 0], [L*N, L, 0],
        [L*N, 0, L], [L*N, L, L],
    ]
    node.coordinates_mat = np.array(coords, dtype=float)

    # CST panels: 4 per panel
    tris = []
    for i in range(1, N + 1):
        b = 8 * (i - 1)
        tris += [
            [b+1, b+2,  b+7],
            [b+2, b+7,  b+8],
            [b+7, b+8,  b+9],
            [b+8, b+10, b+9],
        ]
    cst.node_ijk_mat = np.array(tris, dtype=int)
    cst.t_vec = panel_t * np.ones(len(tris), dtype=float)
    cst.E_vec = panel_E * np.ones(len(tris), dtype=float)
    cst.v_vec = panel_v * np.ones(len(tris), dtype=float)

    # Structural bars
    bars = []
    areas = []
    for i in range(1, N + 1):
        b = 8 * (i - 1)
        base = [
            [b+1, b+7], [b+7, b+9],  [b+2, b+8],  [b+8, b+10],
            [b+3, b+4], [b+9, b+10], [b+1, b+2],  [b+7, b+8],
            [b+1, b+5], [b+3, b+5],  [b+2, b+6],  [b+4, b+6],
            [b+5, b+9], [b+5, b+11], [b+6, b+10], [b+6, b+12],
            [b+2, b+7],
        ]
        if load_case:
            base += [[b+1, b+8], [b+8, b+9], [b+7, b+10]]
            bars += base
            areas += [barA] * 4 + [barA_brace] * 4 + [barA] * 8 + [barA_brace] * 4
        else:
            base += [[b+8, b+9]]
            bars += base
    end = 8 * N
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
        b = 8 * (i - 1)
        act_rows += [[b+1, b+3], [b+2, b+4]]
    end = 8 * N
    act_rows += [[end+1, end+3], [end+2, end+4]]
    for i in range(1, N + 1):
        b = 8 * (i - 1)
        act_rows += [[b+7, b+5], [b+8, b+6]]
    act_bar.node_ij_mat = np.array(act_rows, dtype=int)
    act_bar.A_vec = barA * np.ones(len(act_rows), dtype=float)
    act_bar.E_vec = barE * np.ones(len(act_rows), dtype=float)

    # Rotational springs 3N: 12 per panel
    spr3 = []
    for i in range(1, N + 1):
        b = 8 * (i - 1)
        spr3 += [
            [b+1, b+5,  b+11], [b+3, b+5,  b+9],
            [b+2, b+6,  b+12], [b+4, b+6,  b+10],
            [b+4, b+3,  b+5],  [b+3, b+4,  b+6],
            [b+2, b+1,  b+5],  [b+6, b+2,  b+1],
            [b+5, b+11, b+12], [b+11, b+12, b+6],
            [b+5, b+9,  b+10], [b+9,  b+10, b+6],
        ]
    rot3.node_ijk_mat = np.array(spr3, dtype=int)
    rot3.rot_spr_K_vec = kspr * rot3_factor * np.ones(len(spr3), dtype=float)
    rot3.delta = 1e-3

    # Rotational springs 4N: 2 per panel
    spr4 = []
    for i in range(1, N + 1):
        b = 8 * (i - 1)
        spr4 += [
            [b+1, b+2, b+7, b+8],
            [b+7, b+8, b+9, b+10],
        ]
    rot4.node_ijkl_mat = np.array(spr4, dtype=int)
    rot4.rot_spr_K_vec = rot4K * np.ones(len(spr4), dtype=float)

    plots = Plot_Scissor_Bridge()
    plots.assembly = assembly
    plots.displayRange = np.array([-1, L*N+1, -1, L+1, -1, L+1], dtype=float)
    plots.viewAngle1 = 20
    plots.viewAngle2 = 20

    return assembly, node, bar, act_bar, cst, rot3, rot4, plots


# Deploy supports: both ends pinned for the standard bridge
def standard_deploy_supports(node_num, N):
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
    end = 8 * N
    supp[end,   :] = [end,   0, 1, 1]
    supp[end+1, :] = [end+1, 0, 1, 1]
    supp[end+2, :] = [end+2, 0, 1, 0]
    supp[end+3, :] = [end+3, 0, 1, 0]
    return supp


# Actuator displacement magnitude for a given integer source step
def standard_deploy_delta(source_step):
    if source_step <= 200:
        return 0.001 * source_step
    if source_step <= 400:
        return 0.001 * 200 + 0.0004 * (source_step - 200)
    return 0.001 * 200 + 0.0004 * 200 + 0.0001 * (source_step - 400)


def load_supports(N, stride):
    end = stride * N
    return np.asarray([
        [0,     1, 1, 1],
        [1,     1, 1, 1],
        [end,   0, 1, 1],
        [end+1, 0, 1, 1],
    ], dtype=float)


def source_step_for_frame(frame_index, frame_count, source_step_count):
    if frame_count <= 1:
        return source_step_count
    return int(round(1 + frame_index * (source_step_count - 1) / (frame_count - 1)))


def actuator_target_lengths(base_L0, dL, L, act_bar_num_1, act_bar_count):
    target = np.asarray(base_L0, dtype=float).copy()
    target[:act_bar_num_1] = base_L0[:act_bar_num_1] + dL
    theta = np.arccos(np.clip((L + dL) / np.sqrt(2.0) / L, -1.0, 1.0))
    L2 = L / np.sqrt(2.0) * np.sin(theta)
    L3 = np.sqrt(max((L / 2.0) ** 2 - L2 ** 2, 0.0))
    target[act_bar_num_1:act_bar_count] = base_L0[act_bar_num_1:act_bar_count] + dL / 2.0 - L3
    return target


def save_figure(fig, path, dpi=220):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def save_geometry_diagnostics(model, out_dir, prefix):
    save_figure(model.plots.Plot_Shape_Node_Number(),      os.path.join(out_dir, f"{prefix}_Node_Number.png"))
    save_figure(model.plots.Plot_Shape_CST_Number(),       os.path.join(out_dir, f"{prefix}_CST_Number.png"))
    save_figure(model.plots.Plot_Shape_Bar_Number(),       os.path.join(out_dir, f"{prefix}_Bar_Number.png"))
    save_figure(model.plots.Plot_Shape_RotSpr_3N_Number(), os.path.join(out_dir, f"{prefix}_RotSpr_3N_Number.png"))
    save_figure(model.plots.Plot_Shape_RotSpr_4N_Number(), os.path.join(out_dir, f"{prefix}_RotSpr_4N_Number.png"))
    save_figure(model.plots.Plot_Shape_ActBar_Number(),    os.path.join(out_dir, f"{prefix}_ActBar_Number.png"))


# ---------------------------------------------------------------------------
# Backward-compatibility section — used by Scissor_Bridge_*.py callers
# ---------------------------------------------------------------------------

@dataclass
class ScissorModel:
    assembly: Assembly_Scissor_Bridge
    node: Elements_Nodes
    cst: Vec_Elements_CST
    bar: Vec_Elements_Bars
    act_bar: Std_Elements_Bars
    rot3: CD_Elements_RotSprings_3N
    rot4: Vec_Elements_RotSprings_4N
    plots: Plot_Scissor_Bridge
    settings: dict


def as_int_array(rows):
    return np.asarray(rows, dtype=int)


def local_buckling_message(E, Fy, bt=10.7, ht=24.5):
    lambda_r = 1.28 * np.sqrt(E / Fy)
    passed = (bt <= lambda_r) and (ht <= lambda_r)
    status = "Section is non-slender (local buckling OK)" if passed else "WARNING: Section fails local buckling"
    return passed, lambda_r, status


def build_scissor_model(variant="standard", analysis="deploy", N=None, L=2.0):
    improved = variant == "improved"
    load_case = analysis == "load"
    if N is None:
        N = 8

    H = L
    barE = 2.0e11
    panel_E = 2.0e8
    panel_t = 0.01
    panel_v = 0.3

    if load_case:
        barA = 0.00415
        barA_brace = 0.0019
        Iy = 21.2e-6
        Ix = 7.16e-6
        kspr = barE * Iy / np.sqrt(H**2 + L**2)
        rot3_factor = 1.0
    else:
        barA = 0.0023
        barA_brace = barA
        Iy = 1.88e-6
        Ix = 1.88e-6
        kspr = barE * Iy / np.sqrt(H**2 + L**2)
        rot3_factor = 10.0 if improved else 100.0

    node = Elements_Nodes()
    node.coordinates_mat = _standard_coordinates(N, L)

    cst = Vec_Elements_CST()
    cst.node_ijk_mat = as_int_array(_standard_cst(N))
    cst.t_vec = panel_t * np.ones(cst.node_ijk_mat.shape[0])
    cst.E_vec = panel_E * np.ones(cst.node_ijk_mat.shape[0])
    cst.v_vec = panel_v * np.ones(cst.node_ijk_mat.shape[0])


    bar_rows, bar_areas = _standard_bars(N, load_case, barA, barA_brace)
    rot3_rows = _standard_rot3(N)
    rot4_rows = _standard_rot4(N)
    act_rows, act_bar_num_1 = _standard_act_bars(N)
    stride = 8

    bar = Vec_Elements_Bars()
    bar.node_ij_mat = as_int_array(bar_rows)
    bar.A_vec = np.asarray(bar_areas, dtype=float)
    bar.E_vec = barE * np.ones(bar.node_ij_mat.shape[0])

    act_bar = Std_Elements_Bars()
    act_bar.node_ij_mat = as_int_array(act_rows)
    act_bar.A_vec = barA * np.ones(act_bar.node_ij_mat.shape[0])
    act_bar.E_vec = barE * np.ones(act_bar.node_ij_mat.shape[0])

    rot3 = CD_Elements_RotSprings_3N()
    rot3.node_ijk_mat = as_int_array(rot3_rows)
    rot3.rot_spr_K_vec = kspr * rot3_factor * np.ones(rot3.node_ijk_mat.shape[0])
    rot3.delta = 1e-3

    rot4 = Vec_Elements_RotSprings_4N()
    rot4.node_ijkl_mat = as_int_array(rot4_rows)
    rot4.rot_spr_K_vec = 100000.0 * np.ones(rot4.node_ijkl_mat.shape[0])

    assembly = Assembly_Scissor_Bridge()
    assembly.node = node
    assembly.cst = cst
    assembly.bar = bar
    assembly.actBar = act_bar
    assembly.rot_spr_3N = rot3
    assembly.rot_spr_4N = rot4

    plots = Plot_Scissor_Bridge()
    plots.assembly = assembly
    plots.displayRange = np.asarray([-1, L*N+1, -1, L+1, -1, L+1], dtype=float)
    plots.viewAngle1 = 20
    plots.viewAngle2 = 20

    settings = {
        "variant": variant,
        "analysis": analysis,
        "N": N,
        "H": H,
        "L": L,
        "stride": stride,
        "barA": barA,
        "barA_brace": barA_brace,
        "barE": barE,
        "Iy": Iy,
        "Ix": Ix,
        "panel_E": panel_E,
        "panel_t": panel_t,
        "panel_v": panel_v,
        "act_bar_num_1": act_bar_num_1,
    }

    return ScissorModel(assembly, node, cst, bar, act_bar, rot3, rot4, plots, settings)


def _standard_coordinates(N, L):
    coords = []
    for i in range(1, N + 1):
        x0 = L * (i - 1)
        coords += [
            [x0,     0,   0], [x0,     L,   0],
            [x0,     0,   L], [x0,     L,   L],
            [x0+L/2, 0, L/2], [x0+L/2, L, L/2],
            [x0+L/2, 0,   0], [x0+L/2, L,   0],
        ]
    coords += [
        [L*N, 0, 0], [L*N, L, 0],
        [L*N, 0, L], [L*N, L, L],
    ]
    return np.asarray(coords, dtype=float)


def _standard_cst(N):
    rows = []
    for i in range(1, N + 1):
        b = 8 * (i - 1)
        rows += [
            [b+1, b+2,  b+7],
            [b+2, b+7,  b+8],
            [b+7, b+8,  b+9],
            [b+8, b+10, b+9],
        ]
    return rows


def _standard_bars(N, load_case, barA, barA_brace):
    rows = []
    areas = []
    for i in range(1, N + 1):
        b = 8 * (i - 1)
        base_rows = [
            [b+1, b+7], [b+7, b+9],  [b+2, b+8],  [b+8, b+10],
            [b+3, b+4], [b+9, b+10], [b+1, b+2],  [b+7, b+8],
            [b+1, b+5], [b+3, b+5],  [b+2, b+6],  [b+4, b+6],
            [b+5, b+9], [b+5, b+11], [b+6, b+10], [b+6, b+12],
            [b+2, b+7],
        ]
        if load_case:
            base_rows += [[b+1, b+8], [b+8, b+9], [b+7, b+10]]
            rows += base_rows
            areas += [barA] * 4 + [barA_brace] * 4 + [barA] * 8 + [barA_brace] * 4
        else:
            base_rows += [[b+8, b+9]]
            rows += base_rows
    end = 8 * N
    rows += [[end+1, end+2], [end+3, end+4]]
    if load_case:
        areas += [barA_brace, barA_brace]
    else:
        areas = [barA] * len(rows)
    return rows, areas


def _standard_rot3(N):
    rows = []
    for i in range(1, N + 1):
        b = 8 * (i - 1)
        rows += [
            [b+1, b+5,  b+11], [b+3, b+5,  b+9],
            [b+2, b+6,  b+12], [b+4, b+6,  b+10],
            [b+4, b+3,  b+5],  [b+3, b+4,  b+6],
            [b+2, b+1,  b+5],  [b+6, b+2,  b+1],
            [b+5, b+11, b+12], [b+11, b+12, b+6],
            [b+5, b+9,  b+10], [b+9,  b+10, b+6],
        ]
    return rows


def _standard_rot4(N):
    rows = []
    for i in range(1, N + 1):
        b = 8 * (i - 1)
        rows += [
            [b+1, b+2, b+7, b+8],
            [b+7, b+8, b+9, b+10],
        ]
    return rows


def _standard_act_bars(N):
    rows = []
    for i in range(1, N + 1):
        b = 8 * (i - 1)
        rows += [[b+1, b+3], [b+2, b+4]]
    end = 8 * N
    rows += [[end+1, end+3], [end+2, end+4]]
    act_bar_num_1 = len(rows)
    for i in range(1, N + 1):
        b = 8 * (i - 1)
        rows += [[b+7, b+5], [b+8, b+6]]
    return rows, act_bar_num_1
