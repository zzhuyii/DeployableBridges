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


def save_figure(fig, path, dpi=220):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")


def check_truss_lrfd(Pu, Ag, An, E, KL, r, Fy, Fu, Rp):
    phi_ty = 0.95
    phi_cb = 0.95
    phi_uf = 0.80
    U_shlag = 1.0
    KLr = KL / max(r, 1e-12)

    if Pu >= 0:
        Pny = Fy * Ag
        Pr_yield = phi_ty * Pny
        Pnu = Fu * An * Rp * U_shlag
        Pr_fracture = phi_uf * Pnu
        if Pr_yield <= Pr_fracture:
            Pn = Pny
            phi = phi_ty
            phiPn = Pr_yield
            mode = "Tension-Yield [AASHTO 6.8.2.1-1]"
        else:
            Pn = Pnu
            phi = phi_uf
            phiPn = Pr_fracture
            mode = "Tension-Fracture [AASHTO 6.8.2.1-2]"
    else:
        Po = Fy * Ag
        Pe = (np.pi ** 2 * E * Ag) / max(KLr ** 2, 1e-12)
        ratio = Po / max(Pe, 1e-12)
        if ratio <= 2.25:
            Pn = (0.658 ** ratio) * Po
            mode = "Compression-Inelastic [AASHTO 6.9.4.1.1-1]"
        else:
            Pn = 0.877 * Pe
            mode = "Compression-Elastic [AASHTO 6.9.4.1.1-2]"
        phi = phi_cb
        phiPn = phi * Pn

    DCR = abs(Pu) / max(abs(phiPn), 1e-12)
    return DCR <= 1.0, mode, Pn, phi, phiPn, DCR


def local_buckling_message(E, Fy, bt=10.7, ht=24.5):
    lambda_r = 1.28 * np.sqrt(E / Fy)
    passed = (bt <= lambda_r) and (ht <= lambda_r)
    status = "Section is non-slender (local buckling OK)" if passed else "WARNING: Section fails local buckling"
    return passed, lambda_r, status


def _standard_coordinates(N, L):
    coords = []
    for i in range(1, N + 1):
        x0 = L * (i - 1)
        coords += [
            [x0, 0, 0],
            [x0, L, 0],
            [x0, 0, L],
            [x0, L, L],
            [x0 + L / 2, 0, L / 2],
            [x0 + L / 2, L, L / 2],
            [x0 + L / 2, 0, 0],
            [x0 + L / 2, L, 0],
        ]
    coords += [
        [L * N, 0, 0],
        [L * N, L, 0],
        [L * N, 0, L],
        [L * N, L, L],
    ]
    return np.asarray(coords, dtype=float)


def _improved_coordinates(N, L):
    coords = []
    for i in range(1, N + 1):
        x0 = L * (i - 1)
        coords += [
            [x0, 0, 0],
            [x0, L, 0],
            [x0, 0, L],
            [x0, L, L],
            [x0 + L / 2, 0, L / 2],
            [x0 + L / 2, L, L / 2],
            [x0 + L / 2, 0, 0],
            [x0 + L / 2, L, 0],
            [x0 + L / 2, 0, L],
            [x0 + L / 2, L, L],
        ]
    coords += [
        [L * N, 0, 0],
        [L * N, L, 0],
        [L * N, 0, L],
        [L * N, L, L],
    ]
    return np.asarray(coords, dtype=float)


def _standard_cst(N):
    rows = []
    for i in range(1, N + 1):
        b = 8 * (i - 1)
        rows += [
            [b + 1, b + 2, b + 7],
            [b + 2, b + 7, b + 8],
            [b + 7, b + 8, b + 9],
            [b + 8, b + 10, b + 9],
        ]
    return rows


def _improved_cst(N):
    rows = []
    for i in range(1, N + 1):
        b = 10 * (i - 1)
        rows += [
            [b + 1, b + 2, b + 7],
            [b + 2, b + 7, b + 8],
            [b + 7, b + 8, b + 11],
            [b + 8, b + 12, b + 11],
        ]
    return rows


def _standard_bars(N, load_case, barA, barA_brace):
    rows = []
    areas = []
    for i in range(1, N + 1):
        b = 8 * (i - 1)
        base_rows = [
            [b + 1, b + 7], [b + 7, b + 9], [b + 2, b + 8], [b + 8, b + 10],
            [b + 3, b + 4], [b + 9, b + 10], [b + 1, b + 2], [b + 7, b + 8],
            [b + 1, b + 5], [b + 3, b + 5], [b + 2, b + 6], [b + 4, b + 6],
            [b + 5, b + 9], [b + 5, b + 11], [b + 6, b + 10], [b + 6, b + 12],
            [b + 2, b + 7],
        ]
        if load_case:
            base_rows += [[b + 1, b + 8], [b + 8, b + 9], [b + 7, b + 10]]
            rows += base_rows
            areas += [barA] * 4 + [barA_brace] * 4 + [barA] * 8 + [barA_brace] * 4
        else:
            base_rows += [[b + 8, b + 9]]
            rows += base_rows
    end = 8 * N
    rows += [[end + 1, end + 2], [end + 3, end + 4]]
    if load_case:
        areas += [barA_brace, barA_brace]
    else:
        areas = [barA] * len(rows)
    return rows, areas


def _improved_bars(N, load_case, barA, barA_brace):
    rows = []
    areas = []
    for i in range(1, N + 1):
        b = 10 * (i - 1)
        base_rows = [
            [b + 1, b + 7], [b + 7, b + 11], [b + 2, b + 8], [b + 8, b + 12],
            [b + 3, b + 9], [b + 9, b + 13], [b + 4, b + 10], [b + 10, b + 14],
            [b + 3, b + 4], [b + 9, b + 10],
            [b + 1, b + 2], [b + 7, b + 8],
            [b + 1, b + 5], [b + 3, b + 5], [b + 2, b + 6], [b + 4, b + 6],
            [b + 5, b + 13], [b + 5, b + 11], [b + 6, b + 12], [b + 6, b + 14],
            [b + 3, b + 10],
        ]
        if load_case:
            base_rows += [
                [b + 4, b + 9], [b + 13, b + 10], [b + 9, b + 14],
                [b + 2, b + 7], [b + 1, b + 8], [b + 8, b + 11], [b + 7, b + 12],
            ]
            rows += base_rows
            areas += [barA] * 8 + [barA_brace] * 4 + [barA] * 8 + [barA_brace] * 8
        else:
            base_rows += [[b + 13, b + 10], [b + 2, b + 7], [b + 8, b + 11]]
            rows += base_rows
    end = 10 * N
    rows += [[end + 1, end + 2], [end + 3, end + 4]]
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
            [b + 1, b + 5, b + 11], [b + 3, b + 5, b + 9],
            [b + 2, b + 6, b + 12], [b + 4, b + 6, b + 10],
            [b + 4, b + 3, b + 5], [b + 3, b + 4, b + 6],
            [b + 2, b + 1, b + 5], [b + 6, b + 2, b + 1],
            [b + 5, b + 11, b + 12], [b + 11, b + 12, b + 6],
            [b + 5, b + 9, b + 10], [b + 9, b + 10, b + 6],
        ]
    return rows


def _improved_rot3(N):
    rows = []
    for i in range(1, N + 1):
        b = 10 * (i - 1)
        rows += [
            [b + 1, b + 5, b + 13], [b + 3, b + 5, b + 11],
            [b + 2, b + 6, b + 14], [b + 4, b + 6, b + 12],
            [b + 4, b + 3, b + 5], [b + 3, b + 4, b + 6],
            [b + 2, b + 1, b + 5], [b + 6, b + 2, b + 1],
            [b + 5, b + 11, b + 12], [b + 11, b + 12, b + 6],
            [b + 5, b + 13, b + 14], [b + 13, b + 14, b + 6],
            [b + 11, b + 13, b + 14], [b + 13, b + 14, b + 12],
            [b + 14, b + 12, b + 11], [b + 12, b + 11, b + 13],
        ]
    return rows


def _standard_rot4(N):
    rows = []
    for i in range(1, N + 1):
        b = 8 * (i - 1)
        rows += [[b + 1, b + 2, b + 7, b + 8], [b + 7, b + 8, b + 9, b + 10]]
    return rows


def _improved_rot4(N):
    rows = []
    for i in range(1, N + 1):
        b = 10 * (i - 1)
        rows += [
            [b + 1, b + 2, b + 7, b + 8],
            [b + 7, b + 8, b + 11, b + 12],
            [b + 4, b + 3, b + 10, b + 9],
            [b + 10, b + 9, b + 14, b + 13],
        ]
    return rows


def _standard_act_bars(N):
    rows = []
    for i in range(1, N + 1):
        b = 8 * (i - 1)
        rows += [[b + 1, b + 3], [b + 2, b + 4]]
    end = 8 * N
    rows += [[end + 1, end + 3], [end + 2, end + 4]]
    act_bar_num_1 = len(rows)
    for i in range(1, N + 1):
        b = 8 * (i - 1)
        rows += [[b + 7, b + 5], [b + 8, b + 6]]
    return rows, act_bar_num_1


def _improved_act_bars(N):
    rows = []
    for i in range(1, N + 1):
        b = 10 * (i - 1)
        rows += [[b + 1, b + 3], [b + 2, b + 4]]
    end = 10 * N
    rows += [[end + 1, end + 3], [end + 2, end + 4]]
    act_bar_num_1 = len(rows)
    for i in range(1, N + 1):
        b = 10 * (i - 1)
        rows += [[b + 7, b + 5], [b + 5, b + 9], [b + 6, b + 10], [b + 8, b + 6]]
    return rows, act_bar_num_1


def build_scissor_model(variant="standard", analysis="deploy", N=None):
    improved = variant == "improved"
    load_case = analysis == "load"
    if N is None:
        N = 8

    H = 2.0
    L = 2.0
    barE = 2.0e11
    panel_E = 2.0e8
    panel_t = 0.01
    panel_v = 0.3

    if load_case:
        barA = 0.00415
        barA_brace = 0.0019
        Iy = 21.2e-6
        Ix = 7.16e-6
        kspr = barE * Iy / np.sqrt(H ** 2 + L ** 2)
        rot3_factor = 1.0
    else:
        barA = 0.0023
        barA_brace = barA
        Iy = 1.88e-6
        Ix = 1.88e-6
        kspr = barE * Iy / np.sqrt(H ** 2 + L ** 2)
        rot3_factor = 10.0 if improved else 100.0

    node = Elements_Nodes()
    node.coordinates_mat = _improved_coordinates(N, L) if improved else _standard_coordinates(N, L)

    cst = Vec_Elements_CST()
    cst.node_ijk_mat = as_int_array(_improved_cst(N) if improved else _standard_cst(N))
    cst.t_vec = panel_t * np.ones(cst.node_ijk_mat.shape[0])
    cst.E_vec = panel_E * np.ones(cst.node_ijk_mat.shape[0])
    cst.v_vec = panel_v * np.ones(cst.node_ijk_mat.shape[0])

    if improved:
        bar_rows, bar_areas = _improved_bars(N, load_case, barA, barA_brace)
        rot3_rows = _improved_rot3(N)
        rot4_rows = _improved_rot4(N)
        act_rows, act_bar_num_1 = _improved_act_bars(N)
        stride = 10
    else:
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
    plots.displayRange = np.asarray([-1, 2 * N + 1, -1, 3, -1, 3], dtype=float)
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
    supp[end, :] = [end, 0, 1, 1]
    supp[end + 1, :] = [end + 1, 0, 1, 1]
    supp[end + 2, :] = [end + 2, 0, 1, 0]
    supp[end + 3, :] = [end + 3, 0, 1, 0]
    return supp


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


def load_supports(N, stride):
    end = stride * N
    return np.asarray([
        [0, 1, 1, 1],
        [1, 1, 1, 1],
        [end, 0, 1, 1],
        [end + 1, 0, 1, 1],
    ], dtype=float)


def standard_deploy_delta(source_step):
    if source_step <= 200:
        return 0.001 * source_step
    if source_step <= 400:
        return 0.001 * 200 + 0.0004 * (source_step - 200)
    return 0.001 * 200 + 0.0004 * 200 + 0.0001 * (source_step - 400)


def improved_deploy_delta(source_step):
    return 0.001 * source_step


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


def save_geometry_diagnostics(model, out_dir, prefix):
    save_figure(model.plots.Plot_Shape_Node_Number(), os.path.join(out_dir, f"{prefix}_Node_Number.png"))
    save_figure(model.plots.Plot_Shape_CST_Number(), os.path.join(out_dir, f"{prefix}_CST_Number.png"))
    save_figure(model.plots.Plot_Shape_Bar_Number(), os.path.join(out_dir, f"{prefix}_Bar_Number.png"))
    save_figure(model.plots.Plot_Shape_RotSpr_3N_Number(), os.path.join(out_dir, f"{prefix}_RotSpr_3N_Number.png"))
    save_figure(model.plots.Plot_Shape_RotSpr_4N_Number(), os.path.join(out_dir, f"{prefix}_RotSpr_4N_Number.png"))
    save_figure(model.plots.Plot_Shape_ActBar_Number(), os.path.join(out_dir, f"{prefix}_ActBar_Number.png"))


def bridge_self_weight(model):
    rho_steel = 7850.0
    g = 9.81
    L_total = 0.0
    W_bar = 0.0
    coords = model.node.coordinates_mat
    for i, (n1, n2) in enumerate(model.bar.node_ij_mat):
        length = np.linalg.norm(coords[n1 - 1, :] - coords[n2 - 1, :])
        L_total += length
        W_bar += length * model.bar.A_vec[i] * rho_steel * g
    return L_total, W_bar
