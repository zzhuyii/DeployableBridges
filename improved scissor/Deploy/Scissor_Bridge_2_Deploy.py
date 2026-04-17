import os
import time

import numpy as np

from Solver_NR_TrussAction import Solver_NR_TrussAction
from scissor_common import (
    actuator_target_lengths,
    build_scissor_model,
    improved_deploy_delta,
    improved_deploy_supports,
    save_figure,
    save_geometry_diagnostics,
    source_step_for_frame,
)


def main():
    print("RUNNING FILE:", __file__)
    t0 = time.time()
    out_dir = os.path.dirname(os.path.abspath(__file__))

    N = 8
    model = build_scissor_model(variant="improved", analysis="deploy", N=N)
    if model.settings["N"] != N:
        raise RuntimeError(f"Improved scissor deploy must use N={N}; got N={model.settings['N']}.")
    model.assembly.Initialize_Assembly()

    save_geometry_diagnostics(model, out_dir, "Scissor_Bridge_2_Deploy")

    ta = Solver_NR_TrussAction()
    ta.assembly = model.assembly
    ta.supp = improved_deploy_supports(model.node.coordinates_mat.shape[0])
    ta.verbose = False

    source_steps = 700
    analysis_steps = int(os.environ.get("SCISSOR_DEPLOY_STEPS", "120"))
    gif_stride = max(1, analysis_steps // int(os.environ.get("SCISSOR_GIF_FRAMES", "90")))

    base_L0 = model.act_bar.L0_vec.copy()
    act_bar_count = model.act_bar.node_ij_mat.shape[0]
    act_bar_num_1 = model.settings["act_bar_num_1"]
    L = model.settings["L"]
    Uhis = np.zeros((analysis_steps, model.node.coordinates_mat.shape[0], 3), dtype=float)

    for step in range(analysis_steps):
        source_step = source_step_for_frame(step, analysis_steps, source_steps)
        dL = improved_deploy_delta(source_step)
        ta.increStep = 1
        ta.iterMax = 40
        ta.tol = 5e-3
        ta.targetL0 = actuator_target_lengths(base_L0, dL, L, act_bar_num_1, act_bar_count)
        Uhis[step, :, :] = ta.Solve()[-1, :, :]
        if (step + 1) % max(1, analysis_steps // 10) == 0:
            print(f"Deployment progress: {step + 1}/{analysis_steps}")

    np.save(os.path.join(out_dir, "ScissorUhis2.npy"), Uhis)

    model.plots.fileName = os.path.join(out_dir, "Scissor_Bridge_2_Deploy.gif")
    model.plots.Plot_Deformed_His(Uhis[::gif_stride, :, :])

    U_end = Uhis[-1, :, :]
    save_figure(model.plots.Plot_Deformed_Shape(U_end), os.path.join(out_dir, "Scissor_Bridge_2_Final.png"))

    print("Uhis shape:", Uhis.shape)
    print("Execution Time:", time.time() - t0)


if __name__ == "__main__":
    main()
