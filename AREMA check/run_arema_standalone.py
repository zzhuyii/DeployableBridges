import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.show = lambda *args, **kwargs: None


AREMA_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.abspath(os.path.join(AREMA_DIR, ".."))
if AREMA_DIR not in sys.path:
    sys.path.insert(0, AREMA_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(1, PROJECT_DIR)

from Kirigami_Truss_Load_To_Fail_AREMA import kirigami_fail
from Kirigami_Truss_Strength_During_Deploy_AREMA import kirigami_deploy
from Origami_Bridge_Load_To_Fail_AREMA import origami_fail
from Origami_Bridge_Strength_During_Deploy_AREMA import origami_deploy
from Rolling_Bridge_Load_To_Fail_AREMA import rolling_fail
from Rolling_Bridge_Strength_During_Deploy_AREMA import rolling_deploy
from Scissor_Bridge_2_Load_To_Fail_AREMA import improvedScissor_fail
from Scissor_Bridge_2_Strength_During_Deploy_AREMA import improvedScissor_deploy
from Scissor_Bridge_Load_To_Fail_AREMA import scissor_fail
from Scissor_Bridge_Strength_During_Deploy_AREMA import scissor_deploy


DEPLOY_FUNCS = {
    "kirigami": lambda L, N, dep_rate: kirigami_deploy(L, N, dep_rate),
    "origami": lambda L, N, dep_rate: origami_deploy(L, N, dep_rate),
    "rolling": lambda L, N, dep_rate: rolling_deploy(N, dep_rate),
    "scissor": lambda L, N, dep_rate: scissor_deploy(N, dep_rate),
    "improved_scissor": lambda L, N, dep_rate: improvedScissor_deploy(N, dep_rate),
}

FAIL_FUNCS = {
    "kirigami": lambda L, N: kirigami_fail(L, N),
    "origami": lambda L, N: origami_fail(L, N),
    "rolling": lambda L, N: rolling_fail(N),
    "scissor": lambda L, N: scissor_fail(N),
    "improved_scissor": lambda L, N: improvedScissor_fail(N),
}


def save_figures(figures, output_dir, prefix):
    os.makedirs(output_dir, exist_ok=True)
    paths = []
    for idx, fig in enumerate(figures, start=1):
        path = os.path.join(output_dir, f"{prefix}_figure_{idx}.png")
        fig.savefig(path, dpi=220, bbox_inches="tight")
        paths.append(path)
    plt.close("all")
    return paths


def run_deploy(bridge, sections, length, deployment_ratio, output_dir):
    fig1, fig2, tip = DEPLOY_FUNCS[bridge](length, sections, deployment_ratio)
    paths = save_figures(
        [fig1, fig2],
        output_dir,
        f"{bridge}_deploy_N{sections}_R{deployment_ratio:g}",
    )
    print(f"[AREMA deploy] bridge={bridge}, sections={sections}, deployment_ratio={deployment_ratio}")
    print(f"Tip deflection (m): {tip}")
    for path in paths:
        print(f"Saved figure: {path}")


def run_fail(bridge, sections, length, output_dir):
    fig1, fig2, max_load, self_weight = FAIL_FUNCS[bridge](length, sections)
    paths = save_figures(
        [fig1, fig2],
        output_dir,
        f"{bridge}_fail_N{sections}",
    )
    print(f"[AREMA load-to-failure] bridge={bridge}, sections={sections}")
    print(f"Maximum load (kN): {max_load / 1000.0}")
    print(f"Load / self-weight: {max_load / self_weight}")
    for path in paths:
        print(f"Saved figure: {path}")


def main():
    parser = argparse.ArgumentParser(description="Run AREMA deployable bridge checks without Streamlit.")
    parser.add_argument(
        "--bridge",
        choices=sorted(DEPLOY_FUNCS),
        default="kirigami",
        help="Bridge type to analyze.",
    )
    parser.add_argument(
        "--analysis",
        choices=["deploy", "fail", "both"],
        default="both",
        help="Analysis to run.",
    )
    parser.add_argument("--sections", type=int, default=2, help="Number of bridge sections.")
    parser.add_argument("--length", type=float, default=2.0, help="Section length for kirigami/origami.")
    parser.add_argument("--deployment-ratio", type=float, default=0.5, help="Deployment ratio for deploy analysis.")
    parser.add_argument(
        "--output-dir",
        default=os.path.join(AREMA_DIR, "results"),
        help="Directory where PNG result figures will be saved.",
    )
    args = parser.parse_args()

    if args.analysis in ("deploy", "both"):
        run_deploy(args.bridge, args.sections, args.length, args.deployment_ratio, args.output_dir)
    if args.analysis in ("fail", "both"):
        run_fail(args.bridge, args.sections, args.length, args.output_dir)


if __name__ == "__main__":
    main()
