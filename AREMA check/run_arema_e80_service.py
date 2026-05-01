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

from Kirigami_Truss_E80_Service_Check_AREMA import kirigami_e80_service_arema
from Origami_Bridge_E80_Service_Check_AREMA import origami_e80_service_arema
from Rolling_Bridge_E80_Service_Check_AREMA import rolling_e80_service_arema
from Scissor_Bridge_E80_Service_Check_AREMA import scissor_e80_service_arema
from Scissor_Bridge_2_E80_Service_Check_AREMA import improvedScissor_e80_service_arema


E80_FUNCS = {
    "kirigami": lambda L, N, include_impact, train_step_m: kirigami_e80_service_arema(
        L, N, include_impact=include_impact, train_step_m=train_step_m
    ),
    "origami": lambda L, N, include_impact, train_step_m: origami_e80_service_arema(
        L, N, include_impact=include_impact, train_step_m=train_step_m
    ),
    "rolling": lambda L, N, include_impact, train_step_m: rolling_e80_service_arema(
        N, include_impact=include_impact, train_step_m=train_step_m
    ),
    "scissor": lambda L, N, include_impact, train_step_m: scissor_e80_service_arema(
        N, include_impact=include_impact, train_step_m=train_step_m
    ),
    "improved_scissor": lambda L, N, include_impact, train_step_m: improvedScissor_e80_service_arema(
        N, include_impact=include_impact, train_step_m=train_step_m
    ),
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


def main():
    parser = argparse.ArgumentParser(description="Run AREMA Cooper E80 railway service-load checks.")
    parser.add_argument("--bridge", choices=sorted(E80_FUNCS), default="kirigami")
    parser.add_argument("--sections", type=int, default=2, help="Number of bridge sections.")
    parser.add_argument("--length", type=float, default=2.0, help="Section length for kirigami/origami.")
    parser.add_argument(
        "--train-step-m",
        type=float,
        default=None,
        help="Moving train scan step in meters. Default uses max(span/40, 0.25 m).",
    )
    parser.add_argument(
        "--no-impact",
        action="store_true",
        help="Disable AREMA vertical impact on Cooper E80 live load.",
    )
    parser.add_argument(
        "--output-dir",
        default=os.path.join(AREMA_DIR, "results", "e80_service"),
        help="Directory where PNG result figures will be saved.",
    )
    args = parser.parse_args()

    fig1, fig2, result = E80_FUNCS[args.bridge](
        args.length, args.sections, not args.no_impact, args.train_step_m
    )
    paths = save_figures([fig1, fig2], args.output_dir, f"{args.bridge}_e80_service_N{args.sections}")

    print(f"[AREMA E80 service] bridge={args.bridge}, sections={args.sections}")
    print(f"Pass: {result['safe']}")
    print(f"Maximum DCR: {result['max_dcr']:.4f}")
    print(f"Controlling train direction: {result['direction']}")
    print(f"Controlling train start x (m): {result['train_start_m']:.3f}")
    print(f"Impact percent applied to E80 live load: {result['impact_percent']:.2f}%")
    print(f"Train positions checked: {result['checked_positions']}")
    if result.get("extra"):
        print(f"Extra check: {result['extra']}")
    for path in paths:
        print(f"Saved figure: {path}")


if __name__ == "__main__":
    main()
