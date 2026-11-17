"""Run the expected-belief Dubins RRT* baseline over declared seeds."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[11, 22, 33, 44, 55])
    parser.add_argument("--num-maps", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--replan-interval", type=int, default=25)
    parser.add_argument("--navigation-weight", type=float, default=0.5)
    # Frozen from the declared seed-11 validation pilot grid.
    parser.add_argument("--uncertainty-scale", type=float, default=2.0)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--split", choices=["validation", "final_test"], default="validation"
    )
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument("--execute-final-test", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    output_root = args.output_root or ("outputs/planners/final_test" if args.split == "final_test" else "outputs/planners/validation")
    for seed in args.seeds:
        command = [
            sys.executable,
            str(ROOT / "planners/evaluate_belief.py"),
            "--seed",
            str(seed),
            "--num-maps",
            str(args.num_maps),
            "--iterations",
            str(args.iterations),
            "--replan-interval",
            str(args.replan_interval),
            "--navigation-weight",
            str(args.navigation_weight),
            "--uncertainty-scale",
            str(args.uncertainty_scale),
            "--output-root",
            output_root,
            "--split",
            args.split,
        ]
        if args.evaluation_seed is not None:
            command.extend(["--evaluation-seed", str(args.evaluation_seed)])
        if args.execute_final_test:
            command.append("--execute-final-test")
        if args.dry_run:
            print(" ".join(command))
        else:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
