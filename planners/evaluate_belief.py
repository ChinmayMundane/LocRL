"""Evaluate expected-belief Dubins RRT* through the canonical evaluator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "capql"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/localizationrl-matplotlib")

from evaluate import evaluate  # noqa: E402
from environment.map_dataset import rasterize_map  # noqa: E402
from planners.belief_controller import (  # noqa: E402
    BeliefLocalObservationController,
)


def file_sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class Adapter:
    def __init__(self, cfg, env, args) -> None:
        self.cfg = cfg
        self.env = env
        self.args = args

    def start_episode(self, spec, index) -> None:
        world = rasterize_map(spec, int(self.cfg.map_size))
        self.controller = BeliefLocalObservationController(
            world,
            self.env.unwrapped.goal_point,
            self.cfg,
            seed=self.args.seed * 100000 + index,
            iterations=self.args.iterations,
            replan_interval=self.args.replan_interval,
            navigation_weight=self.args.navigation_weight,
            localization_weight=1.0 - self.args.navigation_weight,
            uncertainty_scale=self.args.uncertainty_scale,
        )
        self.elapsed = 0.0
        self.latencies = []

    def eval(self, observation, preference):
        start = time.perf_counter()
        base = self.env.unwrapped
        self.controller.set_belief_covariance(base.x_cov.copy())
        size = int(self.cfg.view_size)
        obstacle_image = np.asarray(observation[: size * size]).reshape(size, size)
        action = self.controller.action(
            base.x_state.copy(),
            base.steps,
            int(base.collision_streak),
            obstacle_image=obstacle_image,
        )
        duration = time.perf_counter() - start
        self.elapsed += duration
        self.latencies.append(duration)
        return action

    def episode_metrics(self) -> dict:
        covariance = self.controller.planner.last_goal_covariance
        return {
            "planner_calls": self.controller.calls,
            "planner_failures": self.controller.failures,
            "decision_time_seconds": self.elapsed,
            "decision_latency_mean": float(np.mean(self.latencies)),
            "decision_latency_max": float(np.max(self.latencies)),
            "projected_routing_roots": self.controller.projected_roots,
            "rejected_local_controls": self.controller.local_rejections,
            "no_local_action_steps": self.controller.no_local_action,
            "predicted_goal_cost": float(self.controller.planner.last_goal_cost),
            "predicted_goal_covariance_trace": (
                None if covariance is None else float(np.trace(covariance[:2, :2]))
            ),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--num-maps", type=int, default=0)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--replan-interval", type=int, default=25)
    parser.add_argument("--navigation-weight", type=float, default=0.5)
    # Frozen after seed-11 pilots at {0.5, 1.0, 2.0, 4.0}. Scale 2.0 was the
    # efficiency--uncertainty knee selected in the paper validation pilots.
    parser.add_argument("--uncertainty-scale", type=float, default=2.0)
    parser.add_argument("--output-root", default=None)
    parser.add_argument(
        "--split", choices=["validation", "final_test"], default="validation"
    )
    parser.add_argument("--evaluation-seed", type=int, default=None)
    parser.add_argument(
        "--locked-test-config",
        default="configs/experiments/final_test.yaml",
    )
    parser.add_argument("--execute-final-test", action="store_true")
    args = parser.parse_args()
    if not 0.0 <= args.navigation_weight <= 1.0:
        parser.error("--navigation-weight must lie in [0, 1]")
    if (
        args.iterations < 1
        or args.replan_interval < 1
        or args.num_maps < 0
        or args.uncertainty_scale < 0.0
    ):
        parser.error("invalid planner budget, map count, or uncertainty scale")

    locked_path = ROOT / args.locked_test_config
    locked = OmegaConf.load(locked_path)
    if args.split == "final_test":
        if not args.execute_final_test:
            parser.error("--execute-final-test is required for the locked final split")
        manifest = ROOT / str(locked.dataset_dir) / f"{locked.split}.json"
        if file_sha256(manifest) != str(locked.sha256):
            raise RuntimeError("locked final-test checksum mismatch")
        default_eval_seed = int(locked.evaluation_seed)
    else:
        manifest = ROOT / "data/map_splits/validation.json"
        default_eval_seed = 46000
    evaluation_seed = (
        default_eval_seed if args.evaluation_seed is None else args.evaluation_seed
    )
    if args.split == "final_test" and evaluation_seed != int(locked.evaluation_seed):
        raise RuntimeError("final-test evaluation seed must match the locked protocol")

    if args.output_root is None:
        args.output_root = f"outputs/planners/{args.split}"
    cfg = OmegaConf.load(ROOT / "configs/environment.yaml")
    cfg.seed = args.seed
    cfg.eval_seed = evaluation_seed
    cfg.test_map_split = args.split
    if args.split == "final_test":
        cfg.map_generation_seed = int(locked.generation_seed)
    cfg.morl_test_weights = [args.navigation_weight, 1.0 - args.navigation_weight]
    cfg.full_eval_num_maps = args.num_maps
    cfg.full_eval_output_dir = str(
        ROOT / args.output_root / "belief_dubins_rrt_star" / f"seed_{args.seed}"
    )
    output = Path(cfg.full_eval_output_dir)
    output.mkdir(parents=True, exist_ok=True)
    protocol = {
        **vars(args),
        "method": "expected_belief_dubins_rrt_star",
        "dataset_sha256": file_sha256(manifest),
        "evaluation_seed": evaluation_seed,
        "information_access": (
            "known static global map; estimated pose and EKF covariance; "
            "identical egocentric obstacle channel for local execution"
        ),
        "belief_model": "nominal mean with EKF covariance prediction and expected landmark updates",
        "objective": "weighted Dubins length plus integrated log position-covariance trace",
        "process_covariance": np.diag(
            np.square(np.asarray(cfg.motion_noise_std))
        ).tolist(),
        "measurement_covariance": np.asarray(cfg.noise_cov).tolist(),
        "controller_version": BeliefLocalObservationController.VERSION,
        "training": "none",
    }
    if args.split == "final_test":
        protocol["locked_test_config_sha256"] = file_sha256(locked_path)
    protocol_path = output / "planner_protocol.json"
    if protocol_path.exists() and json.loads(protocol_path.read_text()) != protocol:
        raise RuntimeError("existing output has a different protocol; use a new root")
    tag = f"nav_{args.navigation_weight:.2f}_loc_{1.0-args.navigation_weight:.2f}".replace(
        ".", "p"
    )
    if (output / f"summary_{tag}.json").exists():
        print(f"Skipping completed evaluation: {output}")
        return
    protocol_path.write_text(json.dumps(protocol, indent=2) + "\n")
    evaluate(
        cfg,
        policy_factory=lambda current_cfg, env: Adapter(current_cfg, env, args),
        policy_label="belief_dubins_rrt_star",
        progress=True,
    )


if __name__ == "__main__":
    main()
