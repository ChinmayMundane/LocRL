"""Evaluate a fixed scalar-SAC policy on every map in a held-out split."""

from __future__ import annotations

import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, List

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from stable_baselines3 import SAC

from common import ROOT, scalar_model_path
from capql.evaluate import (  # noqa: E402
    euclidean_lower_bound_to_goal_region,
    grid_shortest_path_to_goal_region,
    summarize,
)
from capql.train import make_morl_env, normalize_weights  # noqa: E402


@hydra.main(
    config_path=str(ROOT / "configs"),
    config_name="environment",
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:
    """Run paired, deterministic-policy evaluation without rendering."""
    if not bool(cfg.use_map_splits):
        raise ValueError("evaluate.py requires use_map_splits=true")

    env = make_morl_env(
        cfg,
        seed=int(cfg.eval_seed),
        map_split=str(cfg.test_map_split),
        training=False,
    )
    base_env = env.unwrapped
    available_maps = len(base_env.map_specs)
    requested_maps = int(cfg.full_eval_num_maps)
    num_maps = available_maps if requested_maps <= 0 else requested_maps
    if num_maps > available_maps:
        raise ValueError(
            f"full_eval_num_maps={num_maps} exceeds the {available_maps} maps "
            f"in the {cfg.test_map_split!r} split"
        )

    weights = normalize_weights(cfg.morl_test_weights)
    model_path = scalar_model_path(cfg)
    if not model_path.is_file():
        raise FileNotFoundError(
            f"No scalar SAC model found at {model_path}. "
            "Run fixed_sac/train.py first."
        )
    print(f"Loading scalar SAC model: {model_path}")
    print(
        f"Evaluating {num_maps}/{available_maps} held-out maps with fixed weights "
        f"navigation={weights[0]:.2f}, localization={weights[1]:.2f}"
    )
    model = SAC.load(str(model_path), device="auto")

    rows: List[Dict] = []
    evaluation_start = time.perf_counter()
    for map_index in range(num_maps):
        episode_seed = int(cfg.eval_seed) + map_index
        observation, reset_info = env.reset(
            seed=episode_seed,
            options={"map_index": map_index},
        )
        map_spec = base_env.map_specs[map_index]
        start = (int(base_env.start_x), int(base_env.start_y))
        goal = (int(base_env.goal_x), int(base_env.goal_y))
        reference_path_length = grid_shortest_path_to_goal_region(
            base_env.world_map,
            start,
            goal,
            float(cfg.goal_threshold),
        )
        euclidean_lower_bound = euclidean_lower_bound_to_goal_region(
            start,
            goal,
            float(cfg.goal_threshold),
        )

        normalized_return = np.zeros(2, dtype=np.float64)
        raw_return = np.zeros(2, dtype=np.float64)
        covariance_trace = [float(np.trace(base_env.x_cov[:2, :2]))]
        position_errors = [0.0]
        collision_steps = 0
        localization_steps = 0
        path_length = 0.0
        terminated = False
        truncated = False
        steps = 0
        episode_start = time.perf_counter()

        while not (terminated or truncated):
            previous_x, previous_y = float(base_env.robot_x), float(base_env.robot_y)
            action, _ = model.predict(observation, deterministic=True)
            observation, vector_reward, terminated, truncated, info = env.step(action)
            steps += 1
            normalized_return += np.asarray(vector_reward, dtype=np.float64)
            raw_return += np.asarray(info["vector_reward_raw"], dtype=np.float64)
            path_length += math.hypot(
                float(base_env.robot_x) - previous_x,
                float(base_env.robot_y) - previous_y,
            )
            covariance_trace.append(float(info["covariance_trace"]))
            position_errors.append(float(info["position_error"]))
            collision_steps += int(info["collision"])
            localization_steps += int(base_env.localize)

        success = int(info["is_success"])
        visited_landmarks = int(np.sum(base_env.land_visit))
        landmark_count = len(base_env.land_list)
        path_efficiency = (
            euclidean_lower_bound / path_length
            if success and path_length > 0.0
            else float("nan")
        )
        rows.append(
            {
                "map_index": map_index,
                "map_id": reset_info["map_id"],
                "difficulty": reset_info["map_difficulty"],
                "map_seed": int(map_spec["seed"]),
                "episode_seed": episode_seed,
                "start_x": start[0],
                "start_y": start[1],
                "goal_x": goal[0],
                "goal_y": goal[1],
                "obstacle_count": len(base_env.obs_list),
                "landmark_count": landmark_count,
                "success": success,
                "timeout": int(truncated and not terminated),
                "steps": steps,
                "reference_grid_path_length": reference_path_length,
                "reference_euclidean_lower_bound": euclidean_lower_bound,
                "path_length": path_length,
                "path_efficiency": path_efficiency,
                "final_distance": float(info["distance_to_goal"]),
                "collision_steps": collision_steps,
                "max_collision_streak": int(info["max_collision_streak"]),
                "collision_trap_detected": int(info["collision_trap_detected"]),
                "localization_steps": localization_steps,
                "landmarks_visited": visited_landmarks,
                "landmark_visit_fraction": visited_landmarks / landmark_count,
                "initial_covariance_trace": covariance_trace[0],
                "final_covariance_trace": covariance_trace[-1],
                "mean_covariance_trace": float(np.mean(covariance_trace)),
                "max_covariance_trace": float(np.max(covariance_trace)),
                "covariance_auc": float(np.sum(covariance_trace)),
                "position_rmse": float(np.sqrt(np.mean(np.square(position_errors)))),
                "final_position_error": position_errors[-1],
                "max_position_error": float(np.max(position_errors)),
                "estimated_goal_false_positive": int(
                    info["estimated_goal_reached"] and not success
                ),
                "navigation_return_raw": float(raw_return[0]),
                "localization_return_raw": float(raw_return[1]),
                "navigation_return_normalized": float(normalized_return[0]),
                "localization_return_normalized": float(normalized_return[1]),
                "scalarized_return": float(np.dot(weights, normalized_return)),
                "runtime_seconds": time.perf_counter() - episode_start,
            }
        )
        if bool(getattr(cfg, "terminate_on_collision", False)):
            failed = bool(info["collision_failure"])
            rows[-1]["contact_failure"] = int(failed)
            rows[-1]["stop_reason"] = "first_contact" if failed else "goal" if success else "timeout"
        if (map_index + 1) % 10 == 0 or map_index + 1 == num_maps:
            successes = sum(int(row["success"]) for row in rows)
            print(
                f"Completed {map_index + 1:3d}/{num_maps}: "
                f"success={successes}/{len(rows)} ({100.0 * successes / len(rows):.1f}%)"
            )

    output_directory = Path(str(cfg.full_eval_output_dir))
    if not output_directory.is_absolute():
        output_directory = ROOT / output_directory
    output_directory.mkdir(parents=True, exist_ok=True)
    tag = f"nav_{weights[0]:.2f}_loc_{weights[1]:.2f}".replace(".", "p")
    csv_path = output_directory / f"test_maps_{tag}.csv"
    summary_path = output_directory / f"summary_{tag}.json"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "method": "fixed_scalar_sac",
        "model_path": str(model_path),
        "test_split": str(cfg.test_map_split),
        "preference": {
            "navigation": float(weights[0]),
            "localization": float(weights[1]),
        },
        "dataset_generation_seed": int(cfg.map_generation_seed),
        "evaluation_seed": int(cfg.eval_seed),
        "total_runtime_seconds": time.perf_counter() - evaluation_start,
        "overall": summarize(rows),
        "by_difficulty": {
            difficulty: summarize(
                [row for row in rows if row["difficulty"] == difficulty]
            )
            for difficulty in ("easy", "medium", "hard")
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if bool(getattr(cfg, "terminate_on_collision", False)):
        summary["evaluation_rule"] = "immediate first-contact failure; no subsequent action"
        summary["contact_failures"] = sum(row["contact_failure"] for row in rows)
        summary["timeouts"] = sum(row["timeout"] for row in rows)
    summary_path.write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    overall = summary["overall"]
    interval = overall["success_rate_wilson_95"]
    print("\nScalar SAC evaluation complete (no GIFs rendered).")
    print(
        f"Success: {overall['successes']}/{overall['num_maps']} "
        f"({100.0 * overall['success_rate']:.1f}%, "
        f"95% CI {100.0 * interval[0]:.1f}-{100.0 * interval[1]:.1f}%)"
    )
    print(f"Per-map results: {csv_path}")
    print(f"Aggregate summary: {summary_path}")
    env.close()
if __name__ == "__main__":
    main()
