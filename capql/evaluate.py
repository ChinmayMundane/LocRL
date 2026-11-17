"""Evaluate one MORL preference on every held-out test map without rendering."""

from __future__ import annotations

import csv
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf

from train import build_agent, make_morl_env, normalize_weights


root_dir = Path(__file__).resolve().parents[1]


def grid_shortest_path_to_goal_region(
    world: np.ndarray,
    start: Sequence[int],
    goal: Sequence[int],
    goal_threshold: float,
) -> int:
    """Return an obstacle-aware four-connected grid reference length."""
    start_cell = (int(start[0]), int(start[1]))
    goal_x, goal_y = float(goal[0]), float(goal[1])
    queue = deque([(start_cell, 0)])
    seen = {start_cell}
    while queue:
        (x, y), distance = queue.popleft()
        if math.hypot(x - goal_x, y - goal_y) <= goal_threshold:
            return distance
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (
                0 <= ny < world.shape[0]
                and 0 <= nx < world.shape[1]
                and (nx, ny) not in seen
                and world[ny, nx] != 1
            ):
                seen.add((nx, ny))
                queue.append(((nx, ny), distance + 1))
    raise RuntimeError(f"No obstacle-free path from {start_cell} to {tuple(goal)}")


def euclidean_lower_bound_to_goal_region(
    start: Sequence[int],
    goal: Sequence[int],
    goal_threshold: float,
) -> float:
    """Return a dynamics-independent lower bound on successful path length."""
    return max(
        0.0,
        math.hypot(float(start[0]) - float(goal[0]), float(start[1]) - float(goal[1]))
        - goal_threshold,
    )


def _mean(rows: Iterable[Dict], key: str, condition=None) -> float | None:
    values = []
    for row in rows:
        if condition is not None and not condition(row):
            continue
        # Optional policy diagnostics can legitimately be unavailable, for
        # example when a planner does not find a goal node on its final
        # replan.  They must not invalidate the complete evaluation summary.
        value = row.get(key)
        if value is None or value == "":
            continue
        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(numeric_value):
            values.append(numeric_value)
    return float(np.mean(values)) if values else None


def _wilson_interval(successes: int, total: int, z: float = 1.96) -> List[float]:
    """95% Wilson confidence interval for a binomial success rate."""
    if total == 0:
        return [0.0, 0.0]
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    half_width = (
        z
        * math.sqrt(proportion * (1.0 - proportion) / total + z * z / (4.0 * total * total))
        / denominator
    )
    return [center - half_width, center + half_width]


def summarize(rows: List[Dict]) -> Dict:
    successes = sum(int(row["success"]) for row in rows)
    total = len(rows)
    success_condition = lambda row: bool(row["success"])
    return {
        "num_maps": total,
        "successes": successes,
        "success_rate": successes / total if total else 0.0,
        "success_rate_wilson_95": _wilson_interval(successes, total),
        "mean_steps_all": _mean(rows, "steps"),
        "mean_steps_success": _mean(rows, "steps", success_condition),
        "mean_path_length_success": _mean(rows, "path_length", success_condition),
        "mean_path_efficiency_success": _mean(rows, "path_efficiency", success_condition),
        "mean_final_distance": _mean(rows, "final_distance"),
        "mean_final_covariance_trace": _mean(rows, "final_covariance_trace"),
        "mean_trajectory_covariance_trace": _mean(rows, "mean_covariance_trace"),
        "mean_max_covariance_trace": _mean(rows, "max_covariance_trace"),
        "mean_covariance_auc": _mean(rows, "covariance_auc"),
        "mean_position_rmse": _mean(rows, "position_rmse"),
        "mean_final_position_error": _mean(rows, "final_position_error"),
        "mean_max_position_error": _mean(rows, "max_position_error"),
        "estimated_goal_false_positive_rate": _mean(rows, "estimated_goal_false_positive"),
        "mean_collision_steps": _mean(rows, "collision_steps"),
        "mean_max_collision_streak": _mean(rows, "max_collision_streak"),
        "collision_trap_rate": _mean(rows, "collision_trap_detected"),
        "mean_landmark_visit_fraction": _mean(rows, "landmark_visit_fraction"),
        "mean_navigation_return_raw": _mean(rows, "navigation_return_raw"),
        "mean_localization_return_raw": _mean(rows, "localization_return_raw"),
        "mean_scalarized_return": _mean(rows, "scalarized_return"),
        "mean_episode_runtime_seconds": _mean(rows, "runtime_seconds"),
    }


@hydra.main(config_path=str(root_dir / "configs"), config_name="environment", version_base="1.1")
def main(cfg: DictConfig) -> None:
    evaluate(cfg)


def evaluate(cfg: DictConfig, policy_factory=None, policy_label=None, progress=False,
             stop_on_contact=False) -> None:
    """Run the complete fixed test split and save machine-readable results."""
    contact_terminal_rule = stop_on_contact or bool(getattr(cfg, "terminate_on_collision", False))
    if not bool(cfg.use_map_splits):
        raise ValueError("evaluate.py requires use_map_splits=true")

    env = make_morl_env(
        cfg,
        seed=int(cfg.eval_seed),
        map_split=str(cfg.test_map_split),
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

    preference = normalize_weights(cfg.morl_test_weights)
    checkpoint = str(getattr(cfg, "model_checkpoint", "final_model"))
    if checkpoint.endswith(".tar"):
        checkpoint = checkpoint[:-4]
    model_path = root_dir / "models" / cfg.model_file / f"{checkpoint}.tar"
    if policy_factory is None and not model_path.is_file():
        raise FileNotFoundError(
            f"No trained model found at {model_path}. "
            "Run capql/train.py first."
        )
    print(f"Loading model: {policy_label or model_path}")
    print(
        f"Evaluating {num_maps}/{available_maps} held-out maps with preference "
        f"navigation={preference[0]:.2f}, localization={preference[1]:.2f}"
    )
    if policy_factory is None:
        model = build_agent(cfg, env, training=False)
        model.load(str(model_path), load_replay_buffer=False)
    else:
        model = policy_factory(cfg, env)

    rows: List[Dict] = []
    evaluation_start = time.perf_counter()
    map_indices = range(num_maps)
    if progress:
        from tqdm import tqdm
        map_indices = tqdm(map_indices, desc=str(policy_label), unit='map')
    for map_index in map_indices:
        # Give every map its own repeatable stochastic realization. This makes
        # policy/noise comparisons paired: a different policy can take a
        # different number of steps without shifting the RNG stream for every
        # map that follows it.
        episode_seed = int(cfg.eval_seed) + map_index
        observation, reset_info = env.reset(
            seed=episode_seed,
            options={"map_index": map_index},
        )
        map_spec = base_env.map_specs[map_index]
        if hasattr(model, 'start_episode'):
            model.start_episode(map_spec, map_index)
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
        contact_failure = False
        steps = 0
        episode_start = time.perf_counter()

        while not (terminated or truncated):
            previous_x, previous_y = float(base_env.robot_x), float(base_env.robot_y)
            action = model.eval(observation, preference)
            observation, vector_reward, terminated, truncated, info = env.step(action)
            steps += 1
            normalized_return += vector_reward
            raw_return += info["vector_reward_raw"]
            path_length += math.hypot(
                float(base_env.robot_x) - previous_x,
                float(base_env.robot_y) - previous_y,
            )
            covariance_trace.append(float(info["covariance_trace"]))
            position_errors.append(float(info["position_error"]))
            collision_steps += int(info["collision"])
            localization_steps += int(base_env.localize)
            contact_failure = bool(info.get("collision_failure", False))
            if stop_on_contact and info["collision"]:
                contact_failure = True
                break

        success = int(info["is_success"] and not contact_failure)
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
                "timeout": int(truncated and not terminated and not contact_failure),
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
                "scalarized_return": float(np.dot(preference, normalized_return)),
                "runtime_seconds": time.perf_counter() - episode_start,
            }
        )

        if contact_terminal_rule:
            rows[-1]["contact_failure"] = int(contact_failure)
            rows[-1]["stop_reason"] = (
                "first_contact" if contact_failure else "goal" if success else "timeout"
            )
        if hasattr(model, 'episode_metrics'):
            rows[-1].update(model.episode_metrics())
        if not progress and ((map_index + 1) % 10 == 0 or map_index + 1 == num_maps):
            successes = sum(row["success"] for row in rows)
            print(
                f"Completed {map_index + 1:3d}/{num_maps}: "
                f"success={successes}/{len(rows)} ({100.0 * successes / len(rows):.1f}%)"
            )

    output_dir = Path(str(cfg.full_eval_output_dir))
    if not output_dir.is_absolute():
        output_dir = root_dir / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    weight_tag = f"nav_{preference[0]:.2f}_loc_{preference[1]:.2f}".replace(".", "p")
    csv_path = output_dir / f"test_maps_{weight_tag}.csv"
    summary_path = output_dir / f"summary_{weight_tag}.json"

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "model_path": str(model_path) if policy_factory is None else None,
        "policy_label": policy_label,
        "test_split": str(cfg.test_map_split),
        "preference": {
            "navigation": float(preference[0]),
            "localization": float(preference[1]),
        },
        "dataset_generation_seed": int(cfg.map_generation_seed),
        "evaluation_seed": int(cfg.eval_seed),
        "total_runtime_seconds": time.perf_counter() - evaluation_start,
        "overall": summarize(rows),
        "by_difficulty": {
            difficulty: summarize([row for row in rows if row["difficulty"] == difficulty])
            for difficulty in ("easy", "medium", "hard")
        },
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if contact_terminal_rule:
        summary["evaluation_rule"] = "immediate first-contact failure; no subsequent action"
        summary["contact_failures"] = sum(row["contact_failure"] for row in rows)
        summary["timeouts"] = sum(row["timeout"] for row in rows)
        summary["reward_note"] = (
            "Contact receives the existing collision and collision-abort penalties, "
            "with no goal or timeout bonus, and is terminal in replay."
            if bool(getattr(cfg, "terminate_on_collision", False)) else
            "Rewards retain the original formula through contact; no new terminal "
            "penalty is introduced in this frozen-policy diagnostic."
        )
    if hasattr(model, 'episode_metrics'):
        summary['policy_diagnostics_mean'] = {
            key: _mean(rows, key) for key in model.episode_metrics()
        }
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")

    overall = summary["overall"]
    interval = overall["success_rate_wilson_95"]
    print("\nEvaluation complete (no GIFs rendered).")
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
