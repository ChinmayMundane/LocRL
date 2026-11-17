"""Summarize the locked 200-map final study, including difficulty strata."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
SEEDS = (11, 22, 33, 44, 55)
DIFFICULTIES = ("easy", "medium", "hard")
PREFERENCES = (1.0, 0.8, 0.6, 0.5, 0.4, 0.2, 0.0)
TAG = "nav_0p50_loc_0p50"
EXPECTED_COUNTS = {"easy": 67, "medium": 67, "hard": 66}
LOCKED_CONFIG = OmegaConf.load(ROOT / "configs/experiments/final_test.yaml")
RESULT_LABEL = str(getattr(LOCKED_CONFIG, "result_label", "final_test"))
EVALUATION_SEED = int(LOCKED_CONFIG.evaluation_seed)

METHODS = {
    "CAPQL": ROOT / "results/capql",
    "PC-SAC": ROOT / "results/pc_sac",
    "Belief RRT* + local": ROOT / "results/planners/belief_rrt_star",
    "Scalar SAC": ROOT / "results/fixed_sac",
    "Dubins RRT* + local": ROOT / "results/planners/dubins_rrt_star",
    "Landmark RRT* + local": ROOT / "results/planners/landmark_rrt_star",
}
ABLATIONS = {
    "Full CAPQL": ROOT / "results/capql",
    "No landmark reward": ROOT / "results/ablation/no_landmark_reward",
}

METRICS = {
    "success_rate": "success_rate",
    "steps": "mean_steps_all",
    "path_efficiency": "mean_path_efficiency_success",
    "covariance_auc": "mean_covariance_auc",
    "mean_covariance": "mean_trajectory_covariance_trace",
    "max_covariance": "mean_max_covariance_trace",
    "position_rmse": "mean_position_rmse",
    "final_error": "mean_final_position_error",
    "landmark_fraction": "mean_landmark_visit_fraction",
    "collision_steps": "mean_collision_steps",
    "scalarized_return": "mean_scalarized_return",
    "runtime_seconds": "mean_episode_runtime_seconds",
}


def preference_tag(navigation: float) -> str:
    return f"nav_{navigation:.2f}_loc_{1-navigation:.2f}".replace(".", "p")


def result_path(root: Path, seed: int, tag: str, prefix: str) -> Path:
    return root / f"seed_{seed}" / f"{prefix}_{tag}.json"


def load_result(root: Path, seed: int, tag: str = TAG) -> dict:
    path = result_path(root, seed, tag, "summary")
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("test_split") != "final_test":
        raise ValueError(f"Non-final result supplied: {path}")
    if data.get("evaluation_seed") != EVALUATION_SEED:
        raise ValueError(f"Evaluation-seed mismatch: {path}")
    if data["overall"]["num_maps"] != 200:
        raise ValueError(f"Expected 200 maps: {path}")
    for difficulty, count in EXPECTED_COUNTS.items():
        if data["by_difficulty"][difficulty]["num_maps"] != count:
            raise ValueError(f"Difficulty-count mismatch in {path}: {difficulty}")
    return data


def load_episodes(root: Path, seed: int, tag: str = TAG) -> dict[str, dict]:
    path = root / f"seed_{seed}" / f"test_maps_{tag}.csv"
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 200:
        raise ValueError(f"Expected 200 maps in {path}, found {len(rows)}")
    if [int(row["episode_seed"]) for row in rows] != list(
        range(EVALUATION_SEED, EVALUATION_SEED + 200)
    ):
        raise ValueError(f"Evaluation-seed mismatch: {path}")
    return {row["map_id"]: row for row in rows}


def aggregate(values: list[float]) -> tuple[float, float]:
    return mean(values), stdev(values)


def metric_columns(row: dict, summaries: list[dict]) -> None:
    for short, key in METRICS.items():
        values = [summary.get(key) for summary in summaries]
        if any(value is None for value in values):
            row[f"{short}_mean"] = None
            row[f"{short}_sd"] = None
        else:
            row[f"{short}_mean"], row[f"{short}_sd"] = aggregate(
                [float(value) for value in values]
            )


def method_table(difficulty: str | None = None) -> list[dict]:
    rows = []
    timing = {item['method']: item for item in json.loads((ROOT / 'results/computation_time.json').read_text())}
    for method, root in METHODS.items():
        results = [load_result(root, seed) for seed in SEEDS]
        summaries = [
            result["overall"] if difficulty is None else result["by_difficulty"][difficulty]
            for result in results
        ]
        row = {
            "method": method,
            "difficulty": difficulty or "all",
            "distinct_maps": 200 if difficulty is None else EXPECTED_COUNTS[difficulty],
            "successes_across_seed_runs": sum(item["successes"] for item in summaries),
            "episodes_across_seed_runs": sum(item["num_maps"] for item in summaries),
        }
        metric_columns(row, summaries)
        timing_name = "Fixed SAC" if method == "Scalar SAC" else method.replace(" + local", "")
        measured = [float(item['mean_ms']) for item in timing[timing_name]['per_seed']]
        row['computation_time_ms_mean'], row['computation_time_ms_sd'] = aggregate(measured)
        rows.append(row)
    return rows


def preference_table(difficulty: str | None = None) -> list[dict]:
    rows = []
    for navigation in PREFERENCES:
        results = [
            load_result(METHODS["CAPQL"], seed, preference_tag(navigation))
            for seed in SEEDS
        ]
        summaries = [
            result["overall"] if difficulty is None else result["by_difficulty"][difficulty]
            for result in results
        ]
        row = {
            "navigation_weight": navigation,
            "localization_weight": 1.0 - navigation,
            "difficulty": difficulty or "all",
        }
        metric_columns(row, summaries)
        rows.append(row)
    return rows


def ablation_table(difficulty: str | None = None) -> list[dict]:
    rows = []
    for variant, root in ABLATIONS.items():
        results = [load_result(root, seed) for seed in SEEDS]
        summaries = [
            result["overall"] if difficulty is None else result["by_difficulty"][difficulty]
            for result in results
        ]
        row = {"variant": variant, "difficulty": difficulty or "all"}
        metric_columns(row, summaries)
        rows.append(row)
    return rows


def common_success_table() -> list[dict]:
    metrics = ("steps", "path_efficiency", "covariance_auc", "mean_covariance_trace",
               "max_covariance_trace", "position_rmse", "final_position_error")
    rows = []
    for method, root in METHODS.items():
        if method == "CAPQL":
            continue
        per_seed = []
        for seed in SEEDS:
            capql = load_episodes(METHODS["CAPQL"], seed)
            baseline = load_episodes(root, seed)
            if capql.keys() != baseline.keys():
                raise ValueError(f"Map IDs differ for {method}, seed {seed}")
            common = [key for key in capql if capql[key]["success"] == "1" and baseline[key]["success"] == "1"]
            item = {"common": len(common)}
            for metric in metrics:
                item[f"capql_{metric}"] = mean(float(capql[key][metric]) for key in common)
                item[f"baseline_{metric}"] = mean(float(baseline[key][metric]) for key in common)
            per_seed.append(item)
        row = {"baseline": method, "common_successes_across_seed_runs": sum(x["common"] for x in per_seed)}
        for metric in metrics:
            row[f"capql_{metric}"] = mean(x[f"capql_{metric}"] for x in per_seed)
            row[f"baseline_{metric}"] = mean(x[f"baseline_{metric}"] for x in per_seed)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def value(row: dict, key: str, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    return f"{row[key + '_mean'] * scale:.2f} +/- {row[key + '_sd'] * scale:.2f}"


def make_report(main: list[dict], difficulty: list[dict], preferences: list[dict], ablations: list[dict]) -> str:
    lines = [
        "# Locked final-test study", "",
        "These are the one-time results on 200 unseen maps: 67 easy, 67 medium, and 66 hard.",
        "Each entry is the mean +/- sample SD across five seeds. The seed is the sampling unit.",
        "Path efficiency uses successful episodes; all localization metrics use all episodes.", "",
        "## Main comparison at preference [0.5, 0.5]", "",
        "| Method | Success (%) | Efficiency | Covariance AUC | Position RMSE | Final error |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in main:
        lines.append(f"| {row['method']} | {value(row, 'success_rate', True)} | {value(row, 'path_efficiency')} | {value(row, 'covariance_auc')} | {value(row, 'position_rmse')} | {value(row, 'final_error')} |")
    lines += ["", "## Results by difficulty", "",
              "| Difficulty | Method | Success (%) | Efficiency | Covariance AUC | Position RMSE |",
              "|---|---|---:|---:|---:|---:|"]
    for row in difficulty:
        lines.append(f"| {row['difficulty']} | {row['method']} | {value(row, 'success_rate', True)} | {value(row, 'path_efficiency')} | {value(row, 'covariance_auc')} | {value(row, 'position_rmse')} |")
    lines += ["", "## CAPQL preference response", "",
              "| Navigation/localization | Success (%) | Efficiency | Covariance AUC | RMSE | Landmark fraction |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in preferences:
        lines.append(f"| {row['navigation_weight']:.1f}/{row['localization_weight']:.1f} | {value(row, 'success_rate', True)} | {value(row, 'path_efficiency')} | {value(row, 'covariance_auc')} | {value(row, 'position_rmse')} | {value(row, 'landmark_fraction')} |")
    lines += ["", "## Reward ablations", "",
              "| Variant | Success (%) | Efficiency | Covariance AUC | RMSE | Landmark fraction |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in ablations:
        lines.append(f"| {row['variant']} | {value(row, 'success_rate', True)} | {value(row, 'path_efficiency')} | {value(row, 'covariance_auc')} | {value(row, 'position_rmse')} | {value(row, 'landmark_fraction')} |")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/summary")
    parser.add_argument("--save", action="store_true", help="Also save CSVs and the full report.")
    parser.add_argument("--table", choices=[1, 3, 4], type=int, default=1, help="Paper table to print (default: 1).")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if not args.save and args.table == 3:
        print("Table 3 — three preferences; mean +/- SD over five seeds\n")
        print("Nav./loc. weights | Method | Success (%) | Efficiency | Covariance AUC | RMSE")
        for weight in [0.8, 0.5, 0.2]:
            for name in ["CAPQL", "PC-SAC", "Scalar SAC"]:
                summaries = [load_result(METHODS[name], seed, preference_tag(weight))["overall"] for seed in SEEDS]
                row = {}
                metric_columns(row, summaries)
                cells = [value(row, "success_rate", True)]
                cells += [f"{row[key + '_mean']:.3f} +/- {row[key + '_sd']:.3f}" for key in
                          ["path_efficiency", "covariance_auc", "position_rmse"]]
                label = "Fixed SAC" if name == "Scalar SAC" else name
                print(f"({weight:.1f}, {1-weight:.1f}) | {label} | " + " | ".join(cells))
        return
    if not args.save and args.table == 4:
        print("Table 4 — balanced landmark-reward ablation; mean +/- SD over five seeds\n")
        print("Variant | Success (%) | Efficiency | Mean covariance | RMSE | Landmark fraction")
        for row in ablation_table():
            cells = [value(row, "success_rate", True)]
            cells += [f"{row[key + '_mean']:.3f} +/- {row[key + '_sd']:.3f}" for key in
                      ["path_efficiency", "mean_covariance", "position_rmse", "landmark_fraction"]]
            print(row['variant'] + " | " + " | ".join(cells))
        return
    main_rows = method_table()
    if not args.save:
        print("Table 1 — balanced preference; mean +/- SD over five seeds\n")
        print("Method | Success (%) | Efficiency | Cov. AUC | Mean cov. | RMSE | Final error | Landmark fraction | Comp. time (ms/step)")
        for row in main_rows:
            cells = [value(row, "success_rate", True)]
            cells += [f"{row[key + '_mean']:.3f} +/- {row[key + '_sd']:.3f}" for key in
                      ["path_efficiency", "covariance_auc", "mean_covariance", "position_rmse", "final_error", "landmark_fraction", "computation_time_ms"]]
            print(row['method'] + " | " + " | ".join(cells))
        print("\nComputation times are recorded paper measurements, not a new timing run.")
        return
    output.mkdir(parents=True, exist_ok=True)
    difficulty_rows = [row for level in DIFFICULTIES for row in method_table(level)]
    preference_rows = preference_table()
    preference_difficulty_rows = [row for level in DIFFICULTIES for row in preference_table(level)]
    ablation_rows = ablation_table()
    ablation_difficulty_rows = [row for level in DIFFICULTIES for row in ablation_table(level)]
    common_rows = common_success_table()
    write_csv(output / "main_methods.csv", main_rows)
    write_csv(output / "main_methods_by_difficulty.csv", difficulty_rows)
    write_csv(output / "capql_preferences.csv", preference_rows)
    write_csv(output / "capql_preferences_by_difficulty.csv", preference_difficulty_rows)
    write_csv(output / "reward_ablations.csv", ablation_rows)
    write_csv(output / "reward_ablations_by_difficulty.csv", ablation_difficulty_rows)
    write_csv(output / "common_success.csv", common_rows)
    report = make_report(main_rows, difficulty_rows, preference_rows, ablation_rows)
    (output / "REPORT.md").write_text(report, encoding="utf-8")
    print(f"Final report: {output / 'REPORT.md'}")


if __name__ == "__main__":
    main()
