"""Post-hoc trajectory metrics for the original recovery-enabled evaluations.

Compute method-specific zero-contact cohorts and matched cohorts on identical
map/seed episodes. No replay, retraining or manuscript changes are performed.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
from itertools import combinations
import json
import math
from pathlib import Path
import statistics

SEEDS = (11, 22, 33, 44, 55)
SOURCES = {
    "Fixed SAC": "fixed_sac",
    "CAPQL": "capql",
    "PC-SAC": "pc_sac",
    "Dubins RRT*": "planners/dubins_rrt_star",
    "Landmark RRT*": "planners/landmark_rrt_star",
    "Belief RRT*": "planners/belief_rrt_star",
}

ROOT = Path(__file__).resolve().parents[1]
METRICS = ["path_efficiency", "covariance_auc", "mean_covariance_trace", "position_rmse",
           "final_position_error", "landmark_visit_fraction", "path_length", "steps"]


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        fields = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean(rows, key):
    values = [float(r[key]) for r in rows if math.isfinite(float(r[key]))]
    return statistics.mean(values) if values else None


def summarize_group(method, seed, cohort, rows):
    result = dict(method=method, seed=seed, cohort=cohort, episodes=len(rows),
                  successes=sum(r["success"] for r in rows),
                  timeouts=sum(r["timeout"] for r in rows),
                  easy=sum(r["difficulty"] == "easy" for r in rows),
                  medium=sum(r["difficulty"] == "medium" for r in rows),
                  hard=sum(r["difficulty"] == "hard" for r in rows))
    for metric in METRICS:
        subset = [r for r in rows if r["success"]] if metric in ("path_efficiency", "path_length") else rows
        result[metric] = mean(subset, metric)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/contact_free")
    parser.add_argument("--save", action="store_true", help="Also save CSVs and the full report.")
    args = parser.parse_args()
    output = args.output_dir.resolve()
    data, inputs = {}, []
    reference_ids = {}
    for method, folder in SOURCES.items():
        for seed in SEEDS:
            directory = ROOT / "results" / folder / f"seed_{seed}"
            path = directory / "test_maps_nav_0p50_loc_0p50.csv"
            source = json.loads((directory / "summary_nav_0p50_loc_0p50.json").read_text())
            assert source["test_split"] == "final_test" and source["evaluation_seed"] == 17027
            assert source["preference"] == {"navigation": 0.5, "localization": 0.5}
            assert not source["config"].get("terminate_on_collision", False)
            rows = list(csv.DictReader(path.open()))
            assert len(rows) == 200 and {int(r["map_index"]) for r in rows} == set(range(200))
            for row in rows:
                row["success"] = int(row["success"].lower() in ("1", "true"))
                row["timeout"] = int(row["timeout"])
                row["collision_steps"] = int(row["collision_steps"])
                row["map_index"] = int(row["map_index"])
                key = (seed, row["map_index"])
                identity = (row["map_id"], row["map_seed"], row["episode_seed"])
                assert reference_ids.setdefault(key, identity) == identity
            assert sum(r["success"] for r in rows) == source["overall"]["successes"]
            assert abs(mean(rows, "position_rmse") - source["overall"]["mean_position_rmse"]) < 1e-10
            data[(method, seed)] = {r["map_index"]: r for r in rows}
            inputs.append(dict(method=method, seed=seed, path=str(path.relative_to(ROOT)),
                               sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    common = {}
    for seed in SEEDS:
        common[seed] = set.intersection(*[
            {i for i, r in data[(method, seed)].items() if r["success"] and r["collision_steps"] == 0}
            for method in SOURCES])
    per_seed, selected = [], []
    cohorts = ("all_episodes", "zero_contact", "successful_zero_contact", "all_methods_common_success_zero_contact")
    for method in SOURCES:
        for seed in SEEDS:
            rows = list(data[(method, seed)].values())
            subsets = {"all_episodes": rows,
                       "zero_contact": [r for r in rows if r["collision_steps"] == 0],
                       "successful_zero_contact": [r for r in rows if r["success"] and r["collision_steps"] == 0],
                       "all_methods_common_success_zero_contact": [data[(method, seed)][i] for i in sorted(common[seed])]}
            for cohort, subset in subsets.items():
                per_seed.append(summarize_group(method, seed, cohort, subset))
                if cohort != "all_episodes":
                    for row in subset:
                        selected.append({"method": method, "seed": seed, "cohort": cohort, **row})
    aggregates = []
    for cohort in cohorts:
        for method in SOURCES:
            group = [r for r in per_seed if r["method"] == method and r["cohort"] == cohort]
            result = dict(method=method, cohort=cohort, seeds=len(group),
                          episodes=sum(r["episodes"] for r in group),
                          successful_episodes=sum(r["successes"] for r in group),
                          timeouts=sum(r["timeouts"] for r in group),
                          success_percent_of_all_episodes=100 * sum(r["successes"] for r in group) / 1000,
                          easy=sum(r["easy"] for r in group), medium=sum(r["medium"] for r in group),
                          hard=sum(r["hard"] for r in group))
            for metric in METRICS:
                values = [r[metric] for r in group if r[metric] is not None]
                result[metric] = statistics.mean(values) if values else None
                result[metric + "_seed_sd"] = statistics.stdev(values) if len(values) > 1 else None
            aggregates.append(result)
    if not args.save:
        print("Table 2 — contact-free successful trajectories; mean +/- SD over five seeds\n")
        print("Method | Efficiency | Covariance AUC | Position RMSE")
        for row in aggregates:
            if row['cohort'] != 'successful_zero_contact':
                continue
            cells = [f"{row[key]:.{digits}f} +/- {row[key + '_seed_sd']:.{digits}f}"
                     for key, digits in [("path_efficiency", 3), ("covariance_auc", 2), ("position_rmse", 3)]]
            print(row['method'] + " | " + " | ".join(cells))
        return
    output.mkdir(parents=True, exist_ok=True)
    pairs, pair_seeds = [], []
    for first, second in combinations(SOURCES, 2):
        for seed in SEEDS:
            ids = {i for i, r in data[(first, seed)].items() if r["success"] and r["collision_steps"] == 0}
            ids &= {i for i, r in data[(second, seed)].items() if r["success"] and r["collision_steps"] == 0}
            a = summarize_group(first, seed, "paired_zero_contact_success", [data[(first, seed)][i] for i in sorted(ids)])
            b = summarize_group(second, seed, "paired_zero_contact_success", [data[(second, seed)][i] for i in sorted(ids)])
            row = dict(first=first, second=second, seed=seed, common_episodes=len(ids))
            for metric in METRICS:
                row[metric + "_first"] = a[metric]
                row[metric + "_second"] = b[metric]
                row[metric + "_difference"] = a[metric] - b[metric] if a[metric] is not None and b[metric] is not None else None
            pair_seeds.append(row)
        group = [r for r in pair_seeds if r["first"] == first and r["second"] == second]
        row = dict(first=first, second=second, common_episodes=sum(r["common_episodes"] for r in group))
        for metric in METRICS:
            for suffix in ("first", "second", "difference"):
                key = metric + "_" + suffix
                values = [r[key] for r in group if r[key] is not None]
                row[key] = statistics.mean(values) if values else None
        pairs.append(row)
    write_csv(output / "per_seed.csv", per_seed)
    write_csv(output / "summary.csv", aggregates)
    write_csv(output / "selected_episodes.csv", selected)
    write_csv(output / "paired_common_success_zero_contact.csv", pairs)
    write_csv(output / "paired_per_seed.csv", pair_seeds)
    (output / "manifest.json").write_text(json.dumps(dict(
        inputs=inputs, preference=[0.5, 0.5], seeds=SEEDS, maps_per_seed=200,
        source="original recovery-enabled evaluations; no newly trained controllers",
        contact_definition="collision_steps == 0: no obstacle or boundary motion rejection",
        aggregation="Per-seed episode means, then five-seed mean and sample SD. No step pooling."), indent=2) + "\n")
    lines = ["# Original models: post-hoc zero-contact trajectory statistics", "",
             "Balanced preference (0.5, 0.5), five seeds on the same 200 official test maps. No retraining or rerunning.",
             "The original full-cohort success metric is goal arrival with recovery allowed. Zero-contact goal arrival requires original success AND collision_steps == 0.",
             "Zero-contact episodes include non-contact timeouts; successful-zero-contact episodes exclude them. Counts are map/seed episodes across 200 geometries, not independent map counts.",
             "Metrics are mean ± sample SD over the five per-seed means. Efficiency and path length use successful episodes within each cohort; other metrics use every episode in that cohort. Simulation units: RMSE/final error in m, covariance in m², AUC in m²·s.", ""]
    def fmt(row, key, decimals=3):
        value, sd = row[key], row[key + "_seed_sd"]
        return f"{value:.{decimals}f} ± {sd:.{decimals}f}" if value is not None and sd is not None else "n/a"
    for cohort in cohorts:
        lines += ["", f"## {cohort}", "",
                  "| Method | Episodes | Goals | Timeouts | Efficiency | AUC | Mean cov. | RMSE | Final error | Landmark fraction |",
                  "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for row in aggregates:
            if row["cohort"] != cohort:
                continue
            cells = [fmt(row, key, 2 if key == "covariance_auc" else 3) for key in METRICS[:6]]
            lines.append(f'| {row["method"]} | {row["episodes"]} | {row["successful_episodes"]} | {row["timeouts"]} | ' + " | ".join(cells) + " |")
    lines += ["", "## Matched zero-contact successful cases: CAPQL comparisons", "",
              "Both methods succeed without any recorded contact on exactly the same map/seed episodes.", "",
              "| Comparison | Common episodes | CAPQL AUC | Other AUC | CAPQL RMSE | Other RMSE |",
              "|---|---:|---:|---:|---:|---:|"]
    for row in pairs:
        if row["first"] == "CAPQL" or row["second"] == "CAPQL":
            own = "first" if row["first"] == "CAPQL" else "second"
            other = "second" if own == "first" else "first"
            lines.append(f'| CAPQL vs {row[other]} | {row["common_episodes"]} | '
                         f'{row["covariance_auc_" + own]:.2f} | {row["covariance_auc_" + other]:.2f} | '
                         f'{row["position_rmse_" + own]:.3f} | {row["position_rmse_" + other]:.3f} |')
    lines += ["", "## Scope", "",
              "Method-specific filtered cohorts differ in map composition. Matched cohorts control this by using identical map/seed episodes, but they condition on both/all methods succeeding and can favor easier maps. These are post-hoc conditional statistics; report cohort counts and retain the full-set arrival/contact rates.",
              "Zero-contact trajectories never enter the recovery branch. Their existing RMSE/AUC/path values therefore require no truncation or new rollout. This analysis assesses localization without recovery; it does not establish a physical model for the excluded contact episodes."]
    (output / "REPORT.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
