"""Load paired final evaluations and summarize five-seed metrics."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


SEEDS = [11, 22, 33, 44, 55]


GRID = [0., .2, .4, .5, .6, .8, 1.]


ANCHORS = [.2, .5, .8]


METHODS = {
    "CAPQL": ("capql", GRID),
    "PC-SAC": ("pc_sac", GRID),
    "Fixed-weight SAC": ("fixed_sac", GRID),
    "Dubins RRT*": ("planners/dubins_rrt_star", [.5]),
    "Landmark RRT*": ("planners/landmark_rrt_star", [.5]),
    "Belief Dubins RRT*": ("planners/belief_rrt_star", [.5]),
}


CHECK_KEYS = "morl_objective_scales max_steps goal_threshold continuous_dt motion_noise_std noise_cov process_cov x_cov progress_reward_scale shared_progress_reward_scale shared_step_penalty step_penalty turn_penalty collision_penalty collision_penalty_growth collision_penalty_max_multiplier collision_abort_steps collision_abort_penalty obstacle_clearance_margin obstacle_clearance_penalty revisit_penalty goal_reward timeout_penalty uncertainty_penalty information_gain_reward_scale information_budget_per_landmark landmark_progress_reward_scale terminal_uncertainty_penalty covariance_normalizer".split()


def stats(values):
    a = np.asarray(values, dtype=float)
    sd = a.std(ddof=1)
    half = 2.776445105 * sd / np.sqrt(len(a))
    return {"mean": float(a.mean()), "sample_sd": float(sd),
            "ci95_low": float(a.mean() - half), "ci95_high": float(a.mean() + half)}


def tag(w):
    return f"nav_{w:.2f}_loc_{1-w:.2f}".replace(".", "p")


def load_data():
    rows, manifest = [], []
    canonical_maps = None
    canonical_config = None
    identity = "map_index map_id difficulty map_seed episode_seed start_x start_y goal_x goal_y obstacle_count landmark_count".split()
    for method, (folder, weights) in METHODS.items():
        for seed in SEEDS:
            for w in weights:
                base = ROOT / "results" / folder / f"seed_{seed}"
                csv = base / f"test_maps_{tag(w)}.csv"
                summary = base / f"summary_{tag(w)}.json"
                d = pd.read_csv(csv)
                meta = json.loads(summary.read_text())
                if len(d) != 200 or d.map_id.nunique() != 200:
                    raise ValueError(f"Expected 200 unique maps: {csv}")
                if canonical_maps is None:
                    canonical_maps = d[identity]
                    canonical_config = {k: meta['config'].get(k) for k in CHECK_KEYS}
                pd.testing.assert_frame_equal(d[identity], canonical_maps)
                cfg = {k: meta['config'].get(k) for k in CHECK_KEYS}
                if cfg != canonical_config:
                    raise ValueError(f"Evaluation mismatch {csv}: {[k for k in cfg if cfg[k] != canonical_config[k]]}")
                assert np.isclose(meta['preference']['navigation'], w)
                vector = d[['navigation_return_normalized', 'localization_return_normalized']].to_numpy()
                raw = d[['navigation_return_raw', 'localization_return_raw']].to_numpy()
                # Environment rewards pass through float32 before accumulation.
                np.testing.assert_allclose(vector, raw / np.asarray(cfg['morl_objective_scales']), atol=1e-6, rtol=1e-6)
                np.testing.assert_allclose(d.scalarized_return, vector @ [w, 1-w], atol=1e-7)
                if not np.isfinite(vector).all():
                    raise ValueError(f"Nonfinite returns: {csv}")
                rows.append(dict(method=method, seed=seed, navigation_weight=w,
                    nav_return=vector[:, 0].mean(), loc_return=vector[:, 1].mean(),
                    success=d.success.mean(), efficiency=d.loc[d.success == 1, 'path_efficiency'].mean(),
                    success_weighted_efficiency=d.path_efficiency.where(d.success == 1, 0).mean(),
                    rmse=d.position_rmse.mean(), auc=d.covariance_auc.mean(), steps=d.steps.mean()))
                for path in [csv, summary]:
                    manifest.append(dict(path=str(path.relative_to(ROOT)), sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
    return pd.DataFrame(rows), manifest, canonical_config
