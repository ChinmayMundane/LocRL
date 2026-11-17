"""Train and evaluate fixed-preference scalar SAC anchor specialists."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from omegaconf import OmegaConf

from run_conditioned import (
    file_sha256,
    git_is_dirty,
    git_revision,
    preference_override,
    preference_tag,
    run_command,
    selected_preferences,
    selected_seeds,
    verify_locked_test,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiments/fixed_sac_preferences.yaml"


def model_name(cfg, seed: int, preference) -> str:
    """Return a unique model name, honoring declared checkpoint reuse."""
    tag = preference_tag(preference)
    overrides = getattr(cfg, "model_name_overrides", {})
    if tag in overrides:
        return str(overrides[tag]).format(seed=int(seed), tag=tag)
    return f"{cfg.model_prefix}_{tag}_seed_{int(seed)}"


def model_path(cfg, seed: int, preference) -> Path:
    return ROOT / "models" / model_name(cfg, seed, preference) / "final_model.zip"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("status", "train", "validate", "final-test"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument(
        "--preference",
        type=float,
        nargs=2,
        action="append",
        metavar=("NAVIGATION", "LOCALIZATION"),
        help="Run only this declared preference; may be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--execute-final-test",
        action="store_true",
        help="Required acknowledgement that the specialist bank is frozen.",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    cfg = OmegaConf.load(config_path)
    seeds = selected_seeds(list(cfg.seeds), args.seeds)
    preferences = selected_preferences(cfg.preferences, args.preference)
    entrypoint_directory = ROOT / str(cfg.entrypoint_dir)
    output_root = ROOT / str(cfg.output_root)
    python = sys.executable

    locked_config_path = ROOT / str(cfg.locked_test_config)
    locked_cfg = verify_locked_test(locked_config_path)
    dataset_hashes = {
        split: file_sha256(ROOT / str(locked_cfg.dataset_dir) / f"{split}.json")
        for split in ("train", "validation", str(locked_cfg.split))
    }
    provenance = {
        "experiment_config": str(config_path),
        "experiment_config_sha256": file_sha256(config_path),
        "locked_test_config_sha256": file_sha256(locked_config_path),
        "dataset_sha256": dataset_hashes,
    }

    if args.stage == "status":
        print(f"Protocol: {config_path}")
        print(f"Git revision: {git_revision()}")
        print(f"Git worktree dirty: {git_is_dirty()}")
        print(f"Seeds: {seeds}")
        print(f"Preferences: {preferences}")
        print(f"dataset hashes: {json.dumps(dataset_hashes, indent=2)}")
        for preference in preferences:
            tag = preference_tag(preference)
            for seed in seeds:
                path = model_path(cfg, seed, preference)
                print(
                    f"{tag}, seed {seed}: "
                    f"{'trained' if path.is_file() else 'not trained'} ({path})"
                )
        return

    if args.stage == "final-test" and not args.execute_final_test:
        raise RuntimeError(
            "Final testing is locked. Validate and freeze the complete specialist "
            "bank before re-running with --execute-final-test."
        )

    for preference in preferences:
        tag = preference_tag(preference)
        weights_override = preference_override(preference)
        for seed in seeds:
            name = model_name(cfg, seed, preference)
            path = model_path(cfg, seed, preference)
            if args.stage == "train":
                if path.is_file():
                    print(f"Skipping {tag}, seed {seed}; model exists: {path}")
                    continue
                command = [
                    python,
                    str(entrypoint_directory / "train.py"),
                    f"seed={seed}",
                    f"model_file={name}",
                    f"total_timesteps={int(cfg.total_timesteps)}",
                    f"checkpoint_freq={int(cfg.checkpoint_freq)}",
                    f"eval_freq={int(cfg.eval_freq)}",
                    f"eval_episodes={int(cfg.eval_episodes)}",
                    f"deterministic_torch={str(bool(cfg.deterministic_torch)).lower()}",
                    weights_override,
                    "morl_objective_scales="
                    f"[{float(cfg.objective_scales[0]):g},{float(cfg.objective_scales[1]):g}]",
                    f"morl_learning_rate={float(cfg.learning_rate):g}",
                    "morl_net_arch="
                    f"[{','.join(str(int(width)) for width in cfg.net_arch)}]",
                    f"morl_batch_size={int(cfg.batch_size)}",
                    f"morl_buffer_size={int(cfg.buffer_size)}",
                    f"morl_learning_starts={int(cfg.learning_starts)}",
                    f"morl_gamma={float(cfg.gamma):g}",
                    f"capql_tau={float(cfg.tau):g}",
                    f"scalar_entropy_coefficient={cfg.entropy_coefficient}",
                    f"scalar_reward_scale={float(cfg.reward_scale):g}",
                    f"capql_gradient_updates={int(cfg.gradient_updates)}",
                    f"uncertainty_penalty={float(cfg.uncertainty_penalty):g}",
                    "information_gain_reward_scale="
                    f"{float(cfg.information_gain_reward_scale):g}",
                    "landmark_progress_reward_scale="
                    f"{float(cfg.landmark_progress_reward_scale):g}",
                    "terminal_uncertainty_penalty="
                    f"{float(cfg.terminal_uncertainty_penalty):g}",
                    "shared_progress_reward_scale="
                    f"{float(cfg.shared_progress_reward_scale):g}",
                    f"shared_step_penalty={float(cfg.shared_step_penalty):g}",
                    "use_wandb=false",
                    "use_tensorboard=true",
                ]
                if "training_collision_abort_steps" in cfg:
                    command.append(
                        "++training_collision_abort_steps="
                        f"{int(cfg.training_collision_abort_steps)}"
                    )
                if "terminate_on_collision" in cfg:
                    command.append(f"terminate_on_collision={str(bool(cfg.terminate_on_collision)).lower()}")
                    command.append(f"eval_seed={int(cfg.validation_eval_seed)}")
                if args.dry_run:
                    print(" ".join(command))
                    continue
                run_command(
                    command,
                    stage=f"train-{tag}",
                    seed=seed,
                    output_root=output_root,
                    log_name=f"train_{tag}_seed_{seed}",
                    provenance=provenance,
                )
                continue

            if not path.is_file() and not args.dry_run:
                raise FileNotFoundError(
                    f"Train {tag}, seed {seed} before evaluation: {path}"
                )
            final_test = args.stage == "final-test"
            split = str(locked_cfg.split) if final_test else str(cfg.development_split)
            evaluation_seed = (
                int(locked_cfg.evaluation_seed)
                if final_test
                else int(cfg.validation_eval_seed)
            )
            result_label = str(getattr(locked_cfg, "result_label", "final_test"))
            stage_output = output_root / (result_label if final_test else "validation")
            seed_output = stage_output / f"seed_{seed}"
            expected = seed_output / f"summary_{tag}.json"
            if expected.is_file():
                print(f"Skipping existing result: {expected}")
                continue
            command = [
                python,
                str(entrypoint_directory / "evaluate.py"),
                f"seed={seed}",
                f"model_file={name}",
                "model_checkpoint=final_model",
                f"test_map_split={split}",
                f"eval_seed={evaluation_seed}",
                f"full_eval_output_dir={seed_output}",
                weights_override,
            ]
            if final_test:
                command.append(f"map_generation_seed={int(locked_cfg.generation_seed)}")
            if "terminate_on_collision" in cfg:
                command.append(f"terminate_on_collision={str(bool(cfg.terminate_on_collision)).lower()}")
            if args.dry_run:
                print(" ".join(command))
                continue
            run_command(
                command,
                stage=f"{args.stage}-{tag}",
                seed=seed,
                output_root=output_root,
                log_name=f"{args.stage}_{tag}_seed_{seed}",
                provenance=provenance,
            )


if __name__ == "__main__":
    main()
