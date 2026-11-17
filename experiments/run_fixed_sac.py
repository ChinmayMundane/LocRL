"""Run the reproducible five-seed fixed-scalar SAC baseline."""

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
    selected_seeds,
    verify_locked_test,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiments/fixed_sac.yaml"


def model_name(cfg, seed: int) -> str:
    return f"{cfg.model_prefix}_{seed}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("status", "train", "validate", "final-test"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--execute-final-test",
        action="store_true",
        help="Required acknowledgement that every method is frozen.",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    seeds = selected_seeds(list(cfg.seeds), args.seeds)
    entrypoint_directory = ROOT / str(cfg.entrypoint_dir)
    output_root = ROOT / str(cfg.output_root)
    python = sys.executable

    locked_config_path = ROOT / str(cfg.locked_test_config)
    locked_cfg = verify_locked_test(locked_config_path)
    dataset_hashes = {
        split: file_sha256(
            ROOT / str(locked_cfg.dataset_dir) / f"{split}.json"
        )
        for split in ("train", "validation", str(locked_cfg.split))
    }
    provenance = {
        "experiment_config": str(args.config.resolve()),
        "experiment_config_sha256": file_sha256(args.config.resolve()),
        "locked_test_config_sha256": file_sha256(locked_config_path),
        "dataset_sha256": dataset_hashes,
    }

    if args.stage == "status":
        print(f"Protocol: {args.config}")
        print(f"Git revision: {git_revision()}")
        print(f"Git worktree dirty: {git_is_dirty()}")
        print(f"Seeds: {seeds}")
        print(f"Fixed weights: {list(cfg.weights)}")
        print(f"dataset hashes: {json.dumps(dataset_hashes, indent=2)}")
        for seed in seeds:
            path = ROOT / "models" / model_name(cfg, seed) / "final_model.zip"
            print(f"seed {seed}: {'trained' if path.is_file() else 'not trained'}")
        return

    if args.stage == "final-test" and not args.execute_final_test:
        raise RuntimeError(
            "Final testing is locked. Re-run with --execute-final-test only after "
            "all methods and configurations are frozen."
        )

    weights_override = preference_override(cfg.weights)
    for seed in seeds:
        name = model_name(cfg, seed)
        model_path = ROOT / "models" / name / "final_model.zip"
        if args.stage == "train":
            if model_path.is_file():
                print(f"Skipping seed {seed}; model already exists: {model_path}")
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
                f"scalar_reward_scale={float(getattr(cfg, 'reward_scale', 1.0)):g}",
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
                stage="train",
                seed=seed,
                output_root=output_root,
                log_name=f"train_seed_{seed}",
                provenance=provenance,
            )
            continue

        if not model_path.is_file() and not args.dry_run:
            raise FileNotFoundError(f"Train seed {seed} before evaluation: {model_path}")
        final_test = args.stage == "final-test"
        split = str(locked_cfg.split) if final_test else str(cfg.development_split)
        evaluation_seed = (
            int(locked_cfg.evaluation_seed)
            if final_test
            else int(cfg.validation_eval_seed)
        )
        final_label = str(getattr(locked_cfg, "result_label", "final_test"))
        stage_output = output_root / (final_label if final_test else "validation")
        seed_output = stage_output / f"seed_{seed}"
        expected_summary = seed_output / f"summary_{preference_tag(cfg.weights)}.json"
        if expected_summary.is_file():
            print(f"Skipping existing result: {expected_summary}")
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
            stage=args.stage,
            seed=seed,
            output_root=output_root,
            log_name=(
                f"{final_label}_seed_{seed}_{preference_tag(cfg.weights)}"
                if final_test
                else f"{args.stage}_seed_{seed}_{preference_tag(cfg.weights)}"
            ),
            provenance=provenance,
        )


if __name__ == "__main__":
    main()
