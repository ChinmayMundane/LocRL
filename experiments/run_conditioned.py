"""Run reproducible five-seed training and evaluation for the main method."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/experiments/capql.yaml"


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def preference_override(preference: Iterable[float]) -> str:
    values = [float(value) for value in preference]
    if len(values) != 2 or min(values) < 0.0 or sum(values) <= 0.0:
        raise ValueError(f"Invalid preference: {values}")
    total = sum(values)
    return f"morl_test_weights=[{values[0] / total:g},{values[1] / total:g}]"


def preference_tag(preference: Iterable[float]) -> str:
    values = [float(value) for value in preference]
    total = sum(values)
    nav, loc = values[0] / total, values[1] / total
    return f"nav_{nav:.2f}_loc_{loc:.2f}".replace(".", "p")


def preference_list_override(preferences: Iterable[Iterable[float]]) -> str:
    """Serialize the frozen periodic-evaluation preference set for Hydra."""
    normalized = []
    for preference in preferences:
        values = [float(value) for value in preference]
        total = sum(values)
        if len(values) != 2 or min(values) < 0.0 or total <= 0.0:
            raise ValueError(f"Invalid preference: {values}")
        normalized.append(f"[{values[0] / total:g},{values[1] / total:g}]")
    return "morl_eval_weights=[" + ",".join(normalized) + "]"


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def git_is_dirty() -> bool:
    result = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    return result.returncode != 0 or bool(result.stdout.strip())


def append_run_record(output_root: Path, record: dict) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    with (output_root / "run_manifest.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(record, sort_keys=True) + "\n")


def run_command(
    command: list[str],
    *,
    stage: str,
    seed: int,
    output_root: Path,
    log_name: str,
    provenance: dict,
) -> None:
    log_dir = output_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{log_name}.log"
    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = str(seed)
    environment.setdefault("MPLCONFIGDIR", "/tmp/localizationrl-matplotlib")
    environment.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "stage": stage,
        "seed": seed,
        "command": command,
        "git_revision": git_revision(),
        "git_dirty": git_is_dirty(),
        "provenance": provenance,
        "log": str(log_path.relative_to(ROOT)),
    }
    append_run_record(output_root, record)
    print(f"\n[{stage}] seed={seed}\n{' '.join(command)}", flush=True)
    with log_path.open("wb") as log_stream:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        assert process.stdout is not None
        while chunk := process.stdout.read1(4096):
            # Preserve carriage returns so tqdm updates one progress bar.
            if hasattr(sys.stdout, "buffer"):
                sys.stdout.buffer.write(chunk)
                sys.stdout.buffer.flush()
            else:
                sys.stdout.write(chunk.decode("utf-8", errors="replace"))
                sys.stdout.flush()
            log_stream.write(chunk)
            log_stream.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def verify_locked_test(config_path: Path) -> object:
    cfg = OmegaConf.load(config_path)
    manifest_path = ROOT / str(cfg.dataset_dir) / f"{cfg.split}.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Locked test manifest is missing: {manifest_path}. Generate it first."
        )
    expected = str(cfg.sha256)
    actual = file_sha256(manifest_path)
    if expected == "PENDING_GENERATION":
        raise RuntimeError("Record the generated manifest checksum in final_test.yaml")
    if actual != expected:
        raise RuntimeError(
            f"Locked test checksum mismatch: expected {expected}, found {actual}"
        )
    return cfg


def selected_seeds(configured: list[int], requested: list[int] | None) -> list[int]:
    seeds = [int(seed) for seed in configured]
    if requested is None:
        return seeds
    unknown = sorted(set(requested) - set(seeds))
    if unknown:
        raise ValueError(f"Seeds are not declared in the protocol: {unknown}")
    return [seed for seed in seeds if seed in requested]


def selected_preferences(
    configured: Iterable[Iterable[float]],
    requested: list[list[float]] | None,
) -> list[list[float]]:
    """Return only protocol-declared preferences, preserving protocol order."""
    normalized_configured = []
    for preference in configured:
        values = [float(value) for value in preference]
        total = sum(values)
        normalized_configured.append([values[0] / total, values[1] / total])
    if requested is None:
        return normalized_configured
    normalized_requested = []
    for preference in requested:
        values = [float(value) for value in preference]
        if len(values) != 2 or min(values) < 0.0 or sum(values) <= 0.0:
            raise ValueError(f"Invalid requested preference: {values}")
        total = sum(values)
        normalized_requested.append([values[0] / total, values[1] / total])
    unknown = [
        preference
        for preference in normalized_requested
        if not any(
            all(abs(a - b) < 1e-9 for a, b in zip(preference, declared))
            for declared in normalized_configured
        )
    ]
    if unknown:
        raise ValueError(f"Preferences are not declared in the protocol: {unknown}")
    return [
        preference
        for preference in normalized_configured
        if any(
            all(abs(a - b) < 1e-9 for a, b in zip(preference, wanted))
            for wanted in normalized_requested
        )
    ]


def model_name_for_seed(cfg, seed: int) -> str:
    """Resolve an explicitly promoted pilot model or the standard seed name."""
    overrides = getattr(cfg, "seed_model_overrides", {})
    key = str(seed)
    if key in overrides:
        return str(overrides[key])
    return f"{cfg.model_prefix}_{seed}"


def model_checkpoint_for_seed(cfg, seed: int) -> str:
    """Resolve an explicitly selected checkpoint or the completed final model."""
    overrides = getattr(cfg, "seed_checkpoint_overrides", {})
    key = str(seed)
    if key in overrides:
        return str(overrides[key]).removesuffix(".tar")
    return "final_model"


def model_path_for_seed(cfg, seed: int) -> Path:
    model_name = model_name_for_seed(cfg, seed)
    checkpoint = model_checkpoint_for_seed(cfg, seed)
    return ROOT / "models" / model_name / f"{checkpoint}.tar"


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
        help="Evaluate only this declared preference; may be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--execute-final-test",
        action="store_true",
        help="Required acknowledgement that method selection is frozen.",
    )
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config)
    seeds = selected_seeds(list(cfg.seeds), args.seeds)
    preferences = selected_preferences(cfg.preferences, args.preference)
    entrypoint_dir = ROOT / str(cfg.entrypoint_dir)
    output_root = ROOT / str(cfg.output_root)
    python = sys.executable

    locked_cfg_path = ROOT / str(cfg.locked_test_config)
    locked_cfg = verify_locked_test(locked_cfg_path)
    dataset_hashes = {
        split: file_sha256(ROOT / str(locked_cfg.dataset_dir) / f"{split}.json")
        for split in ("train", "validation", str(locked_cfg.split))
    }
    provenance = {
        "experiment_config": str(args.config.resolve()),
        "experiment_config_sha256": file_sha256(args.config.resolve()),
        "locked_test_config_sha256": file_sha256(locked_cfg_path),
        "dataset_sha256": dataset_hashes,
    }

    if args.stage == "status":
        print(f"Protocol: {args.config}")
        print(f"Git revision: {git_revision()}")
        print(f"Git worktree dirty: {git_is_dirty()}")
        print(f"Seeds: {seeds}")
        print(f"Preferences: {preferences}")
        print(f"dataset hashes: {json.dumps(dataset_hashes, indent=2)}")
        for seed in seeds:
            model_name = model_name_for_seed(cfg, seed)
            checkpoint = model_checkpoint_for_seed(cfg, seed)
            model = model_path_for_seed(cfg, seed)
            print(
                f"seed {seed}: {'trained' if model.is_file() else 'not trained'} "
                f"({model_name}/{checkpoint}.tar)"
            )
        return

    if args.stage == "final-test" and not args.execute_final_test:
        raise RuntimeError(
            "Final testing is locked. Re-run with --execute-final-test only after "
            "all methods, hyperparameters, preferences, and checkpoints are frozen."
        )

    for seed in seeds:
        model_name = model_name_for_seed(cfg, seed)
        checkpoint = model_checkpoint_for_seed(cfg, seed)
        model_path = model_path_for_seed(cfg, seed)
        if args.stage == "train":
            if model_path.is_file():
                print(f"Skipping seed {seed}; model already exists: {model_path}")
                continue
            command = [
                python,
                str(entrypoint_dir / "train.py"),
                f"seed={seed}",
                f"model_file={model_name}",
                f"total_timesteps={int(cfg.total_timesteps)}",
                f"checkpoint_freq={int(cfg.checkpoint_freq)}",
                f"eval_freq={int(cfg.eval_freq)}",
                f"eval_episodes={int(cfg.eval_episodes)}",
                f"capql_alpha={float(getattr(cfg, 'entropy_coefficient', 0.2)):g}",
                f"use_wandb={str(bool(cfg.use_wandb)).lower()}",
                f"wandb_name={cfg.experiment_name}_seed_{seed}",
                f"deterministic_torch={str(bool(cfg.deterministic_torch)).lower()}",
                "capql_stochastic_training_actions="
                f"{str(bool(cfg.stochastic_training_actions)).lower()}",
                "capql_terminal_on_truncation="
                f"{str(bool(cfg.terminal_on_truncation)).lower()}",
                "morl_objective_scales="
                f"[{float(cfg.objective_scales[0]):g},{float(cfg.objective_scales[1]):g}]",
                "morl_reference_point="
                f"[{float(cfg.reference_point[0]):g},{float(cfg.reference_point[1]):g}]",
                preference_list_override(cfg.online_eval_preferences),
                "morl_endpoint_sampling_probability="
                f"{float(cfg.endpoint_sampling_probability):g}",
                f"morl_dirichlet_alpha={float(cfg.dirichlet_alpha):g}",
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
        generation_seed = (
            int(locked_cfg.generation_seed)
            if final_test
            else None
        )
        eval_seed = (
            int(locked_cfg.evaluation_seed)
            if final_test
            else int(cfg.validation_eval_seed)
        )
        final_label = str(getattr(locked_cfg, "result_label", "final_test"))
        stage_output = output_root / (final_label if final_test else "validation")
        seed_output = stage_output / f"seed_{seed}"
        for preference in preferences:
            tag = preference_tag(preference)
            expected_summary = seed_output / f"summary_{tag}.json"
            if expected_summary.is_file():
                print(f"Skipping existing result: {expected_summary}")
                continue
            command = [
                python,
                str(entrypoint_dir / "evaluate.py"),
                f"seed={seed}",
                f"model_file={model_name}",
                f"model_checkpoint={checkpoint}",
                f"test_map_split={split}",
                f"eval_seed={eval_seed}",
                f"full_eval_output_dir={seed_output}",
                preference_override(preference),
            ]
            if generation_seed is not None:
                command.append(f"map_generation_seed={generation_seed}")
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
                    f"{final_label}_seed_{seed}_{tag}"
                    if final_test
                    else f"{args.stage}_seed_{seed}_{tag}"
                ),
                provenance=provenance,
            )


if __name__ == "__main__":
    main()
