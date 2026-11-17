"""Train a fixed-weight scalar SAC baseline for belief-space navigation."""

from __future__ import annotations

import json
import os
from pathlib import Path

import gymnasium as gym
import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.logger import configure
from stable_baselines3.common.monitor import Monitor

from common import ROOT, build_scalar_agent, make_scalar_env

from capql.train import (  # noqa: E402
    configure_deterministic_torch,
    evaluate_policy_preferences,
    evaluation_preference_tag,
    make_morl_env,
    normalize_weights,
    seed_training,
    validate_shared_task_scales,
)


class ScalarPolicyAdapter:
    """Expose an SB3 SAC policy through CAPQL's deterministic eval interface."""

    reward_dim = 2

    def __init__(self, model, gamma: float) -> None:
        self.model = model
        self.gamma = float(gamma)

    def eval(self, observation, _preference):
        action, _ = self.model.predict(observation, deterministic=True)
        return action


class FixedScalarEvaluationCallback(BaseCallback):
    """Evaluate one frozen scalar preference on fixed validation episodes."""

    def __init__(
        self,
        eval_env: gym.Env,
        weights,
        *,
        eval_freq: int,
        episodes: int,
        evaluation_seed: int,
        history_path: Path,
    ) -> None:
        super().__init__(verbose=0)
        self.eval_env = eval_env
        self.weights = normalize_weights(weights)
        self.eval_freq = int(eval_freq)
        self.episodes = int(episodes)
        self.evaluation_seed = int(evaluation_seed)
        self.history_path = history_path

    def _on_step(self) -> bool:
        if self.eval_freq <= 0 or self.num_timesteps % self.eval_freq:
            return True
        result = evaluate_policy_preferences(
            ScalarPolicyAdapter(self.model, self.model.gamma),
            self.eval_env,
            [self.weights],
            self.episodes,
            self.evaluation_seed,
        )[0]
        tag = evaluation_preference_tag(self.weights)
        prefix = f"eval/{tag}"
        values = {
            "scalarized_return": float(result["scalarized_return"]),
            "discounted_scalarized_return": float(
                result["discounted_scalarized_return"]
            ),
            "navigation_return_normalized": float(result["vector_return"][0]),
            "localization_return_normalized": float(result["vector_return"][1]),
            "success_rate": float(result["success_rate"]),
        }
        for name, value in values.items():
            self.logger.record(f"{prefix}/{name}", value)
        self.logger.record("eval/mean_reward", values["scalarized_return"])
        self.logger.record("eval/mean_success_rate", values["success_rate"])
        self.logger.dump(self.num_timesteps)

        record = {
            "timesteps": int(self.num_timesteps),
            "preference": self.weights.tolist(),
            **values,
        }
        with self.history_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, sort_keys=True) + "\n")
        return True


@hydra.main(
    config_path=str(ROOT / "configs"),
    config_name="environment",
    version_base="1.1",
)
def main(cfg: DictConfig) -> None:
    seed_training(int(cfg.seed))
    configure_deterministic_torch(bool(cfg.deterministic_torch))
    validate_shared_task_scales(
        float(cfg.progress_reward_scale),
        float(cfg.step_penalty),
        float(cfg.shared_progress_reward_scale),
        float(cfg.shared_step_penalty),
    )
    weights = normalize_weights(cfg.morl_test_weights)
    train_env = make_scalar_env(
        cfg,
        int(cfg.seed),
        map_split=str(cfg.train_map_split),
        training=True,
    )
    vector_eval_env = make_morl_env(
        cfg,
        int(cfg.seed) + 1,
        map_split=str(cfg.validation_map_split),
        training=False,
    )
    train_env.action_space.seed(int(cfg.seed))
    vector_eval_env.action_space.seed(int(cfg.seed) + 1)

    model_directory = ROOT / "models" / str(cfg.model_file)
    checkpoint_directory = model_directory / "checkpoints"
    monitor_directory = model_directory / "monitor"
    checkpoint_directory.mkdir(parents=True, exist_ok=True)
    monitor_directory.mkdir(parents=True, exist_ok=True)
    monitored_train_env = Monitor(
        train_env,
        str(monitor_directory / "train"),
        info_keywords=("is_success",),
    )
    model = build_scalar_agent(cfg, monitored_train_env, seed=int(cfg.seed))
    if bool(cfg.use_tensorboard):
        model.set_logger(
            configure(str(ROOT / "logs" / str(cfg.model_file)), ["tensorboard"])
        )

    history_path = model_directory / "periodic_evaluations.jsonl"
    callbacks = CallbackList(
        [
            CheckpointCallback(
                save_freq=int(cfg.checkpoint_freq),
                save_path=str(checkpoint_directory),
                name_prefix="scalar_sac",
                save_replay_buffer=False,
                save_vecnormalize=False,
                verbose=0,
            ),
            FixedScalarEvaluationCallback(
                vector_eval_env,
                weights,
                eval_freq=int(cfg.eval_freq),
                episodes=int(cfg.eval_episodes),
                evaluation_seed=int(cfg.eval_seed),
                history_path=history_path,
            ),
        ]
    )

    total_timesteps = int(cfg.total_timesteps)
    print(f"Using device: {model.device}")
    print(f"Training fixed-weight scalar SAC for {total_timesteps:,} timesteps")
    print(
        "Fixed scalarization: "
        f"navigation={weights[0]:.2f}, localization={weights[1]:.2f}"
    )
    print(
        f"Training reward scale: {float(cfg.scalar_reward_scale):g}; "
        f"entropy coefficient: {cfg.scalar_entropy_coefficient}"
    )
    print(
        f"Evaluation: {int(cfg.eval_episodes)} fixed validation episodes every "
        f"{int(cfg.eval_freq):,} steps"
    )
    print(f"Models will be saved to: {model_directory}")
    if bool(getattr(cfg, "terminate_on_collision", False)):
        print("Collision rule: first contact is terminal failure; no recovery or stationary turn.")
    model.learn(
        total_timesteps=total_timesteps,
        callback=callbacks,
        log_interval=10,
        progress_bar=True,
        reset_num_timesteps=True,
    )
    final_model = model_directory / "final_model"
    model.save(str(final_model))
    metadata = {
        "method": "fixed_scalar_sac",
        "seed": int(cfg.seed),
        "total_timesteps": total_timesteps,
        "weights": weights.tolist(),
        "training_reward_scale": float(cfg.scalar_reward_scale),
        "entropy_coefficient": str(cfg.scalar_entropy_coefficient),
        "objective_scales": [float(value) for value in cfg.morl_objective_scales],
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    (model_directory / "metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("Training complete.")
    print(f"Final scalar SAC model: {final_model}.zip")

    monitored_train_env.close()
    vector_eval_env.close()


if __name__ == "__main__":
    # Avoid Matplotlib attempting to cache outside the workspace on clusters.
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/localizationrl-matplotlib")
    main()
