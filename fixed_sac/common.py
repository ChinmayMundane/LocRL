"""Shared scalar-SAC environment and model construction utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import gymnasium as gym
import numpy as np
import torch
from omegaconf import DictConfig
from stable_baselines3 import SAC
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


ROOT = Path(__file__).resolve().parents[1]

# Resolve shared controller imports from the repository root.
import sys

FIXED_BASELINE_DIR = ROOT / "capql"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(FIXED_BASELINE_DIR) not in sys.path:
    sys.path.insert(0, str(FIXED_BASELINE_DIR))

from capql.train import (  # noqa: E402
    ObservationEncoder,
    make_morl_env,
    normalize_weights,
)


class ScalarizedPathPlan(gym.Wrapper):
    """Turn the normalized navigation/localization reward vector into a scalar."""

    def __init__(
        self,
        env: gym.Env,
        weights: Sequence[float],
        reward_scale: float = 1.0,
    ) -> None:
        super().__init__(env)
        self.scalar_weights = normalize_weights(weights)
        self.reward_scale = float(reward_scale)
        if not np.isfinite(self.reward_scale) or self.reward_scale <= 0.0:
            raise ValueError("scalar_reward_scale must be finite and positive")

    def step(self, action):
        observation, vector_reward, terminated, truncated, info = self.env.step(action)
        vector_reward = np.asarray(vector_reward, dtype=np.float32)
        unscaled_reward = float(np.dot(self.scalar_weights, vector_reward))
        scalar_reward = self.reward_scale * unscaled_reward
        info = dict(info)
        info["scalar_weights"] = self.scalar_weights.copy()
        info["scalarized_reward_unscaled"] = unscaled_reward
        info["scalar_reward_scale"] = self.reward_scale
        info["scalarized_reward"] = scalar_reward
        return observation, scalar_reward, terminated, truncated, info


class LocalizationFeaturesExtractor(BaseFeaturesExtractor):
    """Use the same semantic-map CNN and belief features as the CAPQL agent."""

    def __init__(
        self,
        observation_space: gym.spaces.Box,
        image_shape: Sequence[int],
        info_dim: int,
    ) -> None:
        super().__init__(observation_space, features_dim=1)
        self.encoder = ObservationEncoder(tuple(image_shape), int(info_dim))
        self._features_dim = self.encoder.output_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder(observations)


def make_scalar_env(
    cfg: DictConfig,
    seed: int,
    *,
    map_split: str,
    training: bool,
) -> gym.Env:
    """Create the canonical belief-space environment with a fixed scalar reward."""
    vector_env = make_morl_env(
        cfg,
        seed,
        map_split=map_split,
        training=training,
    )
    return ScalarizedPathPlan(
        vector_env,
        cfg.morl_test_weights,
        reward_scale=float(getattr(cfg, "scalar_reward_scale", 1.0)),
    )


def parse_entropy_coefficient(value) -> float | str:
    """Validate an SB3 fixed or automatically tuned entropy coefficient."""
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized == "auto" or normalized.startswith("auto_"):
            if normalized.startswith("auto_"):
                try:
                    initial_value = float(normalized.removeprefix("auto_"))
                except ValueError as exc:
                    raise ValueError(
                        "scalar_entropy_coefficient must be a positive number, "
                        "'auto', or 'auto_<initial-value>'"
                    ) from exc
                if not np.isfinite(initial_value) or initial_value <= 0.0:
                    raise ValueError(
                        "The automatic entropy initial value must be positive"
                    )
            return normalized
    coefficient = float(value)
    if not np.isfinite(coefficient) or coefficient <= 0.0:
        raise ValueError("The fixed entropy coefficient must be finite and positive")
    return coefficient


def build_scalar_agent(cfg: DictConfig, env: gym.Env, *, seed: int) -> SAC:
    """Construct scalar SAC with hyperparameters matched to the CAPQL run."""
    return SAC(
        "MlpPolicy",
        env,
        learning_rate=float(cfg.morl_learning_rate),
        buffer_size=int(cfg.morl_buffer_size),
        learning_starts=int(cfg.morl_learning_starts),
        batch_size=int(cfg.morl_batch_size),
        tau=float(cfg.capql_tau),
        gamma=float(cfg.morl_gamma),
        train_freq=(1, "step"),
        gradient_steps=int(cfg.capql_gradient_updates),
        ent_coef=parse_entropy_coefficient(
            getattr(cfg, "scalar_entropy_coefficient", cfg.capql_alpha)
        ),
        # CAPQL treats both the finite-horizon timeout and collision abort as
        # terminal. Disable SB3's default timeout bootstrapping to match it.
        replay_buffer_kwargs={"handle_timeout_termination": False},
        policy_kwargs={
            "net_arch": [int(width) for width in cfg.morl_net_arch],
            "activation_fn": torch.nn.ReLU,
            "features_extractor_class": LocalizationFeaturesExtractor,
            "features_extractor_kwargs": {
                "image_shape": (3, int(cfg.view_size), int(cfg.view_size)),
                "info_dim": int(env.get_wrapper_attr("info_dim")),
            },
            "normalize_images": False,
        },
        seed=int(seed),
        device="auto",
        verbose=0,
    )


def scalar_model_path(cfg: DictConfig) -> Path:
    checkpoint = str(getattr(cfg, "model_checkpoint", "final_model")).removesuffix(
        ".zip"
    )
    return ROOT / "models" / str(cfg.model_file) / f"{checkpoint}.zip"
