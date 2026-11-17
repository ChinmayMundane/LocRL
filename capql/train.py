"""Train a preference-conditioned navigation/localization MORL agent."""

import os
import random
import sys
from itertools import chain
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
from typing import Dict, Sequence, Tuple

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from gymnasium import spaces
from gymnasium.envs.registration import EnvSpec
from omegaconf import DictConfig
from torch.distributions import Normal
from tqdm.auto import tqdm
from environment.world import PathPlan

root_dir = Path(__file__).resolve().parents[1]
# Ship only CAPQL and its required upstream helper modules.
morl_repo = root_dir / "morl_import"
if not (morl_repo / "morl_baselines").is_dir():
    raise FileNotFoundError("The bundled morl_import/morl_baselines directory is missing.")
sys.path.insert(0, str(morl_repo))

try:
    from mo_gymnasium.wrappers import MORecordEpisodeStatistics
    from morl_baselines.common.evaluation import (
        log_all_multi_policy_metrics,
        log_episode_info,
    )
    from morl_baselines.common.pareto import filter_pareto_dominated
    from morl_baselines.common.performance_indicators import (
        cardinality,
        expected_utility,
        hypervolume,
    )
    from morl_baselines.common.weights import equally_spaced_weights
    from morl_baselines.multi_policy.capql.capql import CAPQL
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Install this repository's requirements.txt; "
        "the required MORL-Baselines modules are bundled in morl_import/."
    ) from exc


OBJECTIVE_NAMES = ("navigation", "localization")


def seed_training(seed: int) -> None:
    """Seed every RNG used by CAPQL, replay sampling, and network setup."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_deterministic_torch(enabled: bool) -> None:
    """Use reproducible CUDA kernels where PyTorch provides them."""
    if not enabled:
        return
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def sample_training_preference(
    rng: np.random.Generator,
    reward_dim: int,
    endpoint_probability: float,
    dirichlet_alpha: float,
) -> np.ndarray:
    """Sample interior preferences while deliberately covering exact endpoints."""
    if not 0.0 <= endpoint_probability <= 1.0:
        raise ValueError("morl_endpoint_sampling_probability must lie in [0, 1]")
    if dirichlet_alpha <= 0.0:
        raise ValueError("morl_dirichlet_alpha must be positive")
    if rng.random() < endpoint_probability:
        preference = np.zeros(reward_dim, dtype=np.float32)
        preference[int(rng.integers(reward_dim))] = 1.0
        return preference
    return rng.dirichlet(
        np.full(reward_dim, dirichlet_alpha, dtype=float)
    ).astype(np.float32)


def normalize_weights(weights: Sequence[float]) -> np.ndarray:
    """Validate and normalize a two-objective preference vector."""
    result = np.asarray(weights, dtype=np.float32)
    if result.shape != (2,):
        raise ValueError(f"Expected two MORL weights, got shape {result.shape}")
    if np.any(result < 0.0) or not np.all(np.isfinite(result)):
        raise ValueError("MORL weights must be finite and non-negative")
    total = float(result.sum())
    if total <= 0.0:
        raise ValueError("At least one MORL weight must be positive")
    return result / total


def replay_transition_done(
    terminated: bool,
    truncated: bool,
    terminal_on_truncation: bool = True,
) -> bool:
    """Return the Bellman terminal mask used by the replay buffer.

    PathPlan resets after both goal termination and truncation.  Truncations
    include the finite-horizon timeout and collision-abort condition, both of
    which end the episodic task and carry a terminal penalty.  Bootstrapping
    through either transition therefore learns values for a continuation that
    the environment never executes.
    """
    return bool(terminated or (terminal_on_truncation and truncated))


def sample_training_action(
    policy: nn.Module,
    observation: np.ndarray,
    preference: torch.Tensor | np.ndarray,
    device: torch.device | str,
) -> np.ndarray:
    """Sample the stochastic CAPQL policy used to collect training data.

    Evaluation continues to use ``policy.get_action`` (the deterministic
    squashed mean).  Training must use the reparameterized policy sample so
    replay retains action-space coverage after the initial random warm-up.
    """
    with torch.no_grad():
        action, _, _ = policy.sample(
            torch.as_tensor(observation, dtype=torch.float32, device=device),
            torch.as_tensor(preference, dtype=torch.float32, device=device),
        )
    return action.detach().cpu().numpy()


def validate_shared_task_scales(
    progress_scale: float,
    step_penalty: float,
    shared_progress_scale: float,
    shared_step_penalty: float,
) -> None:
    """Ensure the shared feasibility shaping is a subset of navigation shaping."""
    if not 0.0 <= shared_progress_scale <= progress_scale:
        raise ValueError(
            "shared_progress_reward_scale must lie between zero and "
            "progress_reward_scale"
        )
    if not 0.0 <= shared_step_penalty <= step_penalty:
        raise ValueError(
            "shared_step_penalty must lie between zero and step_penalty"
        )


def evaluation_preference_tag(preference: Sequence[float]) -> str:
    """Return a stable TensorBoard path component for a two-objective weight."""
    weight = normalize_weights(preference)
    return f"nav_{weight[0]:.2f}_loc_{weight[1]:.2f}".replace(".", "p")


def evaluate_policy_preferences(
    agent,
    eval_env: gym.Env,
    preferences: Sequence[np.ndarray],
    episodes: int,
    evaluation_seed: int,
) -> list[dict]:
    """Evaluate deterministic policies on identical validation maps and noise.

    Every preference receives the same map indices and episode seeds. This
    makes learning curves comparable across both preferences and checkpoints.
    The evaluation environment is separate from the training environment, so
    these resets cannot alter training-map or process-noise randomness.
    """
    if episodes <= 0:
        raise ValueError("Evaluation episodes must be positive")
    results = []
    for raw_preference in preferences:
        preference = normalize_weights(raw_preference)
        vector_returns = []
        discounted_vector_returns = []
        successes = []
        for episode_index in range(episodes):
            observation, _ = eval_env.reset(
                seed=evaluation_seed + episode_index,
                options={"map_index": episode_index},
            )
            vector_return = np.zeros(agent.reward_dim, dtype=np.float64)
            discounted_vector_return = np.zeros(agent.reward_dim, dtype=np.float64)
            discount = 1.0
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action = agent.eval(observation, preference)
                observation, reward, terminated, truncated, info = eval_env.step(action)
                vector_return += reward
                discounted_vector_return += discount * reward
                discount *= agent.gamma
            vector_returns.append(vector_return)
            discounted_vector_returns.append(discounted_vector_return)
            successes.append(float(info["is_success"]))

        mean_vector = np.mean(vector_returns, axis=0)
        mean_discounted_vector = np.mean(discounted_vector_returns, axis=0)
        results.append(
            {
                "preference": preference,
                "scalarized_return": float(np.dot(preference, mean_vector)),
                "discounted_scalarized_return": float(
                    np.dot(preference, mean_discounted_vector)
                ),
                "vector_return": mean_vector,
                "discounted_vector_return": mean_discounted_vector,
                "success_rate": float(np.mean(successes)),
            }
        )
    return results


def log_front_metrics_to_tensorboard(
    writer,
    results: Sequence[dict],
    step: int,
    reference_point: np.ndarray,
    num_utility_weights: int,
) -> None:
    """Log standard MORL quality indicators for the validation front."""
    front = list(
        filter_pareto_dominated(
            [result["discounted_vector_return"] for result in results]
        )
    )
    utility_weights = equally_spaced_weights(2, n=num_utility_weights)
    writer.add_scalar("eval/hypervolume", hypervolume(reference_point, front), step)
    writer.add_scalar(
        "eval/expected_utility",
        expected_utility(front, weights_set=utility_weights),
        step,
    )
    writer.add_scalar("eval/front_cardinality", cardinality(front), step)
    writer.flush()


def log_evaluation_to_tensorboard(
    writer,
    results: Sequence[dict],
    step: int,
    reference_point: np.ndarray,
    num_utility_weights: int,
) -> None:
    """Write per-preference and aggregate deterministic validation metrics."""
    scalarized_returns = []
    discounted_scalarized_returns = []
    success_rates = []
    for result in results:
        tag = evaluation_preference_tag(result["preference"])
        prefix = f"eval/{tag}"
        writer.add_scalar(f"{prefix}/scalarized_return", result["scalarized_return"], step)
        writer.add_scalar(
            f"{prefix}/discounted_scalarized_return",
            result["discounted_scalarized_return"],
            step,
        )
        writer.add_scalar(
            f"{prefix}/navigation_return_normalized",
            result["vector_return"][0],
            step,
        )
        writer.add_scalar(
            f"{prefix}/localization_return_normalized",
            result["vector_return"][1],
            step,
        )
        writer.add_scalar(f"{prefix}/success_rate", result["success_rate"], step)
        scalarized_returns.append(result["scalarized_return"])
        discounted_scalarized_returns.append(result["discounted_scalarized_return"])
        success_rates.append(result["success_rate"])

    writer.add_scalar("eval/mean_scalarized_return", np.mean(scalarized_returns), step)
    writer.add_scalar(
        "eval/mean_discounted_scalarized_return",
        np.mean(discounted_scalarized_returns),
        step,
    )
    writer.add_scalar("eval/mean_success_rate", np.mean(success_rates), step)
    log_front_metrics_to_tensorboard(
        writer,
        results,
        step,
        reference_point,
        num_utility_weights,
    )


class MORLPathPlan(gym.Wrapper):
    """Expose PathPlan as a two-objective MO-Gymnasium-style environment.

    MORL-Baselines does not accept Dict observations, so the semantic image
    and belief vector are flattened into one array. The CAPQL networks split them
    again before applying its CNN and MLP encoders.
    """

    def __init__(
        self,
        env: PathPlan,
        objective_scales: Sequence[float],
        log_metrics: bool = False,
    ) -> None:
        super().__init__(env)
        image_space = env.observation_space.spaces["image"]
        info_space = env.observation_space.spaces["info"]

        self.image_shape = image_space.shape
        self.info_dim = info_space.shape[0]
        self.objective_scales = np.asarray(objective_scales, dtype=np.float32)
        self.log_metrics = log_metrics
        self.raw_episode_return = np.zeros(2, dtype=np.float32)
        if self.objective_scales.shape != (2,) or np.any(self.objective_scales <= 0.0):
            raise ValueError("morl_objective_scales must contain two positive values")

        self.observation_space = spaces.Box(
            low=np.concatenate((image_space.low.reshape(-1), info_space.low)),
            high=np.concatenate((image_space.high.reshape(-1), info_space.high)),
            dtype=np.float32,
        )
        self.reward_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(2,),
            dtype=np.float32,
        )

        # MORL-Baselines accesses this through env.unwrapped.
        env.reward_space = self.reward_space

    @staticmethod
    def _flatten_observation(observation: Dict[str, np.ndarray]) -> np.ndarray:
        return np.concatenate(
            (observation["image"].reshape(-1), observation["info"]),
        ).astype(np.float32, copy=False)

    def reset(self, **kwargs) -> Tuple[np.ndarray, Dict]:
        observation, info = self.env.reset(**kwargs)
        self.raw_episode_return.fill(0.0)
        return self._flatten_observation(observation), info

    def step(self, action: np.ndarray):
        observation, _, terminated, truncated, info = self.env.step(action)
        components = info["reward_components"]
        raw_vector_reward = np.array(
            [components["navigation_objective"], components["localization_objective"]],
            dtype=np.float32,
        )
        vector_reward = raw_vector_reward / self.objective_scales
        self.raw_episode_return += raw_vector_reward

        info = dict(info)
        info["vector_reward_raw"] = raw_vector_reward
        info["vector_reward"] = vector_reward.copy()
        if self.log_metrics and (self.env.global_steps % 100 == 0 or terminated or truncated):
            import wandb

            if wandb.run is not None:
                metrics = {
                    "environment/covariance_trace": info["covariance_trace"],
                    "environment/true_distance_to_goal": info["true_distance_to_goal"],
                    "environment/estimated_distance_to_goal": info["estimated_distance_to_goal"],
                    "environment/position_error": info["position_error"],
                    "environment/commanded_speed": info["commanded_speed"],
                    "environment/commanded_steering": info["commanded_steering"],
                    "environment/landmarks_visited": info["landmarks_visited"],
                    "environment/information_gain": components["information_gain"],
                    "environment/collision_streak": info["collision_streak"],
                    "environment/max_collision_streak": info["max_collision_streak"],
                    "environment/collision_aborted": int(info["collision_aborted"]),
                    "environment/obstacle_clearance": info["obstacle_clearance"],
                    "global_step": self.env.global_steps,
                }
                if terminated or truncated:
                    metrics.update(
                        {
                            "objectives/navigation_episode_return_raw": self.raw_episode_return[0],
                            "objectives/localization_episode_return_raw": self.raw_episode_return[1],
                            "environment/success": int(info["is_success"]),
                        }
                    )
                wandb.log(metrics)
        return self._flatten_observation(observation), vector_reward, terminated, truncated, info


class ObservationEncoder(nn.Module):
    """Encode the semantic image and numerical EKF-belief features."""

    def __init__(
        self,
        image_shape: Tuple[int, int, int],
        info_dim: int,
    ) -> None:
        super().__init__()
        self.image_shape = tuple(image_shape)
        self.image_size = int(np.prod(image_shape))
        self.cnn = nn.Sequential(
            nn.Conv2d(image_shape[0], 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            cnn_dim = self.cnn(torch.zeros((1, *image_shape))).shape[1]
        self.output_dim = int(cnn_dim + info_dim)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        unbatched = observation.ndim == 1
        if unbatched:
            observation = observation.unsqueeze(0)
        image = observation[:, : self.image_size].reshape(-1, *self.image_shape)
        belief = observation[:, self.image_size :]
        features = torch.cat((self.cnn(image), belief), dim=-1)
        return features.squeeze(0) if unbatched else features


def make_mlp(input_dim: int, output_dim: int, net_arch: Sequence[int]) -> nn.Sequential:
    """Build a ReLU MLP while keeping its dimensions explicit."""
    layers = []
    current_dim = input_dim
    for width in net_arch:
        layers.extend((nn.Linear(current_dim, int(width)), nn.ReLU()))
        current_dim = int(width)
    layers.append(nn.Linear(current_dim, output_dim))
    return nn.Sequential(*layers)


class LocalizationPolicy(nn.Module):
    """Preference-conditioned squashed-Gaussian continuous policy."""

    def __init__(self, image_shape, info_dim, reward_dim, action_space, net_arch) -> None:
        super().__init__()
        self.encoder = ObservationEncoder(image_shape, info_dim)
        hidden_arch = [int(width) for width in net_arch]
        if not hidden_arch:
            raise ValueError("morl_net_arch must contain at least one hidden layer")
        trunk_layers = []
        current_dim = self.encoder.output_dim + reward_dim
        for width in hidden_arch:
            trunk_layers.extend((nn.Linear(current_dim, width), nn.ReLU()))
            current_dim = width
        self.trunk = nn.Sequential(*trunk_layers)
        self.mean = nn.Linear(current_dim, action_space.shape[0])
        self.log_std = nn.Linear(current_dim, action_space.shape[0])
        self.register_buffer(
            "action_scale",
            torch.as_tensor((action_space.high - action_space.low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "action_bias",
            torch.as_tensor((action_space.high + action_space.low) / 2.0, dtype=torch.float32),
        )

    def forward(self, observation: torch.Tensor, weights: torch.Tensor):
        features = self.encoder(observation)
        hidden = self.trunk(torch.cat((features, weights), dim=-1))
        return self.mean(hidden), torch.clamp(self.log_std(hidden), min=-20.0, max=2.0)

    def get_action(self, observation: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        mean, _ = self.forward(observation, weights)
        return torch.tanh(mean) * self.action_scale + self.action_bias

    def sample(self, observation: torch.Tensor, weights: torch.Tensor):
        mean, log_std = self.forward(observation, weights)
        distribution = Normal(mean, log_std.exp())
        raw_action = distribution.rsample()
        squashed_action = torch.tanh(raw_action)
        action = squashed_action * self.action_scale + self.action_bias
        log_probability = distribution.log_prob(raw_action).sum(dim=-1)
        log_probability -= torch.log(
            self.action_scale * (1.0 - squashed_action.pow(2)) + 1e-6
        ).sum(dim=-1)
        deterministic_action = torch.tanh(mean) * self.action_scale + self.action_bias
        return action, log_probability.clamp(-1e3, 1e3), deterministic_action


class LocalizationQNet(nn.Module):
    """Vector-valued critic conditioned on state, action, and preference."""

    def __init__(self, image_shape, info_dim, action_dim, reward_dim, net_arch) -> None:
        super().__init__()
        self.encoder = ObservationEncoder(image_shape, info_dim)
        self.head = make_mlp(
            self.encoder.output_dim + action_dim + reward_dim,
            reward_dim,
            net_arch,
        )

    def forward(
        self,
        observation: torch.Tensor,
        action: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        features = self.encoder(observation)
        return self.head(torch.cat((features, action, weights), dim=-1))


class LocalizationCAPQL(CAPQL):
    """Continuous-action CAPQL agent with semantic-map CNN encoders."""

    def __init__(self, env: gym.Env, image_shape, info_dim: int, **kwargs) -> None:
        super().__init__(env, **kwargs)
        self.policy = LocalizationPolicy(
            image_shape, info_dim, self.reward_dim, self.env.action_space, self.net_arch
        ).to(self.device)
        self.q_nets = [
            LocalizationQNet(
                image_shape, info_dim, self.action_dim, self.reward_dim, self.net_arch
            ).to(self.device)
            for _ in range(self.num_q_nets)
        ]
        self.target_q_nets = [
            LocalizationQNet(
                image_shape, info_dim, self.action_dim, self.reward_dim, self.net_arch
            ).to(self.device)
            for _ in range(self.num_q_nets)
        ]
        for critic, target_critic in zip(self.q_nets, self.target_q_nets):
            target_critic.load_state_dict(critic.state_dict())
            target_critic.requires_grad_(False)
        self.q_optim = optim.Adam(
            chain.from_iterable(network.parameters() for network in self.q_nets),
            lr=self.learning_rate,
        )
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.checkpoint_dir: Path | None = None

    def register_additional_config(self, conf: Dict | None = None) -> None:
        """Allow deliberate updates to training metadata in one W&B run."""
        if not conf:
            return
        import wandb

        wandb.config.update(conf, allow_val_change=True)

    def save(self, save_dir=None, filename=None, save_replay_buffer=False):
        """Route automatic CAPQL checkpoints to this experiment's directory."""
        if save_dir is None:
            save_dir = self.checkpoint_dir or Path("weights")
        return super().save(
            save_dir=str(save_dir),
            filename=filename,
            save_replay_buffer=save_replay_buffer,
        )

    def train(
        self,
        total_timesteps: int,
        eval_env: gym.Env,
        ref_point: np.ndarray,
        *,
        eval_freq: int,
        num_eval_weights_for_front: int,
        num_eval_episodes_for_front: int,
        num_eval_weights_for_eval: int,
        checkpoint_freq: int,
        tensorboard_writer=None,
        endpoint_sampling_probability: float = 0.0,
        dirichlet_alpha: float = 1.0,
        evaluation_seed: int = 0,
        evaluation_preferences: Sequence[Sequence[float]] | None = None,
    ) -> None:
        """Train CAPQL with one fixed preference per episode."""
        if self.log:
            self.register_additional_config(
                {
                    "total_timesteps": total_timesteps,
                    "ref_point": ref_point.tolist(),
                    "preference_sampling": "uniform_simplex_per_episode",
                    "eval_freq": eval_freq,
                    "checkpoint_freq": checkpoint_freq,
                    "endpoint_sampling_probability": endpoint_sampling_probability,
                    "dirichlet_alpha": dirichlet_alpha,
                }
            )

        if evaluation_preferences is None:
            eval_weights = equally_spaced_weights(
                self.reward_dim, n=num_eval_weights_for_front
            )
        else:
            eval_weights = [
                normalize_weights(weight) for weight in evaluation_preferences
            ]
        self.global_step = 0
        self.num_episodes = 0
        observation, _ = self.env.reset()

        def sample_preference() -> np.ndarray:
            return sample_training_preference(
                self.np_random,
                self.reward_dim,
                endpoint_sampling_probability,
                dirichlet_alpha,
            )

        preference = sample_preference()
        tensor_preference = torch.as_tensor(preference, device=self.device)
        episode_vector_return = np.zeros(self.reward_dim, dtype=np.float64)
        episode_length = 0

        progress = tqdm(
            range(total_timesteps),
            desc=f"Training {getattr(self, 'method_label', 'CAPQL')}",
            unit="step",
            dynamic_ncols=True,
        )
        for _ in progress:
            self.global_step += 1
            if self.global_step < self.learning_starts:
                action = self.env.action_space.sample()
            else:
                if bool(getattr(self, "stochastic_training_actions", True)):
                    action = sample_training_action(
                        self.policy,
                        observation,
                        tensor_preference,
                        self.device,
                    )
                else:
                    with torch.no_grad():
                        action = self.policy.get_action(
                            torch.as_tensor(
                                observation,
                                dtype=torch.float32,
                                device=self.device,
                            ),
                            tensor_preference,
                        ).cpu().numpy()

            next_observation, vector_reward, terminated, truncated, info = self.env.step(action)
            episode_vector_return += np.asarray(vector_reward, dtype=np.float64)
            episode_length += 1
            self.replay_buffer.push(
                observation,
                action,
                preference,
                vector_reward,
                next_observation,
                replay_transition_done(
                    terminated,
                    truncated,
                    bool(getattr(self, "terminal_on_truncation", True)),
                ),
            )
            if self.global_step >= self.learning_starts:
                self.update()

            if terminated or truncated:
                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar(
                        "rollout/navigation_return_normalized",
                        episode_vector_return[0],
                        self.global_step,
                    )
                    tensorboard_writer.add_scalar(
                        "rollout/localization_return_normalized",
                        episode_vector_return[1],
                        self.global_step,
                    )
                    tensorboard_writer.add_scalar(
                        "rollout/scalarized_return",
                        float(np.dot(preference, episode_vector_return)),
                        self.global_step,
                    )
                    tensorboard_writer.add_scalar(
                        "rollout/episode_length", episode_length, self.global_step
                    )
                    tensorboard_writer.add_scalar(
                        "rollout/success", int(info["is_success"]), self.global_step
                    )
                    if "collision_failure" in info:
                        tensorboard_writer.add_scalar(
                            "rollout/collision_failure", int(info["collision_failure"]), self.global_step
                        )
                    tensorboard_writer.add_scalar(
                        "preference/navigation", preference[0], self.global_step
                    )
                    tensorboard_writer.add_scalar(
                        "preference/localization", preference[1], self.global_step
                    )
                if self.log and "episode" in info:
                    log_episode_info(
                        info["episode"], np.dot, preference, self.global_step
                    )
                observation, _ = self.env.reset()
                self.num_episodes += 1
                preference = sample_preference()
                tensor_preference = torch.as_tensor(preference, device=self.device)
                episode_vector_return.fill(0.0)
                episode_length = 0
            else:
                observation = next_observation

            if tensorboard_writer is not None and self.global_step % 100 == 0:
                for name in (
                    "covariance_trace",
                    "position_error",
                    "true_distance_to_goal",
                    "landmarks_visited",
                    "collision_streak",
                ):
                    if name in info:
                        tensorboard_writer.add_scalar(
                            f"environment/{name}", info[name], self.global_step
                        )

            if (
                (self.log or tensorboard_writer is not None)
                and eval_freq > 0
                and self.global_step % eval_freq == 0
            ):
                evaluation_results = evaluate_policy_preferences(
                    self,
                    eval_env,
                    eval_weights,
                    num_eval_episodes_for_front,
                    evaluation_seed,
                )
                if tensorboard_writer is not None:
                    log_evaluation_to_tensorboard(
                        tensorboard_writer,
                        evaluation_results,
                        self.global_step,
                        ref_point,
                        num_eval_weights_for_eval,
                    )
                if self.log:
                    log_all_multi_policy_metrics(
                        current_front=[
                            result["discounted_vector_return"]
                            for result in evaluation_results
                        ],
                        hv_ref_point=ref_point,
                        reward_dim=self.reward_dim,
                        global_step=self.global_step,
                        n_sample_weights=num_eval_weights_for_eval,
                        ref_front=None,
                    )

            if checkpoint_freq > 0 and self.global_step % checkpoint_freq == 0:
                self.save(
                    filename=f"{getattr(self, 'checkpoint_prefix', 'capql')}_{self.global_step}",
                    save_replay_buffer=False,
                )


def make_morl_env(
    cfg: DictConfig,
    seed: int,
    *,
    log_metrics: bool = False,
    map_split: str | None = None,
    training: bool = False,
) -> gym.Env:
    """Create the wrapped vector-reward environment used for train and test."""
    base_env = PathPlan(cfg, test_mode=not training, seed=seed, map_split=map_split)
    base_env.spec = EnvSpec(
        id="LocalizationRL-MORL-v0",
        max_episode_steps=int(cfg.max_steps),
        disable_env_checker=True,
    )
    env = MORLPathPlan(base_env, cfg.morl_objective_scales, log_metrics=log_metrics)
    return MORecordEpisodeStatistics(env, gamma=float(cfg.morl_gamma))


def build_agent(cfg: DictConfig, env: gym.Env, *, training: bool, agent_class=LocalizationCAPQL) -> LocalizationCAPQL:
    """Construct a continuous CAPQL agent with identical train/test networks."""
    return agent_class(
        env,
        image_shape=(3, int(cfg.view_size), int(cfg.view_size)),
        info_dim=int(env.get_wrapper_attr("info_dim")),
        learning_rate=float(cfg.morl_learning_rate),
        tau=float(cfg.capql_tau),
        buffer_size=int(cfg.morl_buffer_size) if training else 1,
        net_arch=[int(width) for width in cfg.morl_net_arch],
        batch_size=int(cfg.morl_batch_size),
        learning_starts=int(cfg.morl_learning_starts),
        gradient_updates=int(cfg.capql_gradient_updates),
        gamma=float(cfg.morl_gamma),
        num_q_nets=int(cfg.capql_num_q_nets),
        alpha=float(cfg.capql_alpha),
        project_name=cfg.wandb_project,
        experiment_name=f"CAPQL-{cfg.model_file}",
        wandb_entity=cfg.wandb_username,
        log=bool(cfg.use_wandb) if training else False,
        seed=int(cfg.seed),
        device=str(getattr(cfg, "device", "auto")),
    )


@hydra.main(config_path=str(root_dir / "configs"), config_name="environment", version_base="1.1")
def main(cfg: DictConfig) -> None:
    run_training(cfg)


def run_training(cfg: DictConfig, agent_factory=build_agent, method_label="CAPQL") -> None:
    """Common sampling, logging, and evaluation protocol for conditioned agents."""
    seed_training(int(cfg.seed))
    configure_deterministic_torch(bool(cfg.deterministic_torch))
    validate_shared_task_scales(
        float(cfg.progress_reward_scale),
        float(cfg.step_penalty),
        float(cfg.shared_progress_reward_scale),
        float(cfg.shared_step_penalty),
    )
    train_env = make_morl_env(
        cfg,
        seed=int(cfg.seed),
        log_metrics=bool(cfg.use_wandb),
        map_split=str(cfg.train_map_split),
        training=True,
    )
    eval_env = make_morl_env(
        cfg,
        seed=int(cfg.seed) + 1,
        map_split=str(cfg.validation_map_split),
    )
    train_env.action_space.seed(int(cfg.seed))
    eval_env.action_space.seed(int(cfg.seed) + 1)
    agent = agent_factory(cfg, train_env, training=True)
    agent.stochastic_training_actions = bool(cfg.capql_stochastic_training_actions)
    agent.terminal_on_truncation = bool(cfg.capql_terminal_on_truncation)

    log_directory = root_dir / "models" / cfg.model_file
    os.makedirs(log_directory / "checkpoints", exist_ok=True)

    total_timesteps = int(cfg.total_timesteps)
    checkpoint_freq = int(cfg.checkpoint_freq)
    reference_point = np.asarray(cfg.morl_reference_point, dtype=np.float32)
    if reference_point.shape != (2,):
        raise ValueError("morl_reference_point must contain two values")

    print(f"Using device: {agent.device}")
    agent.checkpoint_dir = log_directory / "checkpoints"
    tensorboard_writer = None
    if bool(cfg.use_tensorboard):
        from torch.utils.tensorboard import SummaryWriter

        tensorboard_writer = SummaryWriter(
            log_dir=str(root_dir / "logs" / str(cfg.model_file))
        )

    print(f"Training continuous-action {method_label} for {total_timesteps:,} timesteps")
    print(f"Objectives: {OBJECTIVE_NAMES}")
    print("A new navigation/localization preference is sampled each episode.")
    print(
        "Training actions: "
        + (
            f"stochastic {method_label} samples"
            if agent.stochastic_training_actions
            else "deterministic policy means"
        )
    )
    print(
        "Replay truncations: "
        + ("terminal" if agent.terminal_on_truncation else "bootstrapped")
    )
    print(
        "Control bounds: "
        f"speed=[{train_env.action_space.low[0]:.2f}, {train_env.action_space.high[0]:.2f}], "
        f"steering=[{train_env.action_space.low[1]:.3f}, {train_env.action_space.high[1]:.3f}] rad"
    )
    if bool(cfg.use_map_splits):
        print(
            f"Map protocol: train={cfg.train_map_split}, "
            f"online evaluation={cfg.validation_map_split} (test maps remain held out)."
        )
    print(f"Models will be saved to: {log_directory}")
    if bool(getattr(cfg, "terminate_on_collision", False)):
        print("Collision rule: first contact is terminal failure; no recovery or stationary turn.")

    agent.train(
        total_timesteps=total_timesteps,
        eval_env=eval_env,
        ref_point=reference_point,
        eval_freq=int(cfg.eval_freq),
        num_eval_weights_for_front=11,
        num_eval_episodes_for_front=int(cfg.eval_episodes),
        num_eval_weights_for_eval=50,
        checkpoint_freq=checkpoint_freq,
        tensorboard_writer=tensorboard_writer,
        endpoint_sampling_probability=float(
            cfg.morl_endpoint_sampling_probability
        ),
        dirichlet_alpha=float(cfg.morl_dirichlet_alpha),
        evaluation_seed=int(cfg.eval_seed),
        evaluation_preferences=cfg.morl_eval_weights,
    )

    final_model = log_directory / "final_model.tar"
    agent.save(
        save_dir=str(log_directory),
        filename="final_model",
        save_replay_buffer=False,
    )
    print("Training complete.")
    print(f"Final continuous preference-conditioned model: {final_model}")
    print("Choose the navigation/localization trade-off in test.py with morl_test_weights.")

    train_env.close()
    eval_env.close()
    if tensorboard_writer is not None:
        tensorboard_writer.close()
    if cfg.use_wandb:
        agent.close_wandb()


if __name__ == "__main__":
    main()
