"""Python class implementation of the LocalizationRL world."""

import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import gymnasium as gym
import hydra
import numpy as np
from gymnasium import spaces
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from PIL import Image
from scipy.stats import chi2, multivariate_normal
from skimage.draw import ellipse

from environment.map_dataset import load_map_split, rasterize_map

path = os.getcwd() + "/data/"
plt.ion()
plt.close()
figure_on = 1


def onclose() -> None:
    """Render function."""
    global figure_on
    figure_on = 0


class Config:
    def __init__(self, config_dict: Dict):
        for key, value in config_dict.items():
            if isinstance(value, dict):
                value = Config(value)
            setattr(self, key, value)


class PathPlan(gym.Env):
    """Grid navigation environment with localization uncertainty."""

    def __init__(
        self,
        cfg: DictConfig,
        test_mode: bool = False,
        seed: int = 24,
        map_split: Optional[str] = None,
    ) -> None:
        """Constructor method.

        Args:
            test_mode (Optional[bool], optional): _description_. Defaults to None.
        """
        super().__init__()

        super().reset(seed=seed)
        self.cfg = cfg

        self.map_split = map_split
        self.map_specs: List[Dict] = []
        self.current_map_index = -1
        self.current_map_id = "legacy"
        self.current_map_difficulty = "legacy"
        if bool(getattr(cfg, "use_map_splits", False)) and map_split is not None:
            dataset_dir = Path(str(cfg.map_dataset_dir))
            if not dataset_dir.is_absolute():
                dataset_dir = Path(__file__).resolve().parents[1] / dataset_dir
            self.map_specs = load_map_split(dataset_dir, map_split, int(cfg.map_size))

        self.world_map = np.zeros((self.cfg.map_size, self.cfg.map_size))
        self.world_map = np.pad(self.world_map, pad_width=1, mode="constant", constant_values=1)

        self.goal_reached = False

        self.counter = 0
        self.episodes = 0
        self.test_mode = test_mode

        self.render_init = 1

        if self.map_specs:
            self._apply_map_spec(0)
        else:
            # Legacy single-layout mode remains available with
            # use_map_splits=false.
            self.start_point, self.goal_point = (45, 5), (5, 45)
            self.goal_x, self.goal_y = self.goal_point
            self.start_x, self.start_y = self.start_point
            self.robot_x, self.robot_y = self.start_point
            n = 2
            self.goal_region = [
                (self.goal_x + dx, self.goal_y + dy)
                for dx in range(-n, n + 1)
                for dy in range(-n, n + 1)
            ]
            xs, ys = zip(*self.goal_region)
            self.world_map[ys, xs] = 4
            self.world_map[self.robot_y, self.robot_x] = 3
            self.obs_list = self.get_random_obstacles(self.cfg.num_obstacle)
            self.land_list = self.get_random_landmarks(self.cfg.num_landmark)

        self.obs_visit = np.zeros(len(self.obs_list))
        self.land_visit = np.zeros(len(self.land_list))

        self.robot_heading = self.wrap_angle(
            np.arctan2(self.start_y - self.goal_y, self.goal_x - self.start_x)
        )

        self.episodes = 0
        self.steps = 0

        self.poses = []

        # The simulator keeps the physical pose separate from the EKF belief.
        # ``robot_*`` remains a backwards-compatible alias for the hidden true
        # pose used by rendering and collision checks. ``x_state`` is the
        # belief mean and is the only global pose available to the policy.
        self.true_state = np.array(
            [self.robot_x, self.robot_y, self.robot_heading],
            dtype=float,
        )
        self.x_state = self.true_state.copy()

        self.prev_cell_value = 0

        # EKF Parameters
        # Convert EKF matrices to numpy arrays. ``motion_noise_std`` is used
        # both to sample hidden physical disturbances and to propagate the EKF
        # covariance, so simulated errors and the filter model are consistent.
        self.x_cov = np.array(cfg.x_cov, dtype=float)
        self.Q = np.array(cfg.process_cov, dtype=float)
        self.R = np.array(cfg.noise_cov, dtype=float)
        self.motion_noise_std = np.asarray(
            getattr(cfg, "motion_noise_std", [0.05, 0.05, 0.0075]),
            dtype=float,
        )
        if self.motion_noise_std.shape != (3,) or np.any(self.motion_noise_std < 0.0):
            raise ValueError("motion_noise_std must contain three non-negative values")
        self.process_noise_cov = np.diag(np.square(self.motion_noise_std))

        # Find the maximum eigenvalue
        self.eigenvalues = np.linalg.eigvals(self.x_cov)
        self.determinant = np.linalg.det(self.x_cov)
        self.max_eigenvalue = np.max(self.eigenvalues)

        self.true_path = [self.true_state.copy()]
        self.estimated_path = [self.x_state.copy()]

        # Initialize previous distances to goal
        self.distance_to_goal = np.sqrt((self.robot_x - self.goal_x) ** 2 + (self.robot_y - self.goal_y) ** 2)
        self.prev_distance_to_goal = self.distance_to_goal
        self.estimated_distance_to_goal = self.distance_to_goal
        self.prev_estimated_distance_to_goal = self.estimated_distance_to_goal
        self.last_progress = 0.0
        self.last_belief_progress = 0.0
        self.last_landmark_progress = 0.0
        self.last_information_gain = 0.0
        self.landmark_information_gain = np.zeros(len(self.land_list), dtype=float)
        self.visit_counts = {}
        self.collision_streak = 0
        self.max_collision_streak = 0
        self.collision_aborted = False
        self.collision_failed = False
        self.reward_components = {}
        self.last_measurement = None
        self.last_innovation = None

        self.generate_path_info()

        # Forward-only continuous bicycle controls: [speed, steering angle].
        self.min_speed = float(getattr(cfg, "continuous_min_speed", 0.25))
        self.max_speed = float(getattr(cfg, "continuous_max_speed", 1.0))
        self.max_steering = float(getattr(cfg, "continuous_max_steering", np.pi / 4.0))
        self.wheelbase = float(getattr(cfg, "continuous_wheelbase", 2.0))
        self.control_dt = float(getattr(cfg, "continuous_dt", 1.0))
        if not 0.0 <= self.min_speed < self.max_speed:
            raise ValueError("continuous speeds must satisfy 0 <= min_speed < max_speed")
        if not 0.0 < self.max_steering < np.pi / 2.0:
            raise ValueError("continuous_max_steering must be in (0, pi/2)")
        if self.wheelbase <= 0.0 or self.control_dt <= 0.0:
            raise ValueError("continuous_wheelbase and continuous_dt must be positive")
        self.action_space = spaces.Box(
            low=np.array([self.min_speed, -self.max_steering], dtype=np.float32),
            high=np.array([self.max_speed, self.max_steering], dtype=np.float32),
            dtype=np.float32,
        )

        self.global_steps = 0
        self.env_change_interval = 200000


        # obs_low = np.concatenate(
        #     (
        #         [0, -1000, -2 * np.pi, -2 * np.pi, 0],
        #         [0] * self.cfg.k_land_nearest,
        #         [-1000] * self.cfg.k_land_nearest,
        #         [-2 * np.pi] * self.cfg.k_land_nearest,
        #         [-2 * np.pi] * self.cfg.k_land_nearest,
        #     )
        # )
        # obs_high = np.concatenate(
        #     (
        #         [1000, 1000, 2 * np.pi, 2 * np.pi, 1000],
        #         [1000] * self.cfg.k_land_nearest,
        #         [1000] * self.cfg.k_land_nearest,
        #         [2 * np.pi] * self.cfg.k_land_nearest,
        #         [2 * np.pi] * self.cfg.k_land_nearest,
        #     )
        # )

        # Belief features: estimated distance/bearing/delta/progress,
        # localization uncertainty, contact history and obstacle clearance.
        obs_low = np.array(
            [
                0.0, -1.0, -1.0, -1.0, -1.0, -1.0, 0.0, 0.0,
                0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0, 0.0,
            ],
            dtype=np.float32,
        )
        obs_high = np.ones(16, dtype=np.float32)

        self.observation_space = spaces.Dict(
            {
                # Separate channels prevent the CNN from treating map labels as
                # ordered numeric values: obstacle, landmark and visible goal.
                "image": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(3, self.cfg.view_size, self.cfg.view_size),
                    dtype=np.float32,
                ),
                "info": spaces.Box(low=obs_low, high=obs_high, dtype=np.float32),
            }
        )

    def _apply_map_spec(self, map_index: int) -> None:
        """Load one immutable map specification into the live environment."""
        if not 0 <= map_index < len(self.map_specs):
            raise IndexError(f"map_index {map_index} is outside [0, {len(self.map_specs) - 1}]")
        spec = self.map_specs[map_index]
        self.current_map_index = map_index
        self.current_map_id = str(spec["id"])
        self.current_map_difficulty = str(spec["difficulty"])
        self.world_map = rasterize_map(spec, int(self.cfg.map_size))
        self.obs_list = [
            [int(item["x"]), int(item["y"]), int(item["size"])]
            for item in spec["obstacles"]
        ]
        self.land_list = [
            [int(item["x"]), int(item["y"]), int(item["size"])]
            for item in spec["landmarks"]
        ]
        self.start_x, self.start_y = (int(value) for value in spec["start"])
        self.goal_x, self.goal_y = (int(value) for value in spec["goal"])
        self.start_point = (self.start_x, self.start_y)
        self.goal_point = (self.goal_x, self.goal_y)
        self.robot_x, self.robot_y = self.start_point
        self.goal_region = [(self.goal_x, self.goal_y)]
        self.world_map[self.goal_y, self.goal_x] = 4
        self.world_map[self.robot_y, self.robot_x] = 3

        goal_heading = np.arctan2(
            self.start_y - self.goal_y,
            self.goal_x - self.start_x,
        )
        self.robot_heading = self.wrap_angle(goal_heading)

    def _select_episode_map(self, options: Optional[Dict]) -> None:
        """Sample train maps and iterate evaluation maps deterministically."""
        requested_index = None if options is None else options.get("map_index")
        if requested_index is not None:
            map_index = int(requested_index)
        elif self.map_split == "train":
            map_index = int(self.np_random.integers(len(self.map_specs)))
        else:
            map_index = (self.episodes - 1) % len(self.map_specs)
        self._apply_map_spec(map_index)

    def is_valid_point(self, x, y, margin=2):
        """Return True if (x,y) is safe for start/goal."""
        x, y = int(x), int(y)

        # Out of bounds
        if x < 0 or x >= self.world_map.shape[1]:
            return False
        if y < 0 or y >= self.world_map.shape[0]:
            return False

        # Check map content
        if self.world_map[y, x] in [1, 2]:   # 1=obstacle, 2=landmark
            return False

        # Extra safety bubble (avoid near obstacles)
        for ox, oy, size in self.obs_list:
            half = size / 2.0
            if abs(x - ox) <= half + margin and abs(y - oy) <= half + margin:
                return False

        for lx, ly, size in self.land_list:
            half = size / 2.0
            if abs(x - lx) <= half + margin and abs(y - ly) <= half + margin:
                return False

        return True

    def sample_safe_point(self, low, high):
        while True:
            x = self.np_random.integers(low, high)
            y = self.np_random.integers(low, high)
            if self.is_valid_point(x, y):
                return x, y


    def sample_boundary_points(self) -> Tuple[Tuple, Tuple]:
        # Define the adjusted edges by excluding -2 from all sides
        edges = {
            "top": [(x, 3) for x in range(3, self.world_map.shape[1] - 3)],
            "bottom": [(x, self.world_map.shape[0] - 3) for x in range(3, self.world_map.shape[1] - 3)],
            "left": [(3, y) for y in range(3, self.world_map.shape[0] - 3)],
            "right": [(self.world_map.shape[1] - 3, y) for y in range(3, self.world_map.shape[0] - 3)],
        }

        # Randomly choose an edge for the start
        start_edge_name = self.np_random.choice(list(edges.keys()))
        # Determine the opposite edge
        opposite_edges = {
            "top": "bottom",
            "bottom": "top",
            "left": "right",
            "right": "left",
        }
        goal_edge_name = opposite_edges[start_edge_name]

        # Sample valid start position
        while True:
            start_position = edges[start_edge_name][self.np_random.integers(len(edges[start_edge_name]))]
            if self.is_valid_point(*start_position):
                break

        # Sample valid goal position
        while True:
            goal_position = edges[goal_edge_name][self.np_random.integers(len(edges[goal_edge_name]))]
            if self.is_valid_point(*goal_position):
                break

        return start_position, goal_position, goal_edge_name

    def get_random_landmarks(self, num_landmark: int) -> List:
        land_list = []

        # LANDMARK #1
        # Randomly choose the center for the square obstacle
        center_x = self.np_random.integers(25 + self.cfg.obstacle_size // 2, 45 - self.cfg.obstacle_size // 2)
        center_y = self.np_random.integers(10 + self.cfg.landmark_size // 2, 40 - self.cfg.landmark_size // 2)

        # Calculate top-left corner based on center
        top_left_x = center_x - self.cfg.landmark_size // 2
        top_left_y = center_y - self.cfg.landmark_size // 2

        # Define the region for the landmark
        region = self.world_map[top_left_y : top_left_y + self.cfg.landmark_size, top_left_x : top_left_x + self.cfg.landmark_size]

        # # Check if the landmark overlaps with any existing ones and if the region is empty
        # if self.are_points_far_from_reference(land_list, [center_x, center_y], self.cfg.landmark_size // 2) and np.all(region == 0):

        # Add the landmark with center coordinates and size to land_list
        land_list.append([center_x, center_y, self.cfg.landmark_size])

        # Update the grid world only where it's 0
        region[region == 0] = 2

        # # LANDMARK #2
        # # Randomly choose the center for the square obstacle
        # center_x = np.random.randint(27 + self.cfg.landmark_size // 2, 45 - self.cfg.landmark_size // 2)
        # center_y = np.random.randint(5 + self.cfg.landmark_size // 2, 23 - self.cfg.landmark_size // 2)

        # # Calculate top-left corner based on center
        # top_left_x = center_x - self.cfg.landmark_size // 2
        # top_left_y = center_y - self.cfg.landmark_size // 2

        # # Define the region for the landmark
        # region = self.world_map[top_left_y : top_left_y + self.cfg.landmark_size, top_left_x : top_left_x + self.cfg.landmark_size]

        # # Check if the landmark overlaps with any existing ones and if the region is empty
        # if self.are_points_far_from_reference(land_list, [center_x, center_y], self.cfg.landmark_size) and np.all(region == 0):
        #     # Add the landmark with center coordinates and size to land_list
        #     land_list.append([center_x, center_y, self.cfg.landmark_size])

        #     # Update the grid world only where it's 0
        #     region[region == 0] = 2

        return land_list

    def get_random_obstacles(self, num_obstacle: int) -> List:
        obs_list = []

        # OBSTACLE #1
        # Randomly choose the center for the square obstacle
        center_x = self.np_random.integers(5 + self.cfg.obstacle_size // 2, 25 - self.cfg.obstacle_size // 2)
        center_y = self.np_random.integers(10 + self.cfg.obstacle_size // 2, 45 - self.cfg.obstacle_size // 2)

        # Calculate top-left corner based on center
        top_left_x = center_x - self.cfg.obstacle_size // 2
        top_left_y = center_y - self.cfg.obstacle_size // 2

        # Define the region for the obstacle
        region = self.world_map[top_left_y : top_left_y + self.cfg.obstacle_size, top_left_x : top_left_x + self.cfg.obstacle_size]

        # # Check if the obstacle overlaps with any existing ones and if the region is empty
        # if self.are_points_far_from_reference(obs_list, [center_x, center_y], self.cfg.obstacle_size // 2) and np.all(region == 0):

        # Add the obstacle with center coordinates and size to obs_list
        obs_list.append([center_x, center_y, self.cfg.obstacle_size])

        # Update the grid world only where it's 0
        region[region == 0] = 1

        # # OBSTACLE #2
        # # Randomly choose the center for the square obstacle
        # center_x = np.random.randint(27 + self.cfg.obstacle_size // 2, 45 - self.cfg.obstacle_size // 2)
        # center_y = np.random.randint(27 + self.cfg.obstacle_size // 2, 45 - self.cfg.obstacle_size // 2)

        # # Calculate top-left corner based on center
        # top_left_x = center_x - self.cfg.obstacle_size // 2
        # top_left_y = center_y - self.cfg.obstacle_size // 2

        # # Define the region for the obstacle
        # region = self.world_map[top_left_y : top_left_y + self.cfg.obstacle_size, top_left_x : top_left_x + self.cfg.obstacle_size]

        # # Check if the obstacle overlaps with any existing ones and if the region is empty
        # if self.are_points_far_from_reference(obs_list, [center_x, center_y], self.cfg.obstacle_size) and np.all(region == 0):
        #     # Add the obstacle with center coordinates and size to obs_list
        #     obs_list.append([center_x, center_y, self.cfg.obstacle_size])

        #     # Update the grid world only where it's 0
        #     region[region == 0] = 1

        return obs_list

    def are_points_far_from_reference(self, points: List, ref_point: List, min_distance: float) -> bool:
        """Check if point is far from reference point."""
        points = np.array(points)
        ref_point = np.array(ref_point)
        return all(np.linalg.norm(point[:2] - ref_point) >= min_distance * 2 for point in points)

    def generate_path_info(self) -> None:
        """Generate navigation features from the EKF belief, never truth."""
        belief_x, belief_y, belief_heading = self.x_state
        # Map rows grow downward, while positive mathematical angles point up.
        self.goal_angle = np.arctan2(belief_y - self.goal_y, self.goal_x - belief_x)
        self.delta_angle = self.goal_angle - belief_heading
        self.delta_angle = np.arctan2(np.sin(self.delta_angle), np.cos(self.delta_angle))

        self.dist = np.sqrt((belief_x - self.goal_x) ** 2 + (belief_y - self.goal_y) ** 2)

        self.landdist = np.array(
            [
                self.compute_distance([belief_x, belief_y], [land[0], land[1]])
                - land[2] / 2.0
                for land in self.land_list
            ]
        )
        self.landangle = np.array([np.arctan2(belief_y - land[1], land[0] - belief_x) for land in self.land_list])
        self.landangle = np.arctan2(np.sin(self.landangle), np.cos(self.landangle))

        self.obsdist = np.array(
            [
                self.compute_distance([belief_x, belief_y], [obs[0], obs[1]])
                - obs[2] / 2.0
                for obs in self.obs_list
            ]
        )
        self.obsangle = np.array([np.arctan2(belief_y - obs[1], obs[0] - belief_x) for obs in self.obs_list])
        self.obsangle = np.arctan2(np.sin(self.obsangle), np.cos(self.obsangle))

    def compute_distance(self, p1: List, p2: List) -> float:
        """Compute distance between points p1 and p2.

        Args:
            p1 (List): _description_
            p2 (List): _description_

        Returns:
            float: _description_
        """
        norm_squared = (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
        return np.sqrt(norm_squared)

    def covariance_metric(self, covariance: Optional[np.ndarray] = None) -> float:
        """Return a stable scalar measure of position uncertainty."""
        covariance = self.x_cov if covariance is None else covariance
        position_trace = max(float(np.trace(np.asarray(covariance)[:2, :2])), 0.0)
        return float(np.log1p(position_trace))

    def nearest_landmark_needing_information(self) -> Optional[int]:
        """Return the closest landmark whose information budget is not met."""
        information_budget = float(getattr(self.cfg, "information_budget_per_landmark", 3.5))
        candidates = np.flatnonzero(self.landmark_information_gain < information_budget)
        if candidates.size == 0:
            return None
        distances = [
            self.compute_distance(self.x_state[:2], self.land_list[index][:2])
            for index in candidates
        ]
        return int(candidates[int(np.argmin(distances))])

    def change_seed(self, seed: int) -> None:
        self.np_random = np.random.default_rng(seed)

    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        """Reset the vehicle and select a map from the configured split.

        Args:
            seed (int, optional): _description_. Defaults to None.
            options: May contain ``map_index`` for deterministic evaluation.

        Returns:
            Tuple[np.ndarray, Dict]: _description_
        """
        super().reset(seed=seed)
        self.episodes += 1

        # if self.global_steps > 0 and self.global_steps % self.env_change_interval == 0:
        #     print(f"\n=== Changing START/GOAL at {self.global_steps} steps ===\n")
        #     self.reset_env(
        #         num_obstacle=self.cfg.num_obstacle,
        #         num_landmark=self.cfg.num_landmark
        #     )
        if self.map_specs:
            self._select_episode_map(options)
        else:
            self.reset_env(self.cfg.num_obstacle, self.cfg.num_landmark)
        self.goal_reached = False

        self.land_visit = np.zeros(len(self.land_list))
        self.obs_visit = np.zeros(len(self.obs_list))

        self.robot_x, self.robot_y = self.start_x, self.start_y
        self.true_state = np.array(
            [self.robot_x, self.robot_y, self.robot_heading],
            dtype=float,
        )

        # True distance drives reward and termination. Estimated distance is
        # calculated from the belief and is the only distance shown to policy.
        self.distance_to_goal = np.sqrt((self.robot_x - self.goal_x) ** 2 + (self.robot_y - self.goal_y) ** 2)
        self.prev_distance_to_goal = self.distance_to_goal
        self.estimated_distance_to_goal = self.distance_to_goal
        self.prev_estimated_distance_to_goal = self.estimated_distance_to_goal
        self.last_progress = 0.0
        self.last_belief_progress = 0.0
        self.last_landmark_progress = 0.0
        self.last_information_gain = 0.0
        self.landmark_information_gain = np.zeros(len(self.land_list), dtype=float)
        self.visit_counts = {(int(self.robot_x), int(self.robot_y)): 1}
        self.wall_touch = False
        self.collision_streak = 0
        self.max_collision_streak = 0
        self.collision_aborted = False
        self.collision_failed = False
        self.reward_components = {}
        self.last_measurement = None
        self.last_innovation = None

        self.dist_dot = self.cfg.dist_dot
        self.yaw_dot = self.cfg.yaw_dot
        self.delta_angle_dot = self.cfg.delta_angle_dot

        self.steps = 0
        self.poses = []

        self.prev_cell_value = 0

        # EKF Parameters
        self.x_state = self.true_state.copy()
        self.x_cov = np.array(self.cfg.x_cov, dtype=float)


        # Find the maximum eigenvalue
        self.eigenvalues = np.linalg.eigvals(self.x_cov)
        self.determinant = np.linalg.det(self.x_cov)
        self.max_eigenvalue = np.max(self.eigenvalues)

        self.true_path = [self.true_state.copy()]
        self.estimated_path = [self.x_state.copy()]

        self.generate_path_info()
        info = {
            "map_id": self.current_map_id,
            "map_split": self.map_split,
            "map_difficulty": self.current_map_difficulty,
            "true_pose": self.true_state.copy(),
            "estimated_pose": self.x_state.copy(),
            "position_error": 0.0,
        }
        # self.create_plot()

        return self.get_states(), info

    def reset_env(self, num_obstacle=None, num_landmark=None):
        """
        Smooth curriculum schedule for start/goal placement.
        Obstacles & landmarks remain fixed.
        """

        # Restore the map value hidden under the old robot marker before
        # removing the old start/goal markers.
        if hasattr(self, "prev_cell_value"):
            old_pose = getattr(
                self,
                "true_state",
                np.array([self.robot_x, self.robot_y, self.robot_heading]),
            )
            old_x, old_y = self.pose_to_cell(old_pose)
            self.world_map[old_y, old_x] = self.prev_cell_value

        # Remove old robot & goal markers
        self.world_map[self.world_map == 3] = 0
        self.world_map[self.world_map == 4] = 0

        # ─────────────────────────────────────────────────────────────
        # 1. Curriculum phase selection
        # ─────────────────────────────────────────────────────────────

        ep = self.episodes

        # Phase 1: Very easy (short straight paths)

        if ep < 500:
            candidate_columns = [5, self.cfg.map_size - 5]
            clear_columns = [
                x
                for x in candidate_columns
                if np.all(self.world_map[5 : self.cfg.map_size - 4, x] != 1)
                and self.world_map[5, x] not in (1, 2)
                and self.world_map[self.cfg.map_size - 5, x] not in (1, 2)
            ]
            if not clear_columns:
                clear_columns = [
                    x
                    for x in range(3, self.cfg.map_size - 2)
                    if np.all(self.world_map[5 : self.cfg.map_size - 4, x] != 1)
                    and self.world_map[5, x] not in (1, 2)
                    and self.world_map[self.cfg.map_size - 5, x] not in (1, 2)
                ]
            start_x = int(self.np_random.choice(clear_columns))
            goal_x = start_x
            if self.np_random.random() < 0.5:
                start_y, goal_y = 5, self.cfg.map_size - 5
            else:
                start_y, goal_y = self.cfg.map_size - 5, 5

        elif ep < 2000:
            start_x, start_y = self.sample_safe_point(10, self.cfg.map_size - 10)
            while True:
                goal_x, goal_y = self.sample_safe_point(10, self.cfg.map_size - 10)
                if self.compute_distance((start_x, start_y), (goal_x, goal_y)) >= self.cfg.map_size / 3:
                    break

        else:
            while True:
                (sx, sy), (gx, gy), _ = self.sample_boundary_points()
                if self.is_valid_point(sx, sy) and self.is_valid_point(gx, gy):
                    start_x, start_y = sx, sy
                    goal_x, goal_y = gx, gy
                    break



        # Save points
        self.start_x, self.start_y = start_x, start_y
        self.goal_x,  self.goal_y  = goal_x, goal_y
        self.robot_x, self.robot_y = start_x, start_y

        # Set map markers
        self.world_map[self.robot_y, self.robot_x] = 3
        self.world_map[self.goal_y, self.goal_x]  = 4

        # Start facing the goal; subsequent headings evolve continuously.
        goal_heading = np.arctan2(self.start_y - self.goal_y, self.goal_x - self.start_x)
        self.robot_heading = self.wrap_angle(goal_heading)

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Advance hidden physics, update the EKF belief, and score the step."""
        first_contact_terminal = bool(getattr(self.cfg, "terminate_on_collision", False))
        if first_contact_terminal and self.collision_failed:
            raise RuntimeError("Collision ended the episode; reset before another action.")
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid continuous action {action}; expected [speed, steering] in "
                f"{self.action_space}"
            )

        self.steps += 1
        self.wall_touch = False
        self.localize = False
        old_true_distance = self.distance_to_goal
        old_estimated_distance = self.estimated_distance_to_goal
        old_uncertainty = self.covariance_metric()
        target_landmark = self.nearest_landmark_needing_information()
        if target_landmark is None:
            old_landmark_distance = 0.0
        else:
            old_landmark_distance = self.compute_distance(
                self.true_state[:2],
                self.land_list[target_landmark][:2],
            )
        self.last_landmark_progress = 0.0
        self.last_information_gain = 0.0
        self.last_measurement = None
        self.last_innovation = None

        old_true_state = self.true_state.copy()
        proposed_true_state = self.sample_true_motion(old_true_state, action)
        moved = self.check_step_feasibility(
            proposed_true_state,
            start_pose=old_true_state,
            control=action,
        )
        if moved:
            self.true_state = proposed_true_state
            self.collision_streak = 0
        else:
            # A contact sensor reveals that translation was blocked, but the
            # commanded turn and its physical heading disturbance still occur.
            self.wall_touch = True
            self.collision_streak += 1
            self.max_collision_streak = max(
                self.max_collision_streak,
                self.collision_streak,
            )
            if not first_contact_terminal:
                self.true_state[2] = proposed_true_state[2]
        self.collision_failed = bool(first_contact_terminal and not moved)

        self.robot_x, self.robot_y, self.robot_heading = self.true_state
        self.robot_heading = self.wrap_angle(self.robot_heading)
        self.true_state[2] = self.robot_heading

        # Landmark availability depends on hidden physical position. The
        # landmark provides a noisy absolute-position fix; it never reveals
        # the true pose directly to the policy.
        landmark_index = self.landmark_at_pose(self.true_state)
        self.localize = landmark_index is not None
        measurement = None
        if self.localize:
            measurement = self.true_state[:2] + self.np_random.multivariate_normal(
                np.zeros(2, dtype=float),
                self.R,
            )
            self.last_measurement = measurement.copy()

        self.x_state, self.x_cov = self.ekf_dubins_motion_model(
            self.x_state,
            self.x_cov,
            action,
            measurement=measurement,
            moved=moved,
        )
        self.x_state_ = self.x_state.copy()
        self.x_cov_ = self.x_cov.copy()

        if target_landmark is not None:
            new_landmark_distance = self.compute_distance(
                self.true_state[:2],
                self.land_list[target_landmark][:2],
            )
            self.last_landmark_progress = old_landmark_distance - new_landmark_distance

        if landmark_index is not None:
            raw_information_gain = max(0.0, old_uncertainty - self.covariance_metric())
            information_budget = float(getattr(self.cfg, "information_budget_per_landmark", 3.5))
            remaining_budget = max(
                0.0,
                information_budget - self.landmark_information_gain[landmark_index],
            )
            self.last_information_gain = min(raw_information_gain, remaining_budget)
            self.landmark_information_gain[landmark_index] += self.last_information_gain
            self.land_visit[landmark_index] = 1.0

        # Restore the semantic value underneath the old robot marker and move
        # the marker to the cell containing the hidden physical robot.
        old_cell = self.pose_to_cell(old_true_state)
        self.world_map[old_cell[1], old_cell[0]] = self.prev_cell_value
        new_cell = self.pose_to_cell(self.true_state)
        self.prev_cell_value = self.world_map[new_cell[1], new_cell[0]]
        self.world_map[new_cell[1], new_cell[0]] = 3

        previous_visits = self.visit_counts.get(new_cell, 0)
        self.visit_counts[new_cell] = previous_visits + 1
        self.poses.append([self.robot_x, self.robot_y])
        self.true_path.append(self.true_state.copy())
        self.estimated_path.append(self.x_state.copy())
        self.whether_inside_obstacle = False

        true_distance_to_goal = self.compute_distance(self.true_state[:2], self.goal_point)
        estimated_distance_to_goal = self.compute_distance(self.x_state[:2], self.goal_point)
        self.last_progress = old_true_distance - true_distance_to_goal
        self.last_belief_progress = old_estimated_distance - estimated_distance_to_goal
        self.prev_distance_to_goal = old_true_distance
        self.distance_to_goal = true_distance_to_goal
        self.prev_estimated_distance_to_goal = old_estimated_distance
        self.estimated_distance_to_goal = estimated_distance_to_goal

        # Task success is a property of the physical robot, never its belief.
        goal_success = bool(
            true_distance_to_goal <= self.cfg.goal_threshold and not self.collision_failed
        )
        terminated = bool(goal_success or self.collision_failed)
        collision_abort_steps = int(getattr(self.cfg, "collision_abort_steps", 0))
        # Optional training-only allowance: keep observation normalization and
        # evaluation trap detection tied to the original collision_abort_steps.
        training_abort_steps = int(
            getattr(self.cfg, "training_collision_abort_steps", collision_abort_steps)
        )
        self.collision_aborted = bool(self.collision_failed or (
            not self.test_mode
            and training_abort_steps > 0
            and self.collision_streak >= training_abort_steps
            and not terminated
        ))
        truncated = bool(
            not self.collision_failed
            and (self.steps >= self.cfg.max_steps or self.collision_aborted)
        )
        reward = self.get_rewards(action=action, previous_visits=previous_visits, terminated=goal_success, truncated=truncated)
        next_state = self.get_states()

        position_error = self.compute_distance(self.true_state[:2], self.x_state[:2])

        info = {
            "is_success": goal_success,
            "collision_failure": self.collision_failed,
            "map_id": self.current_map_id,
            "map_split": self.map_split,
            "map_difficulty": self.current_map_difficulty,
            "distance_to_goal": float(true_distance_to_goal),
            "true_distance_to_goal": float(true_distance_to_goal),
            "estimated_distance_to_goal": float(estimated_distance_to_goal),
            "estimated_goal_reached": bool(estimated_distance_to_goal <= self.cfg.goal_threshold),
            "true_pose": self.true_state.copy(),
            "estimated_pose": self.x_state.copy(),
            "position_error": float(position_error),
            "collision": self.wall_touch,
            "collision_streak": int(self.collision_streak),
            "max_collision_streak": int(self.max_collision_streak),
            "collision_aborted": self.collision_aborted,
            "collision_trap_detected": bool(
                collision_abort_steps > 0
                and self.max_collision_streak >= collision_abort_steps
            ),
            "obstacle_clearance": self.minimum_obstacle_clearance(self.true_state),
            "commanded_speed": float(action[0]),
            "commanded_steering": float(action[1]),
            "localized": self.localize,
            "measurement": None if self.last_measurement is None else self.last_measurement.copy(),
            "innovation_norm": (
                0.0 if self.last_innovation is None else float(np.linalg.norm(self.last_innovation))
            ),
            "covariance_trace": float(np.trace(self.x_cov[:2, :2])),
            "covariance_metric": self.covariance_metric(),
            "landmarks_visited": int(np.sum(self.land_visit)),
            "reward_components": self.reward_components.copy(),
        }

        self.global_steps += 1
        return next_state, reward, terminated, truncated, info

    @staticmethod
    def wrap_angle(angle: float) -> float:
        return float(np.arctan2(np.sin(angle), np.cos(angle)))

    def pose_to_cell(self, pose: np.ndarray) -> Tuple[int, int]:
        """Convert a continuous physical pose to its nearest grid cell."""
        x = int(np.rint(float(pose[0])))
        y = int(np.rint(float(pose[1])))
        return x, y

    def landmark_at_pose(self, pose: np.ndarray) -> Optional[int]:
        """Return the sensing region containing the hidden true pose."""
        x, y = float(pose[0]), float(pose[1])
        candidates = []
        for index, (landmark_x, landmark_y, size) in enumerate(self.land_list):
            half = size / 2.0
            if abs(x - landmark_x) <= half and abs(y - landmark_y) <= half:
                candidates.append(index)
        if not candidates:
            return None
        return min(
            candidates,
            key=lambda index: self.compute_distance((x, y), self.land_list[index][:2]),
        )

    def minimum_obstacle_clearance(self, pose: np.ndarray) -> float:
        """Return point-to-obstacle/boundary clearance in map cells.

        The value is equivalent to a short-range proximity sensor and is used
        only near geometry already visible in the egocentric semantic image.
        Obstacle cell footprints extend half a cell beyond their outer centres.
        """
        x, y = (float(value) for value in pose[:2])
        clearances = []
        for obstacle_x, obstacle_y, size in self.obs_list:
            half = int(size) // 2
            x_min = float(obstacle_x - half) - 0.5
            x_max = float(obstacle_x + half) + 0.5
            y_min = float(obstacle_y - half) - 0.5
            y_max = float(obstacle_y + half) + 0.5
            dx = max(x_min - x, 0.0, x - x_max)
            dy = max(y_min - y, 0.0, y - y_max)
            clearances.append(float(np.hypot(dx, dy)))

        # check_step_feasibility accepts rounded cell coordinates 2..N-3.
        lower_boundary = 1.5
        upper_x = float(self.world_map.shape[1]) - 2.5
        upper_y = float(self.world_map.shape[0]) - 2.5
        clearances.extend((
            x - lower_boundary,
            y - lower_boundary,
            upper_x - x,
            upper_y - y,
        ))
        return max(0.0, min(clearances))

    def sample_true_motion(self, state: np.ndarray, action: np.ndarray) -> np.ndarray:
        """Apply the commanded motion plus a hidden physical disturbance."""
        nominal_state = self.dubins_motion_model(state, action)
        disturbance = self.np_random.normal(0.0, self.motion_noise_std)
        disturbed_state = nominal_state + disturbance
        disturbed_state[2] = self.wrap_angle(disturbed_state[2])
        return disturbed_state

    def check_step_feasibility(
        self,
        next_pose: np.ndarray,
        start_pose: Optional[np.ndarray] = None,
        control: Optional[np.ndarray] = None,
    ) -> bool:
        """Check the complete noisy bicycle arc against bounds and obstacles."""
        start_pose = self.true_state if start_pose is None else start_pose
        distance = self.compute_distance(start_pose[:2], next_pose[:2])
        samples = max(2, int(np.ceil(distance / 0.1)) + 1)
        nominal_end = (
            self.dubins_motion_model(start_pose, control)
            if control is not None
            else None
        )
        for fraction in np.linspace(0.0, 1.0, samples):
            if control is None:
                point = start_pose[:2] + fraction * (next_pose[:2] - start_pose[:2])
            else:
                nominal_pose = self.dubins_motion_model(
                    start_pose,
                    control,
                    dt=self.control_dt * fraction,
                )
                # Treat the sampled endpoint process disturbance as accumulating
                # over the decision interval rather than teleporting at its end.
                point = (
                    nominal_pose[:2]
                    + fraction * (next_pose[:2] - nominal_end[:2])
                )
            x, y = self.pose_to_cell(np.array([point[0], point[1], 0.0]))
            if not (1 < x < self.world_map.shape[1] - 2 and 1 < y < self.world_map.shape[0] - 2):
                return False
            if self.world_map[y, x] == 1:
                return False
        return True

    def create_slice_grid_with_visibility(
        self, obstacles: np.ndarray, agent_position: List, slice_size: int = 10, grid_resolution: int = 10
    ) -> np.ndarray:
        grid = np.zeros((grid_resolution + 1, grid_resolution + 1), dtype=int)

        half_slice = slice_size / 2
        x_min = agent_position[0] - half_slice
        y_min = agent_position[1] - half_slice

        # Calculate cell size based on grid resolution
        cell_size = slice_size / grid_resolution

        # Mark obstacle cells as in the previous function
        for ox, oy, radius in obstacles:
            if x_min - radius <= ox <= x_min + slice_size + radius and y_min - radius <= oy <= y_min + slice_size + radius:
                x_start = max(0, int((ox - radius - x_min) / cell_size))
                x_end = min(grid_resolution - 1, int((ox + radius - x_min) / cell_size))
                y_start = max(0, int((oy - radius - y_min) / cell_size))
                y_end = min(grid_resolution - 1, int((oy + radius - y_min) / cell_size))

                for i in range(x_start, x_end + 1):
                    for j in range(y_start, y_end + 1):
                        cell_center_x = x_min + (i + 0.5) * cell_size
                        cell_center_y = y_min + (j + 0.5) * cell_size
                        distance_to_obstacle = np.sqrt((cell_center_x - ox) ** 2 + (cell_center_y - oy) ** 2)
                        if distance_to_obstacle <= radius:
                            grid[j, i] = 1  # Mark as obstacle

        # Check visibility for each cell from the agent's position
        for i in range(grid_resolution):
            for j in range(grid_resolution):
                if grid[j, i] == 1:
                    continue  # Skip cells already marked as obstacles

                cell_center_x = x_min + (i + 0.5) * cell_size
                cell_center_y = y_min + (j + 0.5) * cell_size

                # Calculate direction and distance from agent to the cell center
                dx, dy = cell_center_x - agent_position[0], cell_center_y - agent_position[1]
                distance = np.sqrt(dx**2 + dy**2)
                steps = int(distance / cell_size)

                # Step along the line from agent to cell center to check for obstacles
                obstructed = False
                for step in range(1, steps):
                    # Intermediate point along the line
                    px = agent_position[0] + step * (dx / steps)
                    py = agent_position[1] + step * (dy / steps)

                    # Check if any obstacle obstructs this point
                    for ox, oy, radius in obstacles:
                        if np.sqrt((px - ox) ** 2 + (py - oy) ** 2) <= radius:
                            obstructed = True
                            break
                    if obstructed:
                        grid[j, i] = 1  # Mark as obstructed
                        break
        return grid

    def get_local_view(self, robot_x: float, robot_y: float) -> np.ndarray:
        world_map = self.world_map
        half_size = self.cfg.view_size // 2

        robot_x, robot_y = int(np.rint(robot_x)), int(np.rint(robot_y))

        # Create a padded version of the map to handle boundaries
        padded_map = np.pad(world_map, pad_width=half_size, mode="constant", constant_values=1)
        padded_robot_x, padded_robot_y = robot_x + half_size, robot_y + half_size

        local_view = padded_map[
            padded_robot_y - half_size : padded_robot_y + half_size + 1,
            padded_robot_x - half_size : padded_robot_x + half_size + 1,
        ]

        # Set robot position in the local view as 3 (center)
        local_view[half_size, half_size] = 3

        return local_view

    def get_egocentric_view(self, robot_x: float, robot_y: float, heading: float) -> np.ndarray:
        """Sample a nearest-neighbour semantic view aligned with any heading.

        The centre is the physical sensor location. The top of the returned
        image is the robot's forward direction, so no global pose coordinates
        are exposed to the policy.
        """
        size = int(self.cfg.view_size)
        half = size // 2
        rows, columns = np.indices((size, size), dtype=float)
        forward = half - rows
        right = columns - half
        cos_heading = np.cos(heading)
        sin_heading = np.sin(heading)
        world_x = robot_x + forward * cos_heading + right * sin_heading
        world_y = robot_y - forward * sin_heading + right * cos_heading
        sample_x = np.rint(world_x).astype(int)
        sample_y = np.rint(world_y).astype(int)

        view = np.ones((size, size), dtype=self.world_map.dtype)
        valid = (
            (sample_x >= 0)
            & (sample_x < self.world_map.shape[1])
            & (sample_y >= 0)
            & (sample_y < self.world_map.shape[0])
        )
        view[valid] = self.world_map[sample_y[valid], sample_x[valid]]
        view[half, half] = 3
        return view

    def get_states(self) -> np.ndarray:
        """Return egocentric sensing plus features derived from the EKF belief."""
        self.generate_path_info()

        # The local semantic image represents an onboard exteroceptive sensor:
        # it is physically centred on the true robot but contains no global
        # coordinates. Global goal/landmark vectors below use only the belief.
        rotated_local_view = self.get_egocentric_view(
            float(self.robot_x),
            float(self.robot_y),
            float(self.robot_heading),
        )

        state_view = np.stack(
            (
                rotated_local_view == 1,
                rotated_local_view == 2,
                rotated_local_view == 4,
            ),
            axis=0,
        ).astype(np.float32)

        map_scale = float(self.cfg.map_size)
        diagonal = np.sqrt(2.0) * map_scale
        covariance_normalizer = float(getattr(self.cfg, "covariance_normalizer", 2000.0))
        normalized_uncertainty = self.covariance_metric() / np.log1p(covariance_normalizer)
        collision_abort_steps = max(
            1,
            int(getattr(self.cfg, "collision_abort_steps", 12)),
        )
        normalized_collision_streak = np.clip(
            self.collision_streak / collision_abort_steps,
            0.0,
            1.0,
        )
        clearance_margin = max(
            float(getattr(self.cfg, "obstacle_clearance_margin", 1.5)),
            np.finfo(float).eps,
        )
        normalized_clearance = np.clip(
            self.minimum_obstacle_clearance(self.true_state) / clearance_margin,
            0.0,
            1.0,
        )

        landmark_index = self.nearest_landmark_needing_information()
        if landmark_index is None:
            landmark_features = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        else:
            belief_x, belief_y, belief_heading = self.x_state
            landmark_x, landmark_y = self.land_list[landmark_index][:2]
            landmark_dx = landmark_x - belief_x
            landmark_dy = landmark_y - belief_y
            landmark_distance = np.hypot(landmark_dx, landmark_dy)
            landmark_angle = np.arctan2(belief_y - landmark_y, landmark_x - belief_x)
            relative_landmark_angle = np.arctan2(
                np.sin(landmark_angle - belief_heading),
                np.cos(landmark_angle - belief_heading),
            )
            landmark_features = [
                np.clip(landmark_distance / diagonal, 0.0, 1.0),
                np.sin(relative_landmark_angle),
                np.cos(relative_landmark_angle),
                np.clip(landmark_dx / map_scale, -1.0, 1.0),
                np.clip(landmark_dy / map_scale, -1.0, 1.0),
                1.0,
            ]

        info_vector = np.array(
            [
                np.clip(self.dist / diagonal, 0.0, 1.0),
                np.sin(self.delta_angle),
                np.cos(self.delta_angle),
                np.clip((self.goal_x - self.x_state[0]) / map_scale, -1.0, 1.0),
                np.clip((self.goal_y - self.x_state[1]) / map_scale, -1.0, 1.0),
                np.clip(self.last_belief_progress, -1.0, 1.0),
                np.clip(normalized_uncertainty, 0.0, 1.0),
                float(self.wall_touch),
                normalized_collision_streak,
                normalized_clearance,
                *landmark_features,
            ],
            dtype=np.float32,
        )

        self.state = {
            "image": state_view,
            "info": info_vector,
        }
        return self.state

    def get_rewards(
        self,
        action: np.ndarray,
        previous_visits: int,
        terminated: bool,
        truncated: bool,
    ) -> float:
        """Compute the two objectives and their legacy scalarization."""
        uncertainty = self.covariance_metric()
        normalized_steering = float(action[1]) / self.max_steering
        if self.wall_touch:
            collision_growth = float(getattr(self.cfg, "collision_penalty_growth", 0.5))
            collision_cap = float(
                getattr(self.cfg, "collision_penalty_max_multiplier", 5.0)
            )
            collision_multiplier = min(
                1.0 + collision_growth * max(0, self.collision_streak - 1),
                collision_cap,
            )
        else:
            collision_multiplier = 0.0

        clearance_margin = max(
            float(getattr(self.cfg, "obstacle_clearance_margin", 1.5)),
            np.finfo(float).eps,
        )
        clearance_deficit = np.clip(
            1.0 - self.minimum_obstacle_clearance(self.true_state) / clearance_margin,
            0.0,
            1.0,
        )
        components = {
            "shared_progress": float(
                getattr(self.cfg, "shared_progress_reward_scale", 0.5)
                * self.last_progress
            ),
            "shared_step": -float(getattr(self.cfg, "shared_step_penalty", 0.05)),
            "progress": float(
                (
                    float(getattr(self.cfg, "progress_reward_scale", 2.0))
                    - float(getattr(self.cfg, "shared_progress_reward_scale", 0.5))
                )
                * self.last_progress
            ),
            "step": -float(
                float(getattr(self.cfg, "step_penalty", 0.25))
                - float(getattr(self.cfg, "shared_step_penalty", 0.05))
            ),
            "turn": -float(getattr(self.cfg, "turn_penalty", 0.02)) * normalized_steering**2,
            "collision": -float(getattr(self.cfg, "collision_penalty", 10.0)) * collision_multiplier,
            "clearance": (
                -float(getattr(self.cfg, "obstacle_clearance_penalty", 1.0))
                * clearance_deficit**2
            ),
            "collision_abort": (
                -float(getattr(self.cfg, "collision_abort_penalty", 50.0))
                if self.collision_aborted
                else 0.0
            ),
            "revisit": -min(float(getattr(self.cfg, "revisit_penalty", 0.5)) * previous_visits, 2.0),
            "goal": float(getattr(self.cfg, "goal_reward", 100.0)) if terminated else 0.0,
            "timeout": -float(getattr(self.cfg, "timeout_penalty", 25.0)) if truncated and not terminated else 0.0,
            "uncertainty": -float(getattr(self.cfg, "uncertainty_penalty", 0.25)) * uncertainty,
            "information_gain": (
                float(getattr(self.cfg, "information_gain_reward_scale", 5.0))
                * self.last_information_gain
            ),
            "landmark_progress": (
                float(getattr(self.cfg, "landmark_progress_reward_scale", 0.5))
                * self.last_landmark_progress
            ),
            "terminal_uncertainty": (
                -float(getattr(self.cfg, "terminal_uncertainty_penalty", 5.0)) * uncertainty
                if terminated
                else 0.0
            ),
        }

        # Reaching the goal safely is a feasibility requirement, not a user
        # preference. Shared terminal signals plus mild progress/step shaping
        # prevent localization-heavy policies from abandoning the task.
        shared_task_objective = sum(
            components[name]
            for name in (
                "collision",
                "clearance",
                "collision_abort",
                "goal",
                "timeout",
                "shared_progress",
                "shared_step",
            )
        )
        navigation_objective = shared_task_objective + sum(
            components[name]
            for name in ("progress", "step", "turn", "revisit")
        )
        localization_objective = shared_task_objective + sum(
            components[name]
            for name in ("uncertainty", "information_gain", "landmark_progress", "terminal_uncertainty")
        )
        navigation_weight = float(getattr(self.cfg, "navigation_objective_weight", 1.0))
        localization_weight = float(getattr(self.cfg, "localization_objective_weight", 1.0))
        reward = navigation_weight * navigation_objective + localization_weight * localization_objective
        components["navigation_objective"] = float(navigation_objective)
        components["localization_objective"] = float(localization_objective)
        components["total"] = float(reward)
        self.reward_components = components
        return float(reward)


    def draw_confidence_ellipse(self, global_view: np.ndarray, confidence: int = 0.95) -> np.ndarray:
        scale = np.sqrt(chi2.ppf(confidence, df=2))
        cov_matrix = np.array(self.x_cov)
        vals, vecs = np.linalg.eigh(cov_matrix[:2, :2])
        order = vals.argsort()[::-1]
        vals = vals[order]
        vecs = vecs[:, order]

        a = scale * np.sqrt(vals[0])
        b = scale * np.sqrt(vals[1])
        theta = np.arctan2(vecs[1, 0], vecs[0, 0])

        # Compute bounding box
        pad = int(np.ceil(max(a, b))) + 2
        belief_x, belief_y = self.x_state[:2]
        y_min = int(max(0, belief_y - pad))
        y_max = int(min(global_view.shape[0], belief_y + pad + 1))
        x_min = int(max(0, belief_x - pad))
        x_max = int(min(global_view.shape[1], belief_x + pad + 1))

        # Local meshgrid
        y, x = np.meshgrid(np.arange(y_min, y_max), np.arange(x_min, x_max), indexing="ij")
        x_shift = x - belief_x
        y_shift = y - belief_y

        # Rotate
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)
        x_rot = cos_t * x_shift + sin_t * y_shift
        y_rot = -sin_t * x_shift + cos_t * y_shift

        # Ellipse mask in the local patch
        mask = (x_rot**2 / a**2) + (y_rot**2 / b**2) <= 1.0

        # Only update where view == 0
        sub_view = global_view[y_min:y_max, x_min:x_max]
        sub_view[(mask) & (sub_view == 0)] = 5
        global_view[y_min:y_max, x_min:x_max] = sub_view

        return global_view

    def dubins_motion_model(
        self,
        state: np.ndarray,
        control: np.ndarray,
        dt: Optional[float] = None,
    ) -> np.ndarray:
        """Integrate one bounded forward-only bicycle-model control."""
        x, y, theta = state
        speed, steering = (float(value) for value in control)
        integration_time = self.control_dt if dt is None else float(dt)
        distance = speed * integration_time
        heading_change = distance * np.tan(steering) / self.wheelbase
        midpoint_heading = theta + 0.5 * heading_change
        x_new = x + distance * np.cos(midpoint_heading)
        # Image/grid rows increase downwards, opposite mathematical y.
        y_new = y - distance * np.sin(midpoint_heading)
        theta_new = self.wrap_angle(theta + heading_change)

        return np.array([x_new, y_new, theta_new], dtype=float)

    def motion_jacobian(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Jacobian computation.

        Args:
            state (np.ndarray): _description_
            control (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        _, _, theta = state
        speed, steering = (float(value) for value in control)
        distance = speed * self.control_dt
        heading_change = distance * np.tan(steering) / self.wheelbase
        midpoint_heading = theta + 0.5 * heading_change
        J = np.array(  # noqa: N806
            [
                [1, 0, -distance * np.sin(midpoint_heading)],
                [0, 1, -distance * np.cos(midpoint_heading)],
                [0, 0, 1],
            ],
            dtype=float,
        )  # noqa: N806
        return J

    def control_jacobian(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        """Jacobian computation for control u.

        Args:
            state (np.ndarray): _description_
            control (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        _, _, theta = state
        v, gamma = (float(value) for value in control)
        dt = self.control_dt
        sec_squared = 1.0 / np.cos(gamma) ** 2
        dtheta_dv = dt * np.tan(gamma) / self.wheelbase
        dtheta_dgamma = v * dt * sec_squared / self.wheelbase
        heading_change = v * dt * np.tan(gamma) / self.wheelbase
        midpoint_heading = theta + 0.5 * heading_change
        dmid_dv = 0.5 * dtheta_dv
        dmid_dgamma = 0.5 * dtheta_dgamma
        J = np.array(  # noqa: N806
            [
                [
                    dt * np.cos(midpoint_heading) - v * dt * np.sin(midpoint_heading) * dmid_dv,
                    -v * dt * np.sin(midpoint_heading) * dmid_dgamma,
                ],
                [
                    -dt * np.sin(midpoint_heading) - v * dt * np.cos(midpoint_heading) * dmid_dv,
                    -v * dt * np.cos(midpoint_heading) * dmid_dgamma,
                ],
                [dtheta_dv, dtheta_dgamma],
            ]
        )  # noqa: N806
        return J

    def measurement_jacobian(self, state: np.ndarray) -> np.ndarray:
        """Measurement calculation.

        Args:
            state (np.ndarray): _description_

        Returns:
            np.ndarray: _description_
        """
        H = np.array([[1, 0, 0], [0, 1, 0]])  # noqa: N806
        return H

    def ekf_dubins_motion_model(
        self,
        initial_state: np.ndarray,
        initial_covariance: np.ndarray,
        controls: np.ndarray,
        measurement: Optional[np.ndarray] = None,
        moved: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Predict the belief and correct it with a noisy landmark position fix."""
        if moved:
            jac_x = self.motion_jacobian(initial_state, controls)
            state = self.dubins_motion_model(initial_state, controls)
        else:
            state = np.asarray(initial_state, dtype=float).copy()
            speed, steering = (float(value) for value in controls)
            if not bool(getattr(self.cfg, "terminate_on_collision", False)):
                state[2] = self.wrap_angle(
                    state[2]
                    + speed * self.control_dt * np.tan(steering) / self.wheelbase
                )
            jac_x = np.eye(3, dtype=float)

        covariance = jac_x @ initial_covariance @ jac_x.T + self.process_noise_cov
        covariance = 0.5 * (covariance + covariance.T)

        self.whether_inside_landmark = measurement is not None
        if measurement is not None:
            H = self.measurement_jacobian(state)  # noqa: N806
            innovation = np.asarray(measurement, dtype=float) - H @ state
            innovation_covariance = H @ covariance @ H.T + self.R
            K = covariance @ H.T @ np.linalg.inv(innovation_covariance)  # noqa: N806
            state = state + K @ innovation
            identity_minus_kh = np.eye(3) - K @ H
            # Joseph form is numerically stable and preserves PSD covariance.
            covariance = (
                identity_minus_kh @ covariance @ identity_minus_kh.T
                + K @ self.R @ K.T
            )
            self.last_innovation = innovation.copy()

        state[2] = self.wrap_angle(state[2])
        covariance = 0.5 * (covariance + covariance.T)
        return state, covariance

    def render_(self) -> None:
        """Render function."""
        if self.render_init:
            self.render_init = 0

            for ax_single in self.ax:
                ax_single.set_aspect("equal")
                ax_single.set_xlim([self.cfg.length_min, self.cfg.length_max])
                ax_single.set_ylim([self.cfg.breadth_min, self.cfg.breadth_max])

            circle = plt.Circle((self.goal_x, self.goal_y), self.cfg.goal_threshold, color="black", fill=True, alpha=0.2)
            self.ax[0].scatter(self.goal_x, self.goal_y, s=1, c="black")
            self.ax[0].add_patch(circle)

            for land in self.land_list:
                x_center, y_center, size = land
                circle = plt.Circle((x_center, y_center), size / 2.0, color="blue", fill=True, alpha=0.2)
                self.ax[0].scatter(self.goal_x, self.goal_y, s=1, c="black")
                self.ax[0].add_patch(circle)

            for obs in self.obs_list:
                x_center, y_center, size = obs
                circle = plt.Circle((x_center, y_center), size / 2.0, color="brown", fill=True, alpha=0.2)
                self.ax[0].scatter(self.goal_x, self.goal_y, s=1, c="black")
                self.ax[0].add_patch(circle)

            self.vector = self.ax[0].quiver(
                self.robot_x,
                self.robot_y,
                math.cos(self.robot_heading),
                math.sin(self.robot_heading),
                color="red",
                linewidths=0.2,
            )
            self.ax[0].add_artist(self.vector)

            # Update the mean and covariance for the Gaussian
            mean = self.x_state[:2]
            covariance = self.x_cov[:2, :2]  # Take the first 2x2 elements of the covariance matrix

            # Generate points for the 2D Gaussian
            x, y = np.mgrid[
                self.cfg.length_min : self.cfg.length_max : 0.1,
                self.cfg.breadth_min : self.cfg.breadth_max : 0.1,
            ]
            pos = np.dstack((x, y))
            rv = multivariate_normal(mean=mean, cov=covariance)

            # Plot the 2D Gaussian
            self.ax[1].contourf(x, y, rv.pdf(pos))

        else:
            self.vector.set_offsets(np.array([self.robot_x, self.robot_y]))
            u = 0.1 * np.cos(self.robot_heading)
            v = 0.1 * np.sin(self.robot_heading)
            self.vector.set_UVC(u, v)

            # Draw the true mean plot
            self.ax[0].plot(
                [tp[0] for tp in self.true_path],
                [tp[1] for tp in self.true_path],
                "b-",
                linewidth=0.5,
                label="True Path",
            )

            # Draw the estimated mean plot
            self.ax[0].plot(
                [ep[0] for ep in self.estimated_path],
                [ep[1] for ep in self.estimated_path],
                "r-",
                linewidth=0.5,
                label="Estimated Path",
            )

            # Update the mean and covariance for the Gaussian
            mean = self.x_state[:2]
            covariance = self.x_cov[:2, :2]  # Take the first 2x2 elements of the covariance matrix

            # Generate points for the 2D Gaussian
            x, y = np.mgrid[
                self.cfg.length_min : self.cfg.length_max : 0.5,
                self.cfg.breadth_min : self.cfg.breadth_max : 0.5,
            ]
            pos = np.dstack((x, y))
            rv = multivariate_normal(mean=mean, cov=covariance)

            # Update the mean and covariance for the Gaussian
            mean = self.x_state[:2]
            covariance = self.x_cov[:2, :2]  # Take the first 2x2 elements of the covariance matrix

            # Generate points for the 2D Gaussian
            x, y = np.mgrid[
                self.cfg.length_min : self.cfg.length_max : 0.1,
                self.cfg.breadth_min : self.cfg.breadth_max : 0.1,
            ]
            pos = np.dstack((x, y))
            rv = multivariate_normal(mean=mean, cov=covariance)

            # Plot the 2D Gaussian
            self.ax[1].contourf(x, y, rv.pdf(pos))

            plt.pause(0.01)
            plt.show()
            # plt.savefig(f"maps/{self.counter}.png")
            self.fig.canvas.mpl_connect("close_event", onclose)

    def create_plot(self, filename: str, title_text: Optional[str] = None) -> None:
        """Create plot.

        Args:
            filename (str): _description_
            title_text (Optional[str], optional): _description_. Defaults to None.
        """
        fig, ax = plt.subplots(1, 1, figsize=(8, 8))

        circle = plt.Circle((self.goal_x, self.goal_y), self.cfg.goal_threshold, color="black", fill=True, alpha=0.2)
        ax.scatter(self.goal_x, self.goal_y, s=1, c="black")
        ax.add_patch(circle)

        for land in self.land_list:
            x_center, y_center, size = land
            circle = plt.Circle((x_center, y_center), size / 2.0, color="blue", fill=True, alpha=0.2)
            ax.scatter(x_center, y_center, s=1, c="black")
            ax.add_patch(circle)

        for obs in self.obs_list:
            x_center, y_center, size = obs
            circle = plt.Circle((x_center, y_center), size / 2.0, color="brown", fill=True, alpha=0.2)
            ax.scatter(x_center, y_center, s=1, c="black")
            ax.add_patch(circle)

        vector = ax.quiver(
            self.robot_x,
            self.robot_y,
            math.cos(self.robot_heading),
            math.sin(self.robot_heading),
            color="red",
            linewidths=0.2,
        )
        ax.add_artist(vector)

        # Plot true path and estimated path
        true_path_x = [point[0] for point in self.true_path]
        true_path_y = [point[1] for point in self.true_path]
        estimated_path_x = [point[0] for point in self.estimated_path]
        estimated_path_y = [point[1] for point in self.estimated_path]

        ax.plot(true_path_x, true_path_y, "b-", linewidth=0.5, label="True Path")
        ax.plot(estimated_path_x, estimated_path_y, "r-", linewidth=0.5, label="Estimated Path")

        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlim([self.cfg.length_min, self.cfg.length_max])
        ax.set_ylim([self.cfg.length_min, self.cfg.length_max])
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_title(f"{title_text}")

        plt.savefig(f"{filename}.png")  # Save the plot as an image
        self.counter += 1
        plt.close()


def generate_world_map_dataset(cfg: DictConfig, num_maps: int = 15, save_dir: str = "generated_maps") -> None:
    """Generate and save a dataset of different world map images.

    Args:
        cfg (DictConfig): Configuration for the environment.
        num_maps (int): Number of different world maps to generate.
        save_dir (str): Directory where images will be saved.
    """
    os.makedirs(save_dir, exist_ok=True)

    for i in range(num_maps):
        env = PathPlan(cfg)

        # Optional: Set a different seed if you want different random worlds
        env.change_seed(seed=30)

        # env.reset_env(num_obstacle=cfg.num_obstacle, num_landmark=cfg.num_landmark)
        env.reset()

        # Create RGB image from the world map
        world_map = env.world_map

        color_map = {
            0: (255, 255, 255),  # Free space
            1: (0, 0, 0),  # Obstacle
            2: (173, 216, 230),  # Landmark
            3: (0, 255, 0),  # Robot
            4: (255, 0, 0),  # Goal
        }

        rgb_image = np.zeros((world_map.shape[0], world_map.shape[1], 3), dtype=np.uint8)
        for key, color in color_map.items():
            rgb_image[world_map == key] = color

        img = Image.fromarray(rgb_image, "RGB")
        img.save(os.path.join(save_dir, f"world_map_{i+1:02d}.png"))

    print(f"{num_maps} world maps saved in '{save_dir}/'.")


def save_map_as_image(map_array: np.ndarray, filename: str, color_map: Dict) -> None:
    rgb_image = np.zeros((map_array.shape[0], map_array.shape[1], 3), dtype=np.uint8)
    for key, color in color_map.items():
        rgb_image[map_array == key] = color
    img = Image.fromarray(rgb_image, "RGB")
    img.save(filename)
    print(f"Saved: {filename}")


@hydra.main(config_path="/media/storage/manav/LocalizationRL/src/cfg", config_name="environment", version_base="1.1")
def generate_dataset(cfg: DictConfig) -> None:
    generate_world_map_dataset(cfg, num_maps=15, save_dir="generated_maps")


@hydra.main(config_path="/media/storage/manav/LocalizationRL/src/cfg", config_name="environment", version_base="1.1")
def main(cfg: DictConfig) -> None:
    env = PathPlan(cfg)

    # Convert world_map to RGB image
    color_map = {
        0: (255, 255, 255),  # Free space -> White
        1: (0, 0, 0),  # Obstacle -> Black
        2: (173, 216, 230),  # Lightblue (Reserved for other elements)
        3: (0, 255, 0),  # Robot's current position -> Green
        4: (255, 0, 0),  # Goal position -> Blue
    }

    world_map = env.world_map
    state, _ = env.reset()
    env.robot_x = 45
    env.robot_y = 45
    env.robot_heading = np.deg2rad(0)

    # action = 1
    # next_state, reward, terminated, truncated, info = env.step(action)

    # for _ in range(500):
    #     action = env.action_space.sample()  # Use deterministic policy for testing
    #     obs, reward, terminated, truncated, info = env.step(action)

    # local_view = env.local_view
    # save_map_as_image(local_view, "local_view.png", color_map)

    # global_view = env.global_view
    # save_map_as_image(global_view, "global_view.png", color_map)

    # Create an RGB image array from the world_map
    rgb_image = np.zeros((world_map.shape[0], world_map.shape[1], 3), dtype=np.uint8)
    for key, color in color_map.items():
        rgb_image[world_map == key] = color

    # Convert to PIL Image and save
    img = Image.fromarray(rgb_image, "RGB")
    img.save("world_map_visualization.png")
    print("World map saved as 'world_map_visualization.png'.")


if __name__ == "__main__":
    # generate_dataset()
    main()
