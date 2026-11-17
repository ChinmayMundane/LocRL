"""Expected-belief Dubins RRT* for landmark-aided navigation.

The planner carries one Gaussian covariance at every tree node.  Candidate
Dubins edges propagate the nominal EKF covariance and apply an expected
position-measurement update when the nominal route enters a landmark region.
It is an expected-belief planner, not an observation-branching POMDP solver.
"""

from __future__ import annotations

import math

import numpy as np

from planners.dubins import connection
from planners.rrt_star import RRTStar


class ExpectedBeliefRRTStar(RRTStar):
    """Finite-budget Dubins RRT* with path-dependent EKF covariance."""

    def __init__(
        self,
        world,
        cfg,
        *,
        seed: int = 11,
        iterations: int = 600,
        navigation_weight: float = 0.5,
        localization_weight: float = 0.5,
        uncertainty_scale: float = 2.0,
        extension: float = 6.0,
        spacing: float = 0.2,
        margin: float = 0.3,
    ) -> None:
        radius = float(cfg.continuous_wheelbase) / math.tan(
            float(cfg.continuous_max_steering)
        )
        super().__init__(
            world,
            seed=seed,
            iterations=iterations,
            dubins=True,
            radius=radius,
            extension=extension,
            spacing=spacing,
            margin=margin,
        )
        if navigation_weight < 0.0 or localization_weight < 0.0:
            raise ValueError("belief-planner weights must be non-negative")
        total = float(navigation_weight + localization_weight)
        if total <= 0.0:
            raise ValueError("at least one belief-planner weight must be positive")
        if uncertainty_scale < 0.0:
            raise ValueError("uncertainty_scale must be non-negative")

        self.navigation_weight = float(navigation_weight) / total
        self.localization_weight = float(localization_weight) / total
        self.uncertainty_scale = float(uncertainty_scale)
        motion_std = np.asarray(cfg.motion_noise_std, dtype=float)
        self.process_covariance = np.diag(np.square(motion_std))
        self.measurement_covariance = np.asarray(cfg.noise_cov, dtype=float)
        self.measurement_matrix = np.array(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float
        )
        self.nominal_step_distance = float(cfg.continuous_max_speed) * float(
            cfg.continuous_dt
        )
        if self.nominal_step_distance <= 0.0:
            raise ValueError("nominal belief-planning step must be positive")
        self.default_start_covariance = self.environment_to_planner_covariance(
            np.asarray(cfg.x_cov, dtype=float)
        )
        self.current_start_covariance = self.default_start_covariance.copy()
        self.last_goal_covariance = None
        self.last_goal_cost = math.inf

    @staticmethod
    def environment_to_planner_covariance(covariance: np.ndarray) -> np.ndarray:
        """Reflect the environment y axis into the planner Cartesian frame."""
        covariance = np.asarray(covariance, dtype=float)
        if covariance.shape != (3, 3):
            raise ValueError("pose covariance must have shape (3, 3)")
        reflection = np.diag([1.0, -1.0, 1.0])
        return reflection @ covariance @ reflection.T

    def set_start_covariance(self, covariance: np.ndarray) -> None:
        self.current_start_covariance = self.environment_to_planner_covariance(
            covariance
        )

    @staticmethod
    def _symmetrize(covariance: np.ndarray) -> np.ndarray:
        covariance = 0.5 * (covariance + covariance.T)
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        eigenvalues = np.maximum(eigenvalues, 0.0)
        return (eigenvectors * eigenvalues) @ eigenvectors.T

    def _inside_landmark(self, pose: np.ndarray) -> bool:
        cell = self.cells(np.asarray(pose, dtype=float)[None, :])[0]
        x, y = (int(cell[0]), int(cell[1]))
        if x < 0 or y < 0 or x >= self.world.shape[1] or y >= self.world.shape[0]:
            return False
        return bool(self.world[y, x] == 2)

    def _measurement_update(self, covariance: np.ndarray) -> np.ndarray:
        h = self.measurement_matrix
        innovation_covariance = h @ covariance @ h.T + self.measurement_covariance
        gain = covariance @ h.T @ np.linalg.inv(innovation_covariance)
        identity_minus_gain_h = np.eye(3) - gain @ h
        updated = (
            identity_minus_gain_h @ covariance @ identity_minus_gain_h.T
            + gain @ self.measurement_covariance @ gain.T
        )
        return self._symmetrize(updated)

    @staticmethod
    def _uncertainty(covariance: np.ndarray) -> float:
        return float(np.log1p(max(float(np.trace(covariance[:2, :2])), 0.0)))

    def _decision_samples(self, path: np.ndarray) -> np.ndarray:
        """Resample a geometric edge at one nominal control interval."""
        path = np.asarray(path, dtype=float)
        segment_lengths = np.linalg.norm(np.diff(path[:, :2], axis=0), axis=1)
        cumulative = np.r_[0.0, np.cumsum(segment_lengths)]
        total = float(cumulative[-1])
        if total <= 1e-12:
            return path[[0, -1]].copy()
        distances = np.arange(0.0, total, self.nominal_step_distance)
        if distances.size == 0 or total - distances[-1] > 1e-10:
            distances = np.r_[distances, total]
        elif not np.isclose(distances[-1], total):
            distances = np.r_[distances, total]
        yaw = np.unwrap(path[:, 2])
        samples = np.column_stack(
            [
                np.interp(distances, cumulative, path[:, 0]),
                np.interp(distances, cumulative, path[:, 1]),
                np.interp(distances, cumulative, yaw),
            ]
        )
        return samples

    def propagate_path(
        self, path: np.ndarray, start_covariance: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """Return terminal covariance and integrated log-trace uncertainty."""
        samples = self._decision_samples(path)
        covariance = self._symmetrize(np.asarray(start_covariance, dtype=float))
        uncertainty_integral = 0.0
        previous_uncertainty = self._uncertainty(covariance)
        for start, end in zip(samples[:-1], samples[1:]):
            displacement = end[:2] - start[:2]
            distance = float(np.linalg.norm(displacement))
            if distance <= 1e-12:
                continue
            midpoint_heading = math.atan2(displacement[1], displacement[0])
            jacobian = np.array(
                [
                    [1.0, 0.0, -distance * math.sin(midpoint_heading)],
                    [0.0, 1.0, distance * math.cos(midpoint_heading)],
                    [0.0, 0.0, 1.0],
                ],
                dtype=float,
            )
            noise_fraction = distance / self.nominal_step_distance
            covariance = (
                jacobian @ covariance @ jacobian.T
                + noise_fraction * self.process_covariance
            )
            covariance = self._symmetrize(covariance)
            if self._inside_landmark(end):
                covariance = self._measurement_update(covariance)
            current_uncertainty = self._uncertainty(covariance)
            uncertainty_integral += (
                0.5 * (previous_uncertainty + current_uncertainty) * noise_fraction
            )
            previous_uncertainty = current_uncertainty
        return covariance, float(uncertainty_integral)

    def belief_edge(
        self, start: np.ndarray, goal: np.ndarray, start_covariance: np.ndarray
    ) -> tuple[np.ndarray | None, float, np.ndarray | None]:
        """Evaluate one collision-free Dubins edge in expected belief space."""
        path, length = connection(start, goal, self.radius, self.spacing)
        if not self.free(path):
            return None, math.inf, None
        end_covariance, uncertainty_integral = self.propagate_path(
            path, start_covariance
        )
        cost = (
            self.navigation_weight * float(length)
            + self.localization_weight * self.uncertainty_scale * uncertainty_integral
        )
        return path, float(cost), end_covariance

    def _lower_bound(self, start: np.ndarray, goal: np.ndarray) -> float:
        return self.navigation_weight * float(np.linalg.norm(start[:2] - goal[:2]))

    def _recompute_subtree(self, root: int) -> None:
        stack = [root]
        while stack:
            parent = stack.pop()
            children = [
                index
                for index, candidate_parent in enumerate(self.parents)
                if candidate_parent == parent
            ]
            for child in children:
                path, edge_cost, covariance = self.belief_edge(
                    self.nodes[parent], self.nodes[child], self.covariances[parent]
                )
                if path is None or covariance is None:
                    raise RuntimeError("rewired belief-tree edge became infeasible")
                self.edges[child] = path
                self.costs[child] = self.costs[parent] + edge_cost
                self.covariances[child] = covariance
                stack.append(child)

    def plan(
        self,
        start: np.ndarray,
        goal: np.ndarray,
        start_covariance: np.ndarray | None = None,
    ) -> np.ndarray | None:
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        root_covariance = (
            self.current_start_covariance
            if start_covariance is None
            else np.asarray(start_covariance, dtype=float)
        )
        self.nodes = [start.copy()]
        self.parents = [-1]
        self.costs = [0.0]
        self.edges = [None]
        self.covariances = [self._symmetrize(root_covariance.copy())]
        height, width = self.world.shape

        for _ in range(self.iterations):
            sample = (
                goal.copy()
                if self.rng.random() < 0.15
                else np.array(
                    [
                        self.rng.uniform(1, width - 2),
                        -self.rng.uniform(1, height - 2),
                        self.rng.uniform(-math.pi, math.pi),
                    ]
                )
            )
            nodes = np.asarray(self.nodes)
            distances = np.linalg.norm(nodes[:, :2] - sample[:2], axis=1)
            nearest = int(np.argmin(distances))
            nearest_node = nodes[nearest]
            candidate_path, _ = connection(
                nearest_node, sample, self.radius, self.spacing
            )
            cumulative = np.r_[
                0.0,
                np.cumsum(
                    np.linalg.norm(np.diff(candidate_path[:, :2], axis=0), axis=1)
                ),
            ]
            new = candidate_path[
                min(
                    np.searchsorted(cumulative, self.extension), len(candidate_path) - 1
                )
            ].copy()
            if np.linalg.norm(new[:2] - nearest_node[:2]) < 1e-5:
                continue

            initial, edge_cost, end_covariance = self.belief_edge(
                nearest_node, new, self.covariances[nearest]
            )
            if initial is None or end_covariance is None:
                continue
            near = np.flatnonzero(
                np.linalg.norm(nodes[:, :2] - new[:2], axis=1) <= self.extension * 1.5
            )
            parent = nearest
            best = self.costs[nearest] + edge_cost
            edge = initial
            best_covariance = end_covariance
            for candidate in near:
                candidate = int(candidate)
                if (
                    self.costs[candidate] + self._lower_bound(nodes[candidate], new)
                    >= best
                ):
                    continue
                path, candidate_cost, covariance = self.belief_edge(
                    nodes[candidate], new, self.covariances[candidate]
                )
                if path is not None and self.costs[candidate] + candidate_cost < best:
                    parent = candidate
                    best = self.costs[candidate] + candidate_cost
                    edge = path
                    best_covariance = covariance

            index = len(self.nodes)
            self.nodes.append(new)
            self.parents.append(parent)
            self.costs.append(best)
            self.edges.append(edge)
            self.covariances.append(best_covariance)

            ancestors = set()
            ancestor = parent
            while ancestor >= 0:
                ancestors.add(ancestor)
                ancestor = self.parents[ancestor]
            for candidate in near:
                candidate = int(candidate)
                if candidate in ancestors:
                    continue
                if (
                    best + self._lower_bound(new, nodes[candidate])
                    >= self.costs[candidate]
                ):
                    continue
                path, candidate_cost, covariance = self.belief_edge(
                    new, nodes[candidate], best_covariance
                )
                if path is None or covariance is None:
                    continue
                if best + candidate_cost + 1e-9 < self.costs[candidate]:
                    self.parents[candidate] = index
                    self.edges[candidate] = path
                    self.costs[candidate] = best + candidate_cost
                    self.covariances[candidate] = covariance
                    self._recompute_subtree(candidate)

        best = math.inf
        endpoint = None
        last = None
        last_covariance = None
        for index, node in enumerate(self.nodes):
            if self.costs[index] + self._lower_bound(node, goal) >= best:
                continue
            for heading in np.linspace(-math.pi, math.pi, 8, endpoint=False):
                target = goal.copy()
                target[2] = heading
                path, edge_cost, covariance = self.belief_edge(
                    node, target, self.covariances[index]
                )
                if path is not None and self.costs[index] + edge_cost < best:
                    best = self.costs[index] + edge_cost
                    endpoint = index
                    last = path
                    last_covariance = covariance
        if endpoint is None:
            self.last_goal_covariance = None
            self.last_goal_cost = math.inf
            return None

        parts = [last]
        index = endpoint
        while self.parents[index] >= 0:
            parts.append(self.edges[index])
            index = self.parents[index]
        self.last_goal_covariance = last_covariance
        self.last_goal_cost = float(best)
        return np.concatenate(list(reversed(parts)))
