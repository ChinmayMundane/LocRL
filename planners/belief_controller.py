"""Local-observation execution controller for expected-belief Dubins RRT*."""

from __future__ import annotations

from planners.belief_rrt_star import ExpectedBeliefRRTStar
from planners.local_controller import LocalObservationController


class BeliefLocalObservationController(LocalObservationController):
    """Reuse the common local controller with belief-aware global routing."""

    VERSION = "expected_belief_local_observation_v1"

    def __init__(
        self,
        world,
        goal,
        cfg,
        *,
        seed,
        iterations=600,
        replan_interval=25,
        navigation_weight=0.5,
        localization_weight=0.5,
        uncertainty_scale=2.0,
    ) -> None:
        super().__init__(
            world,
            goal,
            cfg,
            method="dubins_rrt_star",
            seed=seed,
            iterations=iterations,
            replan_interval=replan_interval,
        )
        self.planner = ExpectedBeliefRRTStar(
            world,
            cfg,
            seed=seed,
            iterations=iterations,
            navigation_weight=navigation_weight,
            localization_weight=localization_weight,
            uncertainty_scale=uncertainty_scale,
        )

    def set_belief_covariance(self, covariance) -> None:
        self.planner.set_start_covariance(covariance)
