"""Deterministic map generation and loading for LocalizationRL experiments.

An entity's ``size`` is the side length of its square footprint.  Landmark
footprints are traversable sensing regions; obstacle footprints are blocked.
"""

from __future__ import annotations

import json
import random
from collections import Counter, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np


MAP_DATASET_VERSION = 1

# These ranges deliberately vary geometry instead of tuning a single layout.
# Landmark zones get smaller as difficulty rises, while obstacle density rises.
DIFFICULTY_PROFILES = {
    "easy": {
        "obstacle_count": (1, 2),
        "obstacle_sizes": (5, 7),
        "landmark_count": (3, 4),
        "landmark_sizes": (9, 11),
        "corridor_offset": (1.0, 5.0),
    },
    "medium": {
        "obstacle_count": (3, 5),
        "obstacle_sizes": (5, 7, 9),
        "landmark_count": (2, 3),
        "landmark_sizes": (7, 9),
        "corridor_offset": (4.0, 10.0),
    },
    "hard": {
        "obstacle_count": (5, 8),
        "obstacle_sizes": (5, 7),
        "landmark_count": (1, 2),
        "landmark_sizes": (5, 7),
        "corridor_offset": (7.0, 14.0),
    },
}

# The training and validation splits above are intentionally left unchanged.
# The official paper test is generated from a harder, separately declared
# distribution.  Difficulty is controlled by geometry rather than by looking
# at policy outcomes: more obstacles must block the straight start--goal line,
# obstacle-induced path stretch increases, and landmark support becomes both
# smaller and farther from the direct route.
FINAL_TEST_DIFFICULTY_PROFILES = {
    "easy": {
        "obstacle_count": (4, 6),
        "obstacle_sizes": (5, 7),
        "landmark_count": (2, 3),
        "landmark_sizes": (5, 7),
        "corridor_offset": (7.0, 12.0),
        "corridor_blockers": 1,
        "minimum_detour_ratio": 1.02,
    },
    "medium": {
        "obstacle_count": (7, 9),
        "obstacle_sizes": (5, 7, 9),
        "landmark_count": (1, 2),
        "landmark_sizes": (5, 7),
        "corridor_offset": (11.0, 17.0),
        "corridor_blockers": 2,
        "minimum_detour_ratio": 1.06,
    },
    "hard": {
        "obstacle_count": (10, 12),
        "obstacle_sizes": (7, 9),
        "landmark_count": (1, 1),
        "landmark_sizes": (3, 5),
        "corridor_offset": (15.0, 22.0),
        "corridor_blockers": 3,
        "minimum_detour_ratio": 1.10,
    },
}


def _square_bounds(entity: Dict[str, int]) -> Tuple[int, int, int, int]:
    half = int(entity["size"]) // 2
    return (
        int(entity["x"]) - half,
        int(entity["x"]) + half,
        int(entity["y"]) - half,
        int(entity["y"]) + half,
    )


def _overlaps(first: Dict[str, int], second: Dict[str, int], gap: int = 1) -> bool:
    ax0, ax1, ay0, ay1 = _square_bounds(first)
    bx0, bx1, by0, by1 = _square_bounds(second)
    return not (
        ax1 + gap < bx0
        or bx1 + gap < ax0
        or ay1 + gap < by0
        or by1 + gap < ay0
    )


def _fits(entity: Dict[str, int], map_size: int, others: Iterable[Dict[str, int]]) -> bool:
    x0, x1, y0, y1 = _square_bounds(entity)
    # Coordinates 1..map_size are the unpadded interior. Leave one free cell
    # between an entity and the hard map border.
    if x0 < 2 or y0 < 2 or x1 > map_size - 1 or y1 > map_size - 1:
        return False
    return not any(_overlaps(entity, other) for other in others)


def _sample_boundary_pair(rng: random.Random, map_size: int) -> Tuple[List[int], List[int]]:
    low, high = 3, map_size - 2
    coordinate = lambda: rng.randint(low, high)
    edge = rng.choice(("top", "bottom", "left", "right"))
    if edge == "top":
        return [coordinate(), low], [coordinate(), high]
    if edge == "bottom":
        return [coordinate(), high], [coordinate(), low]
    if edge == "left":
        return [low, coordinate()], [high, coordinate()]
    return [high, coordinate()], [low, coordinate()]


def _random_entity(
    rng: random.Random,
    map_size: int,
    sizes: Sequence[int],
    others: Sequence[Dict[str, int]],
    attempts: int = 400,
) -> Dict[str, int] | None:
    for _ in range(attempts):
        size = int(rng.choice(sizes))
        half = size // 2
        candidate = {
            "x": rng.randint(2 + half, map_size - 1 - half),
            "y": rng.randint(2 + half, map_size - 1 - half),
            "size": size,
        }
        if _fits(candidate, map_size, others):
            return candidate
    return None


def _corridor_landmark(
    rng: random.Random,
    map_size: int,
    sizes: Sequence[int],
    offset_range: Sequence[float],
    start: Sequence[int],
    goal: Sequence[int],
    others: Sequence[Dict[str, int]],
) -> Dict[str, int] | None:
    dx, dy = float(goal[0] - start[0]), float(goal[1] - start[1])
    norm = max((dx * dx + dy * dy) ** 0.5, 1.0)
    perpendicular = (-dy / norm, dx / norm)
    for _ in range(400):
        t = rng.uniform(0.25, 0.75)
        offset = rng.uniform(float(offset_range[0]), float(offset_range[1]))
        offset *= rng.choice((-1.0, 1.0))
        candidate = {
            "x": int(round(start[0] + t * dx + offset * perpendicular[0])),
            "y": int(round(start[1] + t * dy + offset * perpendicular[1])),
            "size": int(rng.choice(sizes)),
        }
        if _fits(candidate, map_size, others):
            return candidate
    return None


def _corridor_obstacle(
    rng: random.Random,
    map_size: int,
    sizes: Sequence[int],
    start: Sequence[int],
    goal: Sequence[int],
    others: Sequence[Dict[str, int]],
    t_center: float,
) -> Dict[str, int] | None:
    """Place an obstacle across the direct start--goal segment."""
    dx, dy = float(goal[0] - start[0]), float(goal[1] - start[1])
    norm = max((dx * dx + dy * dy) ** 0.5, 1.0)
    perpendicular = (-dy / norm, dx / norm)
    for _ in range(400):
        t = rng.uniform(max(0.15, t_center - 0.055), min(0.85, t_center + 0.055))
        offset = rng.uniform(-1.0, 1.0)
        candidate = {
            "x": int(round(start[0] + t * dx + offset * perpendicular[0])),
            "y": int(round(start[1] + t * dy + offset * perpendicular[1])),
            "size": int(rng.choice(sizes)),
        }
        if _fits(candidate, map_size, others):
            return candidate
    return None


def rasterize_map(map_spec: Dict, map_size: int) -> np.ndarray:
    """Build the padded semantic grid for a map specification."""
    world = np.zeros((map_size + 2, map_size + 2), dtype=np.float32)
    world[[0, -1], :] = 1
    world[:, [0, -1]] = 1
    for value, key in ((2, "landmarks"), (1, "obstacles")):
        for entity in map_spec[key]:
            x0, x1, y0, y1 = _square_bounds(entity)
            world[y0 : y1 + 1, x0 : x1 + 1] = value
    return world


def _shortest_path_length(map_spec: Dict, map_size: int) -> int | None:
    world = rasterize_map(map_spec, map_size)
    start = tuple(int(value) for value in map_spec["start"])
    goal = tuple(int(value) for value in map_spec["goal"])
    queue = deque([(start, 0)])
    seen = {start}
    while queue:
        (x, y), distance = queue.popleft()
        if (x, y) == goal:
            return distance
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if (nx, ny) not in seen and world[ny, nx] != 1:
                seen.add((nx, ny))
                queue.append(((nx, ny), distance + 1))
    return None


def _straight_line_blockers(map_spec: Dict) -> int:
    """Count obstacles intersected by a densely sampled direct segment."""
    start = np.asarray(map_spec["start"], dtype=float)
    goal = np.asarray(map_spec["goal"], dtype=float)
    count = 0
    for obstacle in map_spec["obstacles"]:
        x0, x1, y0, y1 = _square_bounds(obstacle)
        intersects = False
        for fraction in np.linspace(0.0, 1.0, 501):
            point = start + fraction * (goal - start)
            if x0 - 0.5 <= point[0] <= x1 + 0.5 and y0 - 0.5 <= point[1] <= y1 + 0.5:
                intersects = True
                break
        count += int(intersects)
    return count


def generate_map(map_id: str, difficulty: str, map_size: int, seed: int) -> Dict:
    """Generate one reachable map, retrying deterministically if necessary."""
    if difficulty not in DIFFICULTY_PROFILES:
        raise ValueError(f"Unknown difficulty {difficulty!r}")
    profile = DIFFICULTY_PROFILES[difficulty]
    rng = random.Random(seed)

    for _ in range(500):
        start, goal = _sample_boundary_pair(rng, map_size)
        reserved = [
            {"x": start[0], "y": start[1], "size": 3},
            {"x": goal[0], "y": goal[1], "size": 3},
        ]
        landmark_count = rng.randint(*profile["landmark_count"])
        obstacle_count = rng.randint(*profile["obstacle_count"])
        landmarks: List[Dict[str, int]] = []
        obstacles: List[Dict[str, int]] = []

        first_landmark = _corridor_landmark(
            rng,
            map_size,
            profile["landmark_sizes"],
            profile["corridor_offset"],
            start,
            goal,
            reserved,
        )
        if first_landmark is None:
            continue
        landmarks.append(first_landmark)

        placement_failed = False
        for _ in range(landmark_count - 1):
            entity = _random_entity(
                rng,
                map_size,
                profile["landmark_sizes"],
                [*reserved, *landmarks],
            )
            if entity is None:
                placement_failed = True
                break
            landmarks.append(entity)
        if placement_failed:
            continue

        for _ in range(obstacle_count):
            entity = _random_entity(
                rng,
                map_size,
                profile["obstacle_sizes"],
                [*reserved, *landmarks, *obstacles],
            )
            if entity is None:
                placement_failed = True
                break
            obstacles.append(entity)
        if placement_failed:
            continue

        result = {
            "id": map_id,
            "difficulty": difficulty,
            "seed": seed,
            "start": start,
            "goal": goal,
            "obstacles": obstacles,
            "landmarks": landmarks,
        }
        path_length = _shortest_path_length(result, map_size)
        if path_length is not None:
            result["shortest_path_length"] = path_length
            return result

    raise RuntimeError(f"Could not generate a valid {difficulty} map for seed {seed}")


def generate_final_test_map(map_id: str, difficulty: str, map_size: int, seed: int) -> Dict:
    """Generate one official-test map from outcome-independent constraints."""
    if difficulty not in FINAL_TEST_DIFFICULTY_PROFILES:
        raise ValueError(f"Unknown difficulty {difficulty!r}")
    profile = FINAL_TEST_DIFFICULTY_PROFILES[difficulty]
    rng = random.Random(seed)

    for _ in range(1500):
        start, goal = _sample_boundary_pair(rng, map_size)
        reserved = [
            {"x": start[0], "y": start[1], "size": 3},
            {"x": goal[0], "y": goal[1], "size": 3},
        ]
        landmark_count = rng.randint(*profile["landmark_count"])
        obstacle_count = rng.randint(*profile["obstacle_count"])
        landmarks: List[Dict[str, int]] = []
        obstacles: List[Dict[str, int]] = []

        first_landmark = _corridor_landmark(
            rng,
            map_size,
            profile["landmark_sizes"],
            profile["corridor_offset"],
            start,
            goal,
            reserved,
        )
        if first_landmark is None:
            continue
        landmarks.append(first_landmark)

        placement_failed = False
        for _ in range(landmark_count - 1):
            entity = _random_entity(
                rng,
                map_size,
                profile["landmark_sizes"],
                [*reserved, *landmarks],
            )
            if entity is None:
                placement_failed = True
                break
            landmarks.append(entity)
        if placement_failed:
            continue

        blocker_count = int(profile["corridor_blockers"])
        blocker_positions = np.linspace(0.25, 0.75, blocker_count)
        for position in blocker_positions:
            entity = _corridor_obstacle(
                rng,
                map_size,
                profile["obstacle_sizes"],
                start,
                goal,
                [*reserved, *landmarks, *obstacles],
                float(position),
            )
            if entity is None:
                placement_failed = True
                break
            obstacles.append(entity)
        if placement_failed:
            continue

        for _ in range(obstacle_count - blocker_count):
            entity = _random_entity(
                rng,
                map_size,
                profile["obstacle_sizes"],
                [*reserved, *landmarks, *obstacles],
            )
            if entity is None:
                placement_failed = True
                break
            obstacles.append(entity)
        if placement_failed:
            continue

        result = {
            "id": map_id,
            "difficulty": difficulty,
            "seed": seed,
            "start": start,
            "goal": goal,
            "obstacles": obstacles,
            "landmarks": landmarks,
        }
        path_length = _shortest_path_length(result, map_size)
        if path_length is None:
            continue
        direct_grid_length = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
        detour_ratio = path_length / max(direct_grid_length, 1)
        straight_line_blockers = _straight_line_blockers(result)
        if straight_line_blockers < blocker_count:
            continue
        if detour_ratio < float(profile["minimum_detour_ratio"]):
            continue

        result["shortest_path_length"] = path_length
        result["obstacle_detour_ratio"] = detour_ratio
        result["straight_line_blockers"] = straight_line_blockers
        return result

    raise RuntimeError(f"Could not generate a valid official-test {difficulty} map for seed {seed}")


def generate_split(split: str, count: int, map_size: int, seed: int) -> Dict:
    """Generate a deterministic, balanced split manifest."""
    maps = []
    seen_start_goal_pairs = set()
    difficulty_names = tuple(DIFFICULTY_PROFILES)
    candidate_index = 0
    while len(maps) < count:
        index = len(maps)
        difficulty = difficulty_names[index % len(difficulty_names)]
        map_seed = seed + candidate_index * 7919
        candidate_index += 1
        map_spec = generate_map(f"{split}_{index:04d}", difficulty, map_size, map_seed)
        pair = (tuple(map_spec["start"]), tuple(map_spec["goal"]))
        if pair in seen_start_goal_pairs:
            continue
        seen_start_goal_pairs.add(pair)
        maps.append(map_spec)

    return {
        "version": MAP_DATASET_VERSION,
        "split": split,
        "map_size": map_size,
        "generation_seed": seed,
        "difficulty_profiles": DIFFICULTY_PROFILES,
        "maps": maps,
    }


def generate_final_test_split(split: str, count: int, map_size: int, seed: int) -> Dict:
    """Generate the official harder test distribution without policy feedback."""
    maps = []
    seen_start_goal_pairs = set()
    difficulty_names = tuple(FINAL_TEST_DIFFICULTY_PROFILES)
    candidate_index = 0
    while len(maps) < count:
        index = len(maps)
        difficulty = difficulty_names[index % len(difficulty_names)]
        map_seed = seed + candidate_index * 7919
        candidate_index += 1
        map_spec = generate_final_test_map(
            f"{split}_{index:04d}", difficulty, map_size, map_seed
        )
        pair = (tuple(map_spec["start"]), tuple(map_spec["goal"]))
        if pair in seen_start_goal_pairs:
            continue
        seen_start_goal_pairs.add(pair)
        maps.append(map_spec)

    return {
        "version": MAP_DATASET_VERSION,
        "split": split,
        "map_size": map_size,
        "generation_seed": seed,
        "generator": "official_final_test_v2",
        "difficulty_profiles": FINAL_TEST_DIFFICULTY_PROFILES,
        "maps": maps,
    }


def save_split(manifest: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def load_map_split(dataset_dir: str | Path, split: str, expected_map_size: int) -> List[Dict]:
    """Load and minimally validate one immutable map split."""
    path = Path(dataset_dir) / f"{split}.json"
    if not path.is_file():
        raise FileNotFoundError(
            f"Map split not found: {path}. Run "
            "`python3 capql/generate_map_splits.py`."
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != MAP_DATASET_VERSION:
        raise ValueError(f"Unsupported map dataset version in {path}")
    if manifest.get("split") != split:
        raise ValueError(f"Expected split {split!r}, found {manifest.get('split')!r} in {path}")
    if int(manifest.get("map_size", -1)) != expected_map_size:
        raise ValueError(
            f"Map size mismatch in {path}: dataset={manifest.get('map_size')}, "
            f"environment={expected_map_size}"
        )
    maps = manifest.get("maps", [])
    if not maps:
        raise ValueError(f"No maps found in {path}")
    return maps


def split_summary(manifest: Dict) -> str:
    difficulties = Counter(item["difficulty"] for item in manifest["maps"])
    obstacle_counts = [len(item["obstacles"]) for item in manifest["maps"]]
    landmark_counts = [len(item["landmarks"]) for item in manifest["maps"]]
    return (
        f"{manifest['split']}: {len(manifest['maps'])} maps, "
        f"difficulty={dict(difficulties)}, obstacles={min(obstacle_counts)}-{max(obstacle_counts)}, "
        f"landmarks={min(landmark_counts)}-{max(landmark_counts)}"
    )
