from __future__ import annotations

import hashlib
from dataclasses import dataclass

import pandas as pd


def suppress_rare_cells(frame: pd.DataFrame, count_column: str, minimum_count: int) -> pd.DataFrame:
    if minimum_count < 1:
        raise ValueError("minimum_count must be positive")
    return frame.loc[frame[count_column] >= minimum_count].copy()


def sequence_partition(sequence_key: str, proportions: tuple[int, int, int] = (70, 15, 15),
                       seed: str = "chapter3-phase5-v1") -> str:
    if sum(proportions) != 100 or min(proportions) < 0:
        raise ValueError("split proportions must be nonnegative and sum to 100")
    bucket = int.from_bytes(hashlib.sha256(f"{seed}:{sequence_key}".encode()).digest()[:8], "big") % 100
    if bucket < proportions[0]:
        return "train"
    if bucket < proportions[0] + proportions[1]:
        return "validation"
    return "test"


def chronological_partition(week: int, train_end: int = 25, validation_end: int = 30) -> str:
    if week <= train_end:
        return "train"
    if week <= validation_end:
        return "validation"
    return "test"


def combined_partition(min_week: int, max_week: int, train_end: int = 25,
                       validation_end: int = 30) -> str:
    if max_week <= train_end:
        return "train"
    if min_week > train_end and max_week <= validation_end:
        return "validation"
    if min_week > validation_end:
        return "test"
    return "boundary_spanning_quarantine"


def direct_pair_corridor(origin_camera: str, destination_camera: str) -> str:
    if origin_camera == destination_camera:
        raise ValueError("corridor requires different cameras")
    return f"camera_pair:{origin_camera}->{destination_camera}"


def od_aggregate(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"origin_camera", "destination_camera", "duration_seconds", "route_distance_m", "speed_kph"}
    if not required.issubset(frame.columns):
        raise ValueError("missing OD aggregation columns")
    return frame.groupby(["origin_camera", "destination_camera"], as_index=False).agg(
        leg_count=("duration_seconds", "size"), median_duration_seconds=("duration_seconds", "median"),
        median_distance_m=("route_distance_m", "median"), median_speed_kph=("speed_kph", "median"))


def leakage_overlaps(assignments: pd.DataFrame) -> dict[str, int]:
    sets = {name: set(group.sequence_key) for name, group in assignments.groupby("split")}
    names = ("train", "validation", "test")
    return {f"{left}_{right}": len(sets.get(left, set()) & sets.get(right, set()))
            for index, left in enumerate(names) for right in names[index + 1:]}
