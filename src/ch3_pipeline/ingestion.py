from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import pyarrow as pa


@dataclass(frozen=True)
class Observation:
    vehicle: str
    camera: str
    timestamp: str
    confidence: float | None


@dataclass(frozen=True)
class ConsecutivePair:
    vehicle: str
    origin_camera: str
    destination_camera: str
    origin_timestamp: str
    destination_timestamp: str


OBSERVATION_SCHEMA = pa.schema([
    pa.field("vehicle", pa.string(), nullable=False),
    pa.field("camera", pa.string(), nullable=False),
    pa.field("timestamp", pa.timestamp("ms"), nullable=False),
    pa.field("confidence", pa.float64(), nullable=True),
    pa.field("source_week", pa.int8(), nullable=False),
])


def consecutive_pairs(sorted_observations: Iterator[Observation]) -> Iterator[ConsecutivePair]:
    """Pair adjacent observations; caller must supply token/time sorted rows.

    This interface deliberately applies no journey-gap, timezone, or camera rule.
    """
    previous: Observation | None = None
    for current in sorted_observations:
        if previous is not None and current.vehicle == previous.vehicle:
            yield ConsecutivePair(previous.vehicle, previous.camera, current.camera, previous.timestamp, current.timestamp)
        previous = current


def parquet_partition_path(root: Path, week: int) -> Path:
    return root / f"week={week:02d}" / "observations.parquet"
