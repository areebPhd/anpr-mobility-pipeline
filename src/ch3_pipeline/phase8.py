"""Privacy-aware inferred journey-chain reconstruction primitives.

No function in this module logs or returns source vehicle-token values.  The
production input contains only an irreversible restricted sequence key.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from typing import Iterable


@dataclass(frozen=True)
class AcceptedLeg:
    sequence_key: str
    week: int
    origin_camera: str
    destination_camera: str
    origin_timestamp: datetime
    destination_timestamp: datetime
    route_distance_m: float
    cross_week: bool = False


def break_reason(previous: AcceptedLeg | None, current: AcceptedLeg,
                 gap_threshold_seconds: int) -> str | None:
    """Return why *current* starts a new chain; None means it may be joined."""
    if previous is None:
        return "sequence_start"
    if previous.sequence_key != current.sequence_key or previous.week != current.week:
        return "authorised_linkage_boundary"
    if previous.destination_camera != current.origin_camera:
        return "camera_discontinuity"
    gap = (current.origin_timestamp - previous.destination_timestamp).total_seconds()
    if gap < 0:
        return "temporal_overlap"
    if gap == 0:
        return "nonpositive_interleg_gap"
    if gap > gap_threshold_seconds:
        return "gap_threshold_exceeded"
    return None


def restricted_chain_token(sequence_key: str, week: int, ordinal: int,
                           threshold_minutes: int) -> str:
    payload = f"phase8|{threshold_minutes}|{sequence_key}|{week}|{ordinal}"
    return sha256(payload.encode("utf-8")).hexdigest()


def chain_legs(legs: Iterable[AcceptedLeg], threshold_minutes: int) -> list[list[AcceptedLeg]]:
    """Reference implementation used by synthetic boundary tests.

    Cross-week legs are outside the authorised linkage scope and are omitted;
    production reconciliation records them as explicit exclusions.
    """
    eligible = [leg for leg in legs if not leg.cross_week]
    eligible.sort(key=lambda x: (x.sequence_key, x.week, x.origin_timestamp,
                                 x.destination_timestamp, x.origin_camera,
                                 x.destination_camera))
    output: list[list[AcceptedLeg]] = []
    current: list[AcceptedLeg] = []
    previous: AcceptedLeg | None = None
    threshold = threshold_minutes * 60
    for leg in eligible:
        reason = break_reason(previous, leg, threshold)
        if reason is not None:
            if current:
                output.append(current)
            current = [leg]
        else:
            current.append(leg)
        previous = leg
    if current:
        output.append(current)
    return output


def weighted_average_speed_kph(total_distance_m: float,
                               total_elapsed_seconds: float) -> float | None:
    if total_distance_m < 0 or total_elapsed_seconds <= 0:
        return None
    return total_distance_m / total_elapsed_seconds * 3.6


def publishable(count: int, k: int = 10) -> bool:
    return count >= k


def privacy_safe_text(text: str) -> bool:
    lowered = text.lower()
    prohibited = ("vehicle_token", "sequence_key=", "restricted_trip_token=",
                  "authorised_pseudonymous_vehicle_token=")
    return not any(value in lowered for value in prohibited)
