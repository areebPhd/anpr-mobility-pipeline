from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransitionFacts:
    elapsed_seconds: float
    same_camera: bool
    identical_time: bool
    origin_exact: bool
    destination_exact: bool
    route_available: bool
    cross_week: bool = False


def primary_classification(facts: TransitionFacts, long_gap_reference_seconds: float = 14_400) -> str:
    """Return a mutually exclusive, audit-oriented class without deleting a record."""
    if facts.identical_time or facts.elapsed_seconds == 0:
        return "identical_time_observation"
    if facts.elapsed_seconds < 0:
        return "negative_duration_other"
    if facts.same_camera:
        return "same_camera_repeat"
    if not facts.origin_exact or not facts.destination_exact:
        return "spatially_unresolved"
    if not facts.route_available:
        return "route_unavailable"
    if facts.elapsed_seconds > long_gap_reference_seconds:
        return "long_gap_over_240m"
    return "route_valid_intercamera_le_240m_candidate"


def derived_average_speed_kph(route_distance_m: float | None, elapsed_seconds: float,
                              route_available: bool) -> float | None:
    if not route_available or route_distance_m is None or route_distance_m < 0 or elapsed_seconds <= 0:
        return None
    return route_distance_m / elapsed_seconds * 3.6


def global_speed_rule(speed_kph: float | None, maximum_kph: float) -> bool | None:
    return None if speed_kph is None else speed_kph <= maximum_kph


def route_speed_rule(speed_kph: float | None, provisional_route_time_s: float | None,
                     route_distance_m: float | None, tolerance: float = 1.5) -> bool | None:
    if speed_kph is None or provisional_route_time_s is None or route_distance_m is None or provisional_route_time_s <= 0:
        return None
    route_average_kph = route_distance_m / provisional_route_time_s * 3.6
    return speed_kph <= route_average_kph * tolerance
