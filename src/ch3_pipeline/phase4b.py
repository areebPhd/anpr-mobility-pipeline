from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class EligibleLeg:
    duration_seconds: float
    derived_speed_kph: float
    route_average_kph: float | None
    direction_compatibility: float | None


def primary_rule(leg: EligibleLeg) -> bool:
    return (
        leg.duration_seconds <= 3600
        and leg.derived_speed_kph <= 130
        and leg.route_average_kph is not None
        and math.isfinite(leg.route_average_kph)
        and leg.derived_speed_kph <= leg.route_average_kph * 1.5
        and leg.direction_compatibility is not None
        and math.isfinite(leg.direction_compatibility)
        and leg.direction_compatibility >= 0.25
    )


def conservative_rule(leg: EligibleLeg) -> bool:
    return leg.duration_seconds <= 1800 and primary_rule(
        EligibleLeg(3600, leg.derived_speed_kph, leg.route_average_kph, leg.direction_compatibility)
    )


def inclusive_rule(leg: EligibleLeg) -> bool:
    return leg.duration_seconds <= 7200 and leg.derived_speed_kph <= 130


def primary_rejection_reason(leg: EligibleLeg) -> str:
    if leg.duration_seconds > 3600:
        return "duration_above_60m"
    if leg.derived_speed_kph > 130:
        return "global_speed_above_130kph"
    if leg.route_average_kph is None or not math.isfinite(leg.route_average_kph):
        return "route_speed_reference_missing"
    if leg.derived_speed_kph > leg.route_average_kph * 1.5:
        return "route_speed_above_1_5x"
    if leg.direction_compatibility is None or not math.isfinite(leg.direction_compatibility):
        return "direction_compatibility_missing"
    if leg.direction_compatibility < 0.25:
        return "direction_compatibility_below_0_25"
    return "accepted_plausible_observed_intercamera_journey_leg"


def route_class_rule(speed_kph: float, road_class: str, thresholds: dict[str, float]) -> bool:
    return speed_kph <= thresholds.get(road_class, thresholds["unclassified"])
