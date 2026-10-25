"""Evidence-bounded primitives for the Phase 8C Pinto sidecar.

These functions contain only rules stated unambiguously in the reference.
They never log or expose pseudonymous identifiers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass(frozen=True)
class Step:
    origin_camera: str
    destination_camera: str
    elapsed_seconds: float
    distance_m: float | None
    freeflow_kph: float | None
    hour: int


def is_duplicate(step: Step, threshold_seconds: float = 600.0) -> bool:
    """Final-results duplicate rule: same camera and elapsed time < Tdup."""
    return (
        step.origin_camera == step.destination_camera
        and 0 <= step.elapsed_seconds < threshold_seconds
    )


def derived_speed_kph(step: Step) -> float | None:
    if step.distance_m is None or step.distance_m < 0 or step.elapsed_seconds <= 0:
        return None
    return step.distance_m / step.elapsed_seconds * 3.6


def first_pass_bounds_kph(
    freeflow_kph: float | None,
    hour: int,
    *,
    unresolved_fallback_upper_kph: float,
) -> tuple[float, float]:
    """Reference first-pass bounds.

    Night is 22:00--05:59. The caller must explicitly choose 130 or 140 for
    the conflicting no-free-flow upper bound; there is deliberately no default.
    """
    if freeflow_kph is None:
        if unresolved_fallback_upper_kph not in (130.0, 140.0):
            raise ValueError("fallback must be an explicitly audited 130 or 140 km/h")
        return 5.0, unresolved_fallback_upper_kph
    night = hour >= 22 or hour < 6
    return (freeflow_kph / (3.0 if night else 10.0), 2.0 * freeflow_kph)


def tukey_fences(values: Iterable[float], k_low: float = 2.0, k_high: float = 4.0) -> tuple[float, float] | None:
    """Transparent Tukey sensitivity, not a substitute for undocumented Kimber."""
    ordered = sorted(float(v) for v in values)
    if len(ordered) < 10:
        return None

    def quantile(p: float) -> float:
        pos = (len(ordered) - 1) * p
        lo = int(pos)
        hi = min(lo + 1, len(ordered) - 1)
        frac = pos - lo
        return ordered[lo] * (1 - frac) + ordered[hi] * frac

    q1, q3 = quantile(0.25), quantile(0.75)
    iqr = q3 - q1
    return q1 - k_low * iqr, q3 + k_high * iqr


def cumulative_trip_labels(high_outlier: Iterable[bool]) -> list[int]:
    """Increment a trip label after each high-valued invalid step."""
    label = 1
    result: list[int] = []
    previous_high = False
    for current_high in high_outlier:
        if previous_high:
            label += 1
        result.append(label)
        previous_high = bool(current_high)
    return result


def route_utility(endpoint_shortest_m: float | None, component_distance_m: float) -> float | None:
    if endpoint_shortest_m is None or endpoint_shortest_m < 0 or component_distance_m <= 0:
        return None
    return endpoint_shortest_m / component_distance_m


def publishable(count: int, k: int = 10) -> bool:
    return count >= k


def privacy_safe_text(text: str) -> bool:
    lowered = text.lower()
    prohibited = ("vehicle_token=", "sequence_key=", "trip_token=", "number_plate=")
    return not any(item in lowered for item in prohibited)
