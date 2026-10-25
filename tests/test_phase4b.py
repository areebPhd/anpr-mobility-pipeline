import pytest

from ch3_pipeline.phase4b import (
    EligibleLeg,
    conservative_rule,
    inclusive_rule,
    primary_rejection_reason,
    primary_rule,
    route_class_rule,
)


def leg(duration=3600, speed=90, route_speed=60, direction=.25):
    return EligibleLeg(duration, speed, route_speed, direction)


def test_primary_duration_boundary_60_minutes():
    assert primary_rule(leg(duration=3600))
    assert not primary_rule(leg(duration=3600.001))
    assert primary_rejection_reason(leg(duration=3600.001)) == "duration_above_60m"


def test_conservative_duration_boundary_30_minutes():
    assert conservative_rule(leg(duration=1800))
    assert not conservative_rule(leg(duration=1800.001))


def test_inclusive_duration_boundary_120_minutes():
    assert inclusive_rule(leg(duration=7200, speed=130))
    assert not inclusive_rule(leg(duration=7200.001, speed=130))


def test_global_speed_boundary_130_kph():
    assert primary_rule(leg(speed=130, route_speed=100))
    assert not primary_rule(leg(speed=130.001, route_speed=100))
    assert primary_rejection_reason(leg(speed=130.001, route_speed=100)) == "global_speed_above_130kph"


def test_route_speed_tolerance_boundary_1_5():
    assert primary_rule(leg(speed=90, route_speed=60))
    assert not primary_rule(leg(speed=90.001, route_speed=60))
    assert primary_rejection_reason(leg(speed=90.001, route_speed=60)) == "route_speed_above_1_5x"
    assert not primary_rule(leg(route_speed=None))
    assert primary_rejection_reason(leg(route_speed=None)) == "route_speed_reference_missing"


def test_direction_boundary_point_25():
    assert primary_rule(leg(direction=.25))
    assert not primary_rule(leg(direction=.249999))
    assert primary_rejection_reason(leg(direction=.249999)) == "direction_compatibility_below_0_25"
    assert not primary_rule(leg(direction=None))
    assert primary_rejection_reason(leg(direction=None)) == "direction_compatibility_missing"
    assert not primary_rule(leg(direction=float("nan")))
    assert primary_rejection_reason(leg(direction=float("nan"))) == "direction_compatibility_missing"


def test_rejection_precedence_and_route_class_boundary():
    assert primary_rejection_reason(leg(duration=4000, speed=200, direction=0)) == "duration_above_60m"
    thresholds = {"primary": 100, "unclassified": 70}
    assert route_class_rule(100, "primary", thresholds)
    assert not route_class_rule(100.001, "primary", thresholds)
    assert route_class_rule(70, "unknown", thresholds)
