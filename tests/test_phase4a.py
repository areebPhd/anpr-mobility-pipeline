import pytest

from ch3_pipeline.phase4a import (
    TransitionFacts,
    derived_average_speed_kph,
    global_speed_rule,
    primary_classification,
    route_speed_rule,
)


def facts(**changes):
    values = dict(elapsed_seconds=60, same_camera=False, identical_time=False,
                  origin_exact=True, destination_exact=True, route_available=True)
    values.update(changes)
    return TransitionFacts(**values)


def test_mutually_exclusive_classification_precedence():
    assert primary_classification(facts(elapsed_seconds=0, identical_time=True, same_camera=True)) == "identical_time_observation"
    assert primary_classification(facts(elapsed_seconds=-1)) == "negative_duration_other"
    assert primary_classification(facts(same_camera=True)) == "same_camera_repeat"
    assert primary_classification(facts(origin_exact=False)) == "spatially_unresolved"
    assert primary_classification(facts(route_available=False)) == "route_unavailable"
    assert primary_classification(facts(elapsed_seconds=14_401)) == "long_gap_over_240m"
    assert primary_classification(facts(elapsed_seconds=14_400)) == "route_valid_intercamera_le_240m_candidate"


def test_speed_only_after_route_and_positive_duration():
    assert derived_average_speed_kph(1_000, 60, True) == pytest.approx(60)
    assert derived_average_speed_kph(1_000, 0, True) is None
    assert derived_average_speed_kph(1_000, 60, False) is None
    assert derived_average_speed_kph(None, 60, True) is None


def test_candidate_speed_rules_are_tristate():
    assert global_speed_rule(None, 130) is None
    assert global_speed_rule(120, 130) is True
    assert global_speed_rule(140, 130) is False
    assert route_speed_rule(None, 100, 1_000) is None
    assert route_speed_rule(54, 100, 1_000, 1.5) is True
    assert route_speed_rule(55, 100, 1_000, 1.5) is False


def test_cross_week_is_preserved_as_flag_not_identity_class():
    value = facts(cross_week=True)
    assert value.cross_week is True
    assert primary_classification(value) == "route_valid_intercamera_le_240m_candidate"
