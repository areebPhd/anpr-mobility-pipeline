from datetime import datetime, timedelta

from ch3_pipeline.phase8 import (
    AcceptedLeg, break_reason, chain_legs, privacy_safe_text, publishable,
    restricted_chain_token, weighted_average_speed_kph,
)


BASE = datetime(2018, 1, 1, 8, 0)


def leg(start, end, origin="A", destination="B", key="restricted-a", week=1,
        cross_week=False, distance=1000.0):
    return AcceptedLeg(key, week, origin, destination, BASE + timedelta(minutes=start),
                       BASE + timedelta(minutes=end), distance, cross_week)


def test_continuous_legs_join_and_unordered_input_is_sorted():
    chains = chain_legs([leg(12, 20, "B", "C"), leg(0, 10)], 30)
    assert [len(c) for c in chains] == [2]


def test_camera_discontinuity_breaks_chain():
    chains = chain_legs([leg(0, 10), leg(12, 20, "X", "C")], 30)
    assert [len(c) for c in chains] == [1, 1]


def test_threshold_boundary_is_inclusive():
    a = leg(0, 10)
    assert break_reason(a, leg(40, 50, "B", "C"), 30 * 60) is None
    assert break_reason(a, leg(41, 50, "B", "C"), 30 * 60) == "gap_threshold_exceeded"


def test_overlap_and_zero_gap_both_break():
    a = leg(0, 10)
    assert break_reason(a, leg(9, 15, "B", "C"), 1800) == "temporal_overlap"
    assert break_reason(a, leg(10, 15, "B", "C"), 1800) == "nonpositive_interleg_gap"


def test_week_boundary_breaks_and_cross_week_leg_is_excluded():
    a = leg(0, 10)
    b = leg(12, 20, "B", "C", week=2)
    assert break_reason(a, b, 1800) == "authorised_linkage_boundary"
    assert len(chain_legs([a, leg(12, 20, "B", "C", cross_week=True)], 30)) == 1


def test_chain_token_is_deterministic_and_threshold_specific():
    a = restricted_chain_token("x", 1, 1, 30)
    assert a == restricted_chain_token("x", 1, 1, 30)
    assert a != restricted_chain_token("x", 1, 1, 60)
    assert len(a) == 64


def test_speed_uses_total_distance_over_total_elapsed():
    assert weighted_average_speed_kph(10_000, 1_000) == 36.0
    assert weighted_average_speed_kph(1, 0) is None
    assert weighted_average_speed_kph(-1, 1) is None


def test_suppression_boundary_and_privacy_safe_reporting():
    assert not publishable(9) and publishable(10)
    assert privacy_safe_text("aggregate chain count only")
    assert not privacy_safe_text("sequence_key=secret")
