from ch3_pipeline.phase8c import (
    Step, cumulative_trip_labels, derived_speed_kph, first_pass_bounds_kph,
    is_duplicate, privacy_safe_text, publishable, route_utility, tukey_fences,
)


def test_duplicate_boundary_is_strict_and_final_threshold_is_ten_minutes():
    assert is_duplicate(Step("A", "A", 599.999, None, None, 12))
    assert not is_duplicate(Step("A", "A", 600, None, None, 12))
    assert not is_duplicate(Step("A", "B", 10, None, None, 12))


def test_speed_units_and_invalid_inputs():
    assert derived_speed_kph(Step("A", "B", 100, 1000, None, 12)) == 36
    assert derived_speed_kph(Step("A", "B", 0, 1000, None, 12)) is None


def test_reference_freeflow_relative_bounds():
    assert first_pass_bounds_kph(60, 12, unresolved_fallback_upper_kph=130) == (6, 120)
    assert first_pass_bounds_kph(60, 23, unresolved_fallback_upper_kph=130) == (20, 120)


def test_conflicting_fallback_requires_explicit_choice():
    assert first_pass_bounds_kph(None, 12, unresolved_fallback_upper_kph=130) == (5, 130)
    assert first_pass_bounds_kph(None, 12, unresolved_fallback_upper_kph=140) == (5, 140)
    try:
        first_pass_bounds_kph(None, 12, unresolved_fallback_upper_kph=135)
    except ValueError:
        pass
    else:
        raise AssertionError("an unaudited compromise must not be accepted")


def test_tukey_minimum_sample_and_fences():
    assert tukey_fences(range(9)) is None
    low, high = tukey_fences(range(10))
    assert low < 0 < high


def test_trip_labels_increment_after_high_outlier():
    assert cumulative_trip_labels([False, True, False, False, True, False]) == [1, 1, 2, 2, 2, 3]


def test_route_utility_is_ratio_not_clipped():
    assert route_utility(500, 1000) == .5
    assert route_utility(None, 1000) is None
    assert route_utility(500, 0) is None


def test_suppression_and_privacy():
    assert not publishable(9)
    assert publishable(10)
    assert privacy_safe_text("aggregate benchmark result")
    assert not privacy_safe_text("sequence_key=restricted")
