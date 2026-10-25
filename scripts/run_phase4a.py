from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase4a_config.yaml"
TRANSITIONS = ROOT / "restricted/phase3a_candidate_transitions/candidates/candidate_transitions.parquet"
ROUTES = ROOT / "restricted/phase3b_camera_osm/calibrated_time_audit/all_metadata_directed_routes.csv"
ANCHOR_TIMES = ROOT / "restricted/phase3b_camera_osm/calibrated_time_audit/anchor_observed_time_summaries.csv"
INVENTORY = ROOT / "reports/camera_metadata_inventory.csv"
EDGES = ROOT / "reports/camera_edge_candidates.csv"
RESTRICTED = ROOT / "restricted/phase4a_transition_classification"
REPORTS = ROOT / "reports"
THESIS = ROOT / "thesis"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def direction_score(direction: str, bearing: float) -> float:
    expected = {"Northbound": [0], "Eastbound": [90], "Southbound": [180], "Westbound": [270],
                "North & Southbound": [0, 180], "East & Westbound": [90, 270]}.get(direction, [])
    if not expected or pd.isna(bearing):
        return .5
    difference = min(abs((bearing - value + 180) % 360 - 180) for value in expected)
    return max(0.0, 1.0 - difference / 90.0)


def register_inputs(connection: duckdb.DuckDBPyConnection) -> None:
    inventory = pd.read_csv(INVENTORY, dtype={"camera_id": "string"})
    exact = inventory[inventory.status.eq("exact_verified_metadata_match")][["camera_id", "camera_direction"]].copy()
    nearest = pd.read_csv(EDGES, dtype={"camera_id": "string"})
    nearest = nearest[(nearest.candidate_rank == 1) & nearest.camera_id.isin(exact.camera_id)]
    direction = exact.merge(nearest[["camera_id", "edge_bearing_deg"]], on="camera_id", how="left")
    direction["direction_compatibility"] = [direction_score(value, bearing) for value, bearing in
                                               zip(direction.camera_direction, direction.edge_bearing_deg)]
    connection.register("exact_cameras", direction)
    routes = pd.read_csv(ROUTES, dtype={"origin_metadata_id": "string", "destination_metadata_id": "string"})
    routes = routes[routes.origin_metadata_id.isin(set(exact.camera_id)) & routes.destination_metadata_id.isin(set(exact.camera_id))]
    routes["route_average_kph"] = routes.route_distance_m / routes.provisional_route_time_s * 3.6
    connection.register("exact_routes", routes)
    empirical = pd.read_csv(ANCHOR_TIMES, dtype={"origin_camera": "string", "destination_camera": "string"})
    empirical = empirical[(empirical.split == "inference") & (empirical.transition_count >= 100) & (empirical.supporting_weeks >= 4)]
    connection.register("empirical_od", empirical[["origin_camera", "destination_camera", "q10_seconds",
                                                     "transition_count", "supporting_weeks"]])


def classified_query(where_clause: str = "") -> str:
    return f"""
    WITH enriched AS (
      SELECT t.origin_camera, t.destination_camera, t.origin_timestamp, t.destination_timestamp,
             t.elapsed_ms, t.elapsed_seconds, t.origin_week, t.destination_week, t.cross_week,
             t.same_camera, t.identical_timestamp, t.exact_duplicate,
             t.repeat_gap_0s, t.repeat_gap_1s, t.repeat_gap_5s, t.repeat_gap_10s, t.repeat_gap_30s, t.repeat_gap_60s,
             t.origin_confidence, t.destination_confidence,
             t.origin_source_file, t.destination_source_file,
             t.origin_source_record_number, t.destination_source_record_number,
             o.camera_id IS NOT NULL origin_exact, d.camera_id IS NOT NULL destination_exact,
             r.route_distance_m, r.provisional_route_time_s, r.predominant_road_class, r.route_edge_count,
             r.route_average_kph, o.direction_compatibility origin_direction_compatibility,
             d.direction_compatibility destination_direction_compatibility,
             e.q10_seconds empirical_q10_seconds, e.transition_count empirical_support,
             e.supporting_weeks empirical_weeks
      FROM read_parquet('{TRANSITIONS}') t
      LEFT JOIN exact_cameras o ON t.origin_camera=o.camera_id
      LEFT JOIN exact_cameras d ON t.destination_camera=d.camera_id
      LEFT JOIN exact_routes r ON t.origin_camera=r.origin_metadata_id AND t.destination_camera=r.destination_metadata_id
      LEFT JOIN empirical_od e ON t.origin_camera=e.origin_camera AND t.destination_camera=e.destination_camera
      {where_clause}
    ), derived AS (
      SELECT *, CASE WHEN route_distance_m IS NOT NULL AND elapsed_seconds > 0
                     THEN route_distance_m / elapsed_seconds * 3.6 END derived_average_speed_kph,
             CASE WHEN EXTRACT(hour FROM origin_timestamp) >= 22 OR EXTRACT(hour FROM origin_timestamp) < 6
                  THEN 'night_clock_22_06' ELSE 'day_clock_06_22' END time_of_day,
             LEAST(origin_direction_compatibility, destination_direction_compatibility) direction_compatibility
      FROM enriched
    )
    SELECT *,
      CASE WHEN identical_timestamp OR elapsed_seconds=0 THEN 'identical_time_observation'
           WHEN elapsed_seconds < 0 THEN 'negative_duration_other'
           WHEN same_camera THEN 'same_camera_repeat'
           WHEN NOT origin_exact OR NOT destination_exact THEN 'spatially_unresolved'
           WHEN route_distance_m IS NULL THEN 'route_unavailable'
           WHEN elapsed_seconds > 14400 THEN 'long_gap_over_240m'
           ELSE 'route_valid_intercamera_le_240m_candidate' END primary_classification,
      (route_distance_m IS NOT NULL AND elapsed_seconds > 0) speed_calculated,
      CASE WHEN route_distance_m IS NOT NULL AND elapsed_seconds > 0 THEN derived_average_speed_kph <= 130 END global_130kph_pass,
      CASE WHEN route_distance_m IS NOT NULL AND elapsed_seconds > 0 AND provisional_route_time_s > 0
           THEN derived_average_speed_kph <= route_average_kph * 1.5 END route_speed_tolerance_pass,
      CASE WHEN empirical_q10_seconds IS NOT NULL AND elapsed_seconds > 0
           THEN elapsed_seconds >= empirical_q10_seconds * 0.5 END empirical_freeflow_envelope_pass,
      CASE WHEN route_distance_m IS NOT NULL AND elapsed_seconds > 0 THEN
           elapsed_seconds <= 14400 AND derived_average_speed_kph <= 130
           AND derived_average_speed_kph <= route_average_kph * 1.5
           AND direction_compatibility >= 0.25
           AND (empirical_q10_seconds IS NULL OR elapsed_seconds >= empirical_q10_seconds * 0.5) END combined_exploratory_pass
    FROM derived
    """


def bounded(connection: duckdb.DuckDBPyConnection) -> dict[str, object]:
    query = classified_query("WHERE t.origin_week IN (1,2) QUALIFY ROW_NUMBER() OVER (PARTITION BY t.origin_week ORDER BY t.origin_source_record_number) <= 250000")
    output = RESTRICTED / "bounded/classified.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    connection.execute(f"COPY ({query}) TO '{output}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    # Direct, independent reconciliation avoids depending on category names.
    total_rows = connection.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(output)]).fetchone()[0]
    class_sum = connection.execute("SELECT SUM(n) FROM (SELECT primary_classification, COUNT(*) n FROM read_parquet(?) GROUP BY 1)", [str(output)]).fetchone()[0]
    cross_week = connection.execute("SELECT COUNT(*) FROM read_parquet(?) WHERE cross_week", [str(output)]).fetchone()[0]
    invalid_speed = connection.execute("SELECT COUNT(*) FROM read_parquet(?) WHERE speed_calculated AND route_distance_m IS NULL", [str(output)]).fetchone()[0]
    return {"rows": total_rows, "class_sum": class_sum, "cross_week_rows": cross_week,
            "invalid_speed_rows": invalid_speed, "pass": total_rows == class_sum and invalid_speed == 0 and cross_week > 0}


def full(connection: duckdb.DuckDBPyConnection) -> None:
    output = RESTRICTED / "full"
    output.mkdir(parents=True, exist_ok=True)
    query = classified_query()
    connection.execute(f"COPY ({query}) TO '{output}' (FORMAT PARQUET, PARTITION_BY (origin_week), COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000, OVERWRITE_OR_IGNORE TRUE)")


def reports(connection: duckdb.DuckDBPyConnection, runtime: float) -> dict[str, object]:
    pattern = str(RESTRICTED / "full/origin_week=*/*.parquet")
    classification = connection.execute("""SELECT primary_classification, COUNT(*) candidate_count,
      COUNT(*) FILTER (WHERE cross_week) cross_week_count,
      COUNT(*) FILTER (WHERE exact_duplicate) exact_duplicate_count,
      COUNT(*) FILTER (WHERE time_of_day='night_clock_22_06') nighttime_count,
      COUNT(*) FILTER (WHERE speed_calculated) speed_calculated_count
      FROM read_parquet(?) GROUP BY 1 ORDER BY candidate_count DESC""", [pattern]).fetchdf()
    expected_classes = ["identical_time_observation", "negative_duration_other", "same_camera_repeat",
                        "spatially_unresolved", "route_unavailable", "long_gap_over_240m",
                        "route_valid_intercamera_le_240m_candidate"]
    missing_classes = [value for value in expected_classes if value not in set(classification.primary_classification)]
    if missing_classes:
        classification = pd.concat([classification, pd.DataFrame({"primary_classification": missing_classes,
            "candidate_count": 0, "cross_week_count": 0, "exact_duplicate_count": 0,
            "nighttime_count": 0, "speed_calculated_count": 0})], ignore_index=True)
    total = int(classification.candidate_count.sum())
    classification["proportion"] = classification.candidate_count / total
    atomic_csv(classification, REPORTS / "phase4a_transition_classification.csv")

    durations = []
    for seconds in (60,300,600,900,1800,3600,7200,14400):
        row = connection.execute("""SELECT COUNT(*) total_candidates,
          COUNT(*) FILTER (WHERE elapsed_seconds > 0 AND elapsed_seconds <= ?) all_positive_within,
          COUNT(*) FILTER (WHERE route_distance_m IS NOT NULL AND elapsed_seconds > 0 AND elapsed_seconds <= ?) routed_within,
          COUNT(*) FILTER (WHERE NOT same_camera AND elapsed_seconds > 0 AND elapsed_seconds <= ?) intercamera_within,
          COUNT(*) FILTER (WHERE cross_week AND elapsed_seconds > 0 AND elapsed_seconds <= ?) cross_week_within
          FROM read_parquet(?)""", [seconds,seconds,seconds,seconds,pattern]).fetchone()
        durations.append({"threshold_seconds":seconds,"threshold_minutes":seconds/60,"total_candidates":row[0],
                          "all_positive_within":row[1],"routed_within":row[2],"intercamera_within":row[3],"cross_week_within":row[4]})
    over = connection.execute("SELECT COUNT(*) FROM read_parquet(?) WHERE elapsed_seconds > 14400", [pattern]).fetchone()[0]
    durations.append({"threshold_seconds":">14400","threshold_minutes":">240","total_candidates":total,
                      "all_positive_within":over,"routed_within":None,"intercamera_within":None,"cross_week_within":None})
    duration_frame = pd.DataFrame(durations)
    atomic_csv(duration_frame, REPORTS / "duration_threshold_sensitivity.csv")
    atomic_csv(duration_frame, THESIS / "table_duration_sensitivity.csv")

    route_profile = connection.execute("""SELECT COUNT(*) FILTER (WHERE route_distance_m IS NOT NULL) routed_transitions,
      COUNT(DISTINCT origin_camera || '|' || destination_camera) FILTER (WHERE route_distance_m IS NOT NULL) routed_od_pairs,
      min(route_distance_m) min_distance_m, quantile_cont(route_distance_m,.1) q10_distance_m,
      quantile_cont(route_distance_m,.25) q25_distance_m, median(route_distance_m) median_distance_m,
      quantile_cont(route_distance_m,.75) q75_distance_m, quantile_cont(route_distance_m,.9) q90_distance_m,
      max(route_distance_m) max_distance_m FROM read_parquet(?) WHERE NOT same_camera""", [pattern]).fetchdf()
    atomic_csv(route_profile, REPORTS / "route_distance_profile.csv")

    speed_rows = []
    speed_total = connection.execute("SELECT COUNT(*) FROM read_parquet(?) WHERE speed_calculated", [pattern]).fetchone()[0]
    for threshold in (30,50,70,90,110,130,160,200):
        values = connection.execute("""SELECT COUNT(*) FILTER (WHERE derived_average_speed_kph <= ?),
          COUNT(*) FILTER (WHERE derived_average_speed_kph > ?),
          COUNT(*) FILTER (WHERE derived_average_speed_kph <= ? AND elapsed_seconds <= 14400)
          FROM read_parquet(?) WHERE speed_calculated""", [threshold,threshold,threshold,pattern]).fetchone()
        speed_rows.append({"speed_threshold_kph":threshold,"speed_calculated_total":speed_total,
                           "at_or_below":values[0],"above":values[1],"at_or_below_and_duration_le_240m":values[2]})
    speed_frame = pd.DataFrame(speed_rows)
    atomic_csv(speed_frame, REPORTS / "derived_speed_sensitivity.csv")
    atomic_csv(speed_frame, THESIS / "table_speed_sensitivity.csv")

    failure = connection.execute("""SELECT origin_camera, destination_camera, COUNT(*) transition_count,
      MIN(origin_week) first_origin_week, MAX(origin_week) last_origin_week, 'directed_route_unavailable' failure_category
      FROM read_parquet(?) WHERE origin_exact AND destination_exact AND NOT same_camera AND route_distance_m IS NULL
      GROUP BY 1,2 ORDER BY transition_count DESC""", [pattern]).fetchdf()
    atomic_csv(failure, REPORTS / "route_failure_audit.csv")

    rules = connection.execute("""SELECT
      COUNT(*) FILTER (WHERE route_distance_m IS NOT NULL AND elapsed_seconds > 0 AND elapsed_seconds <= 14400) duration_only,
      COUNT(*) FILTER (WHERE route_distance_m IS NOT NULL AND elapsed_seconds > 0 AND elapsed_seconds <= 14400 AND global_130kph_pass) global_max_speed,
      COUNT(*) FILTER (WHERE route_distance_m IS NOT NULL AND elapsed_seconds > 0 AND elapsed_seconds <= 14400 AND route_speed_tolerance_pass) route_speed_tolerance,
      COUNT(*) FILTER (WHERE route_distance_m IS NOT NULL AND elapsed_seconds > 0 AND elapsed_seconds <= 14400 AND empirical_freeflow_envelope_pass) empirical_od_envelope,
      COUNT(*) FILTER (WHERE combined_exploratory_pass) combined_rule,
      COUNT(*) FILTER (WHERE route_distance_m IS NOT NULL AND elapsed_seconds > 0) eligible_routed
      FROM read_parquet(?)""", [pattern]).fetchone()
    rule_frame = pd.DataFrame([
        {"candidate_rule":"duration_only_le_240m","passing_transitions":rules[0],"eligible_routed":rules[5]},
        {"candidate_rule":"duration_plus_global_130kph","passing_transitions":rules[1],"eligible_routed":rules[5]},
        {"candidate_rule":"duration_plus_route_speed_1.5x","passing_transitions":rules[2],"eligible_routed":rules[5]},
        {"candidate_rule":"duration_plus_empirical_od_q10_envelope","passing_transitions":rules[3],"eligible_routed":rules[5]},
        {"candidate_rule":"combined_exploratory","passing_transitions":rules[4],"eligible_routed":rules[5]},
    ])
    rule_frame["passing_proportion"] = rule_frame.passing_transitions / rule_frame.eligible_routed
    atomic_csv(rule_frame, THESIS / "table_transition_plausibility.csv")

    input_total = connection.execute("SELECT COUNT(*) FROM read_parquet(?)", [str(TRANSITIONS)]).fetchone()[0]
    output_total = connection.execute("SELECT COUNT(*) FROM read_parquet(?)", [pattern]).fetchone()[0]
    class_sum = int(classification.candidate_count.sum())
    speed_invalid = connection.execute("SELECT COUNT(*) FROM read_parquet(?) WHERE speed_calculated AND (route_distance_m IS NULL OR elapsed_seconds <= 0)", [pattern]).fetchone()[0]
    return {"input_total":input_total,"output_total":output_total,"class_sum":class_sum,"speed_invalid":speed_invalid,
            "route_failures":int(failure.transition_count.sum()) if len(failure) else 0,
            "routed_transitions":int(route_profile.routed_transitions.iloc[0]),"runtime":runtime,
            "reconciled":input_total==output_total==class_sum and speed_invalid==0,
            "rule_frame":rule_frame,"classification":classification}


def write_methodology(result: dict[str, object]) -> None:
    classes = result["classification"]
    class_lines = "\n".join(f"- {row.primary_classification}: {int(row.candidate_count):,}" for row in classes.itertuples())
    findings = f"""# Phase 4A methodological findings

Phase 4A preserves and classifies all {result['input_total']:,} Phase 3A candidates. Only the 119 exact verified cameras receive spatial enrichment; inferred mappings are prohibited. Route distance and derived average speed are present only when both cameras are exact, the directed 2018 OSM route exists, and duration is positive.

Mutually exclusive reconciliation classes:

{class_lines}

The 240-minute boundary is an exploratory sensitivity reference, not a frozen final threshold. Cross-week, repeated-gap, time-of-day, exact-duplicate, confidence, and direction fields remain flags so their overlap is not lost. Clock-time night uses the unresolved naive timestamp and is not claimed as verified local time.

Five transparent candidate rules were compared: duration-only, global 130 km/h, route-average speed with 1.5× tolerance, empirical OD q10 envelope, and a combined exploratory rule. None is adopted as final. Empirical envelopes inherit the established long-gap contamination and therefore require caution.

Exact reconciliation: {result['input_total']:,} input = {result['output_total']:,} classified = {result['class_sum']:,} sum of exclusive classes. Invalid speed calculations: {result['speed_invalid']:,}.
"""
    atomic_text(findings, REPORTS / "phase4a_methodological_findings.md")
    status = f"""# Phase 4A final status

**{'PASS' if result['reconciled'] else 'FAIL'}** — all Phase 3A candidates are preserved and explicitly classified.

- Input candidates: {result['input_total']:,}
- Classified candidates: {result['output_total']:,}
- Exclusive-class sum: {result['class_sum']:,}
- Reconciliation exact: {result['reconciled']}
- Routed exact-camera transitions: {result['routed_transitions']:,}
- Route-failed transitions: {result['route_failures']:,}
- Invalid speed calculations: {result['speed_invalid']:,}
- Runtime: {result['runtime']:.3f} seconds
- Exact spatial cameras: 119; inferred mappings used: 0

No final duration or speed threshold was selected. No final corridors, dataset splits, dashboard, or speed-filtered final dataset were created.
"""
    atomic_text(status, REPORTS / "phase4a_final_status.md")
    method = """# Section 3.6: Journey-leg segmentation and plausibility method

Observed consecutive inter-camera transition candidates were retained intact and classified through an auditable mutually exclusive hierarchy. Same-camera and identical-time records were separated before spatial evaluation. Spatial enrichment was restricted to exact verified cameras. Directed historical OSM distances respected one-way topology, and average speed was derived only for positive-duration records with an available route. Duration and speed rules were evaluated as sensitivities rather than final filters.
"""
    atomic_text(method, THESIS / "section_3_6_journey_leg_method.md")
    thesis_findings = """# Section 3.10: Transition plausibility findings

Raw consecutive observations contain substantial long-gap contamination and cannot be equated with complete physical trips. Phase 4A therefore reports mutually exclusive transition classes and overlapping sensitivity flags. Primary spatial evidence uses only exact camera matches; no inferred mapping enters the analysis. Final duration and speed thresholds remain unresolved pending interpretation of the sensitivity tables.
"""
    atomic_text(thesis_findings, THESIS / "section_3_10_plausibility_findings.md")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("bounded", "full", "reports"))
    args = parser.parse_args()
    connection = duckdb.connect()
    connection.execute("SET threads=16")
    connection.execute("SET memory_limit='120GB'")
    register_inputs(connection)
    started = time.perf_counter()
    if args.mode == "bounded":
        result = bounded(connection)
    elif args.mode == "full":
        full(connection)
        result = {"status":"full_written","runtime":time.perf_counter()-started}
        atomic_text(json.dumps(result, sort_keys=True) + "\n", RESTRICTED / "full_run_status.json")
    else:
        status_path = RESTRICTED / "full_run_status.json"
        construction_runtime = json.loads(status_path.read_text())["runtime"] if status_path.exists() else time.perf_counter()-started
        result = reports(connection, construction_runtime)
        write_methodology(result)
        result = {key:value for key,value in result.items() if key not in ("rule_frame","classification")}
    connection.close()
    print(json.dumps(result, default=str, sort_keys=True))


if __name__ == "__main__":
    main()
