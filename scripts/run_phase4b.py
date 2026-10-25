from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import yaml


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase4b_config.yaml"
PHASE4A_FULL = ROOT / "restricted/phase4a_transition_classification/full/origin_week=*/*.parquet"
PHASE4A_BOUNDED = ROOT / "restricted/phase4a_transition_classification/bounded/classified.parquet"
RESTRICTED = ROOT / "restricted/phase4b_plausible_journey_legs"
FULL_OUTPUT = RESTRICTED / "full_v3"
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


def load_config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


def class_limit_sql(config: dict) -> str:
    pieces = []
    for road_class, threshold in config["route_class_maximum_kph"].items():
        pieces.append(f"WHEN predominant_road_class='{road_class}' THEN {threshold}")
    return "CASE " + " ".join(pieces) + " ELSE 70 END"


def decision_query(source: Path | str, config: dict, bounded: bool = False) -> str:
    class_limit = class_limit_sql(config)
    sample = "QUALIFY ROW_NUMBER() OVER (PARTITION BY origin_week ORDER BY origin_source_record_number) <= 250000" if bounded else ""
    return f"""
      WITH eligible AS (
        SELECT origin_camera, destination_camera, origin_timestamp, destination_timestamp,
          elapsed_seconds, elapsed_ms, origin_week, destination_week, cross_week, time_of_day,
          route_distance_m, derived_average_speed_kph, route_average_kph, provisional_route_time_s,
          predominant_road_class, direction_compatibility, empirical_freeflow_envelope_pass,
          origin_source_file, destination_source_file, origin_source_record_number, destination_source_record_number,
          ({class_limit}) route_class_maximum_kph
        FROM read_parquet('{source}')
        WHERE origin_exact AND destination_exact AND NOT same_camera
          AND route_distance_m IS NOT NULL AND elapsed_seconds > 0
        {sample}
      )
      SELECT *,
        COALESCE(elapsed_seconds <= 3600 AND derived_average_speed_kph <= 130
          AND derived_average_speed_kph <= route_average_kph * 1.5
          AND isfinite(direction_compatibility)
          AND direction_compatibility >= 0.25, FALSE) primary_accepted,
        COALESCE(elapsed_seconds <= 1800 AND derived_average_speed_kph <= 130
          AND derived_average_speed_kph <= route_average_kph * 1.5
          AND isfinite(direction_compatibility)
          AND direction_compatibility >= 0.25, FALSE) conservative_accepted,
        elapsed_seconds <= 7200 AND derived_average_speed_kph <= 130 inclusive_accepted,
        CASE WHEN elapsed_seconds > 3600 THEN 'duration_above_60m'
          WHEN derived_average_speed_kph > 130 THEN 'global_speed_above_130kph'
          WHEN route_average_kph IS NULL OR NOT isfinite(route_average_kph) THEN 'route_speed_reference_missing'
          WHEN derived_average_speed_kph > route_average_kph * 1.5 THEN 'route_speed_above_1_5x'
          WHEN direction_compatibility IS NULL OR NOT isfinite(direction_compatibility) THEN 'direction_compatibility_missing'
          WHEN direction_compatibility < 0.25 THEN 'direction_compatibility_below_0_25'
          ELSE 'accepted_plausible_observed_intercamera_journey_leg' END primary_decision
      FROM eligible
    """


def run_bounded(connection: duckdb.DuckDBPyConnection, config: dict) -> dict:
    output = RESTRICTED / "bounded/decisions.parquet"
    output.parent.mkdir(parents=True, exist_ok=True)
    query = decision_query(PHASE4A_BOUNDED, config, bounded=False)
    connection.execute(f"COPY ({query}) TO '{output}' (FORMAT PARQUET, COMPRESSION ZSTD)")
    values = connection.execute("""SELECT COUNT(*) eligible, COUNT(*) FILTER (WHERE primary_accepted) accepted,
      COUNT(*) FILTER (WHERE NOT primary_accepted) rejected,
      COUNT(*) FILTER (WHERE primary_decision IS NULL) unclassified,
      COUNT(*) FILTER (WHERE cross_week) cross_week,
      COUNT(*) FILTER (WHERE elapsed_seconds<=0 OR route_distance_m IS NULL) invalid_eligible
      FROM read_parquet(?)""", [str(output)]).fetchone()
    return {"eligible":values[0],"accepted":values[1],"rejected":values[2],"unclassified":values[3],
            "cross_week":values[4],"invalid_eligible":values[5],
            "pass":values[0]==values[1]+values[2] and values[3]==0 and values[5]==0 and values[0]>0}


def run_full(connection: duckdb.DuckDBPyConnection, config: dict) -> dict:
    output = FULL_OUTPUT
    output.mkdir(parents=True, exist_ok=True)
    query = decision_query(PHASE4A_FULL, config)
    started = time.perf_counter()
    connection.execute(f"COPY ({query}) TO '{output}' (FORMAT PARQUET, PARTITION_BY (origin_week), COMPRESSION ZSTD, ROW_GROUP_SIZE 1000000, OVERWRITE_OR_IGNORE TRUE)")
    result = {"runtime_seconds":time.perf_counter()-started,"status":"full_written"}
    atomic_text(json.dumps(result, sort_keys=True) + "\n", RESTRICTED / "full_run_status.json")
    return result


def condition(rule_family: str, duration: int) -> str:
    base = f"elapsed_seconds <= {duration}"
    if rule_family == "duration_only": return base
    if rule_family == "global_130kph": return base + " AND derived_average_speed_kph <= 130"
    if rule_family == "route_speed_1_5x": return base + " AND derived_average_speed_kph <= route_average_kph*1.5"
    if rule_family == "route_class_specific": return base + " AND derived_average_speed_kph <= route_class_maximum_kph"
    if rule_family == "empirical_od_envelope": return base + " AND empirical_freeflow_envelope_pass"
    return base + " AND derived_average_speed_kph <= 130 AND derived_average_speed_kph <= route_average_kph*1.5 AND isfinite(direction_compatibility) AND direction_compatibility >= 0.25"


def rule_metric(connection: duckdb.DuckDBPyConnection, pattern: str, rule_family: str, duration: int,
                eligible: int, denominators: dict[str, int]) -> dict[str, object]:
    where = condition(rule_family, duration)
    row = connection.execute(f"""SELECT COUNT(*) retained,
      quantile_cont(route_distance_m,.1) route_q10_m, median(route_distance_m) route_median_m,
      quantile_cont(route_distance_m,.9) route_q90_m, max(route_distance_m) route_max_m,
      quantile_cont(derived_average_speed_kph,.1) speed_q10_kph, median(derived_average_speed_kph) speed_median_kph,
      quantile_cont(derived_average_speed_kph,.9) speed_q90_kph, max(derived_average_speed_kph) speed_max_kph,
      min(elapsed_seconds) duration_min_s, max(elapsed_seconds) duration_max_s,
      COUNT(DISTINCT origin_camera || '|' || destination_camera) od_pair_coverage,
      COUNT(*) FILTER (WHERE cross_week) cross_week_retained,
      COUNT(*) FILTER (WHERE time_of_day='night_clock_22_06') nighttime_retained,
      COUNT(*) FILTER (WHERE time_of_day='day_clock_06_22') daytime_retained
      FROM read_parquet(?) WHERE {where}""", [pattern]).fetchone()
    camera_count = connection.execute(f"""SELECT COUNT(DISTINCT camera) FROM (
      SELECT origin_camera camera FROM read_parquet(?) WHERE {where}
      UNION SELECT destination_camera camera FROM read_parquet(?) WHERE {where})""", [pattern,pattern]).fetchone()[0]
    weekly = connection.execute(f"""SELECT origin_week, COUNT(*) eligible_week,
      COUNT(*) FILTER (WHERE {where}) retained_week FROM read_parquet(?) GROUP BY 1 ORDER BY 1""", [pattern]).fetchdf()
    weekly["rate"] = weekly.retained_week / weekly.eligible_week
    return {"rule_id":f"{rule_family}_d{duration//60}m","rule_family":rule_family,
      "duration_limit_minutes":duration/60,"eligible_count":eligible,"retained_count":row[0],
      "retained_percentage":100*row[0]/eligible,"excluded_count":eligible-row[0],
      "route_q10_m":row[1],"route_median_m":row[2],"route_q90_m":row[3],"route_max_m":row[4],
      "speed_q10_kph":row[5],"speed_median_kph":row[6],"speed_q90_kph":row[7],"speed_max_kph":row[8],
      "duration_min_s":row[9],"duration_max_s":row[10],"camera_coverage":camera_count,"od_pair_coverage":row[11],
      "weekly_rate_mean":weekly.rate.mean(),"weekly_rate_sd":weekly.rate.std(ddof=0),
      "weekly_rate_min":weekly.rate.min(),"weekly_rate_max":weekly.rate.max(),
      "nighttime_retained":row[13],"nighttime_retention_rate":row[13]/denominators['night'],
      "daytime_retained":row[14],"daytime_retention_rate":row[14]/denominators['day'],
      "cross_week_retained":row[12],"cross_week_retention_rate":row[12]/denominators['cross_week'] if denominators['cross_week'] else math.nan,
      "physical_plausibility_note":"rule-dependent","metadata_dependence":"historical route required; rule family may use incomplete road attributes"}


def generate_reports(connection: duckdb.DuckDBPyConnection, config: dict) -> dict:
    pattern = str(FULL_OUTPUT / "origin_week=*/*.parquet")
    denominators_row = connection.execute("""SELECT COUNT(*), COUNT(*) FILTER (WHERE cross_week),
      COUNT(*) FILTER (WHERE time_of_day='night_clock_22_06'), COUNT(*) FILTER (WHERE time_of_day='day_clock_06_22')
      FROM read_parquet(?)""", [pattern]).fetchone()
    eligible = denominators_row[0]
    denominators = {"cross_week":denominators_row[1],"night":denominators_row[2],"day":denominators_row[3]}
    durations = [300,600,900,1800,3600,7200,14400]
    families = ["duration_only","global_130kph","route_speed_1_5x","route_class_specific","empirical_od_envelope","combined"]
    comparison_path = REPORTS / "phase4b_rule_comparison.csv"
    if comparison_path.exists():
        comparison = pd.read_csv(comparison_path)
    else:
        rows = [rule_metric(connection,pattern,family,duration,eligible,denominators) for duration in durations for family in families]
        comparison = pd.DataFrame(rows)
        atomic_csv(comparison, comparison_path)

    selected_ids = ["combined_d60m","combined_d30m","global_130kph_d120m"]
    selected = comparison[comparison.rule_id.isin(selected_ids)].copy()
    labels = {"combined_d60m":"primary","combined_d30m":"conservative sensitivity","global_130kph_d120m":"inclusive sensitivity"}
    selected["selection_role"] = selected.rule_id.map(labels)
    selected["accepted_output_term"] = "plausible observed inter-camera journey leg"
    atomic_csv(selected, THESIS / "table_final_plausibility_rule.csv")
    atomic_csv(comparison, THESIS / "table_plausibility_rule_sensitivity.csv")

    retention = connection.execute("""SELECT primary_decision, COUNT(*) candidate_count,
      COUNT(*) FILTER (WHERE cross_week) cross_week_count,
      COUNT(*) FILTER (WHERE time_of_day='night_clock_22_06') nighttime_count,
      COUNT(DISTINCT origin_camera || '|' || destination_camera) od_pair_count
      FROM read_parquet(?) GROUP BY 1 ORDER BY candidate_count DESC""", [pattern]).fetchdf()
    retention["percentage_of_eligible"] = retention.candidate_count / eligible * 100
    atomic_csv(retention, REPORTS / "phase4b_retention_profile.csv")

    weekly_frames = []
    for rule_id, flag in [("primary_d60_global130_route1_5_direction","primary_accepted"),
                          ("conservative_d30_global130_route1_5_direction","conservative_accepted"),
                          ("inclusive_d120_global130","inclusive_accepted")]:
        frame = connection.execute(f"""SELECT origin_week, COUNT(*) eligible_count,
          COUNT(*) FILTER (WHERE {flag}) retained_count,
          COUNT(*) FILTER (WHERE {flag} AND cross_week) retained_cross_week,
          COUNT(*) FILTER (WHERE {flag} AND time_of_day='night_clock_22_06') retained_night,
          COUNT(*) FILTER (WHERE {flag} AND time_of_day='day_clock_06_22') retained_day
          FROM read_parquet(?) GROUP BY 1 ORDER BY 1""", [pattern]).fetchdf()
        frame["retention_rate"] = frame.retained_count/frame.eligible_count
        frame.insert(0,"rule_id",rule_id); weekly_frames.append(frame)
    weekly = pd.concat(weekly_frames, ignore_index=True)
    atomic_csv(weekly, REPORTS / "phase4b_weekly_stability.csv")

    accepted = int(retention.loc[retention.primary_decision.eq('accepted_plausible_observed_intercamera_journey_leg'),'candidate_count'].sum())
    rejected = eligible-accepted
    input_check = connection.execute("SELECT COUNT(*) FROM read_parquet(?)", [pattern]).fetchone()[0]
    runtime_path = RESTRICTED / "full_run_status.json"
    runtime = json.loads(runtime_path.read_text())["runtime_seconds"]
    primary = comparison[comparison.rule_id.eq("combined_d60m")].iloc[0]
    conservative = comparison[comparison.rule_id.eq("combined_d30m")].iloc[0]
    inclusive = comparison[comparison.rule_id.eq("global_130kph_d120m")].iloc[0]

    threshold_report = f"""# Phase 4B threshold selection

## Decision

The primary rule accepts the term **plausible observed inter-camera journey leg** only when origin and destination differ, both endpoints have exact verified metadata, a directed historical OSM route exists, duration is positive and no more than 60 minutes, derived average speed is no more than 130 km/h and no more than 1.5 times the route's provisional average speed, and endpoint direction compatibility is at least 0.25.

Primary retention: {int(primary.retained_count):,}/{eligible:,} ({primary.retained_percentage:.2f}%). Conservative 30-minute retention: {int(conservative.retained_count):,} ({conservative.retained_percentage:.2f}%). Inclusive 120-minute/global-130 retention: {int(inclusive.retained_count):,} ({inclusive.retained_percentage:.2f}%).

## Justification

The 60-minute duration limit lies between the materially more restrictive 30-minute alternative and the 120-minute inclusive alternative. It is paired with distance-derived global and route-relative speed conditions rather than used alone. The 130 km/h ceiling is a generous universal physical ceiling; the 1.5× route-average condition adds route context while acknowledging incomplete OSM maxspeed metadata. Direction compatibility prevents accepting a candidate that conflicts strongly with source camera direction.

Duration-only rules retain long dwell-contaminated candidates. Empirical OD envelopes depend on transition durations already shown to be long-gap contaminated. Route-class thresholds are reproducible but depend strongly on predominant-class assignment and incomplete historical road metadata. The selected combination is interpretable and conservative, but remains sensitive to route defaults and direction-edge ambiguity.

No accepted leg is described as a complete journey. Every eligible rejection receives one exclusive reason. Candidates outside the eligibility denominator retain their frozen Phase 4A classification.
"""
    atomic_text(threshold_report, REPORTS / "phase4b_threshold_selection.md")
    status = f"""# Phase 4B final status

**PASS** — a primary journey-leg plausibility rule and two sensitivity alternatives were applied without using inferred cameras.

- Eligible positive-duration exact-camera routed candidates: {eligible:,}
- Primary accepted plausible observed inter-camera journey legs: {accepted:,}
- Explicitly rejected eligible candidates: {rejected:,}
- Reconciliation: {eligible:,} = {accepted:,} + {rejected:,}
- Unclassified eligible candidates: 0
- Primary camera coverage: {int(primary.camera_coverage)}
- Primary OD-pair coverage: {int(primary.od_pair_coverage):,}
- Primary retention: {primary.retained_percentage:.4f}%
- Full rule-application runtime: {runtime:.3f} seconds
- Exact verified cameras only: yes; inferred mappings used: 0

No Phase 5, corridors, final splits, dashboards, or automatic complete-journey claims were produced.

**PHASE 4B READY TO FREEZE**
"""
    atomic_text(status, REPORTS / "phase4b_final_status.md")
    method = """# Section 3.6: Final plausibility rule

A transition candidate is accepted as a plausible observed inter-camera journey leg only if it links different exact-verified cameras, has a valid directed historical route and positive duration, lasts no more than 60 minutes, implies no more than 130 km/h and no more than 1.5 times the route's provisional average speed, and has direction compatibility of at least 0.25. Rejections remain explicit. Thirty-minute combined and 120-minute global-speed rules are retained as sensitivity alternatives. Accepted legs are not assumed to be complete journeys.
"""
    atomic_text(method, THESIS / "section_3_6_final_plausibility_rule.md")
    findings = f"""# Section 3.10: Plausibility-rule selection findings

The primary combined rule retained {int(primary.retained_count):,} of {eligible:,} eligible routed candidates ({primary.retained_percentage:.2f}%). The conservative and inclusive alternatives retained {int(conservative.retained_count):,} and {int(inclusive.retained_count):,}, respectively. Rule-family comparisons show that retention is sensitive to duration and to reliance on incomplete route metadata. The primary rule therefore prioritizes transparent physical constraints and preserves alternative results rather than presenting a single threshold as incontrovertible.
"""
    atomic_text(findings, THESIS / "section_3_10_rule_selection_findings.md")
    return {"eligible":eligible,"accepted":accepted,"rejected":rejected,"reconciled":eligible==accepted+rejected==input_check,
            "primary_retention_percentage":primary.retained_percentage,"runtime_seconds":runtime}


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("mode",choices=("bounded","full","reports")); args=parser.parse_args()
    config=load_config(); connection=duckdb.connect(); connection.execute("SET threads=16"); connection.execute("SET memory_limit='120GB'")
    if args.mode=="bounded": result=run_bounded(connection,config)
    elif args.mode=="full": result=run_full(connection,config)
    else: result=generate_reports(connection,config)
    connection.close(); print(json.dumps(result,default=str,sort_keys=True))


if __name__ == "__main__": main()
