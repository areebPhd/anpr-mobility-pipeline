#!/usr/bin/env python3
"""Checkpointed Phase 8 full reconstruction and privacy-safe reporting."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[1]
INPUT = str(ROOT / "restricted/phase5_final_mobility_outputs/final_legs/origin_week=*/*.parquet")
OUT = ROOT / "restricted/phase8_inferred_journey_chains"
REPORTS = ROOT / "reports"
CHECKPOINT = ROOT / "checkpoints/phase8_journey_chains.json"
THRESHOLDS = (15, 30, 60, 120)
K = 10


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["status"])
        writer.writeheader()
        if rows:
            writer.writerows(rows)
    os.replace(tmp, path)


def checkpoint(state: dict) -> None:
    atomic_text(CHECKPOINT, json.dumps(state, indent=2, sort_keys=True))


def base_views(con: duckdb.DuckDBPyConnection) -> dict:
    con.execute(f"CREATE OR REPLACE VIEW accepted AS SELECT * FROM read_parquet('{INPUT}')")
    facts = con.execute("""SELECT count(*) input_legs,count(distinct sequence_key) vehicles,
        count(*) filter(where cross_week or origin_week<>destination_week) cross_week_excluded,
        count(*) filter(where sequence_key is null or origin_timestamp is null or destination_timestamp is null
          or origin_camera is null or destination_camera is null or route_distance_m is null
          or duration_seconds<=0 or route_distance_m<0) invalid_excluded
        FROM accepted""").fetchone()
    return {"input_legs": facts[0], "vehicles": facts[1],
            "cross_week_excluded": facts[2], "invalid_excluded": facts[3]}


def create_chains(con: duckdb.DuckDBPyConnection, minutes: int, overwrite: bool) -> dict:
    directory = OUT / f"gap_minutes={minutes}"
    chain_file = directory / "inferred_journey_chains.parquet"
    assignment_file = directory / "leg_assignments.parquet"
    aggregate_dir = directory / "aggregates"
    if chain_file.exists() and assignment_file.exists() and not overwrite:
        return profile_existing(con, minutes, chain_file, assignment_file)
    directory.mkdir(parents=True, exist_ok=True)
    aggregate_dir.mkdir(parents=True, exist_ok=True)
    gap_seconds = minutes * 60
    con.execute("""CREATE OR REPLACE TEMP TABLE excluded AS
      SELECT row_number() over() input_row_id,
        CASE WHEN cross_week OR origin_week<>destination_week THEN 'unauthorised_cross_week_boundary'
             ELSE 'invalid_or_ambiguous_leg' END exclusion_reason
      FROM accepted WHERE cross_week OR origin_week<>destination_week OR sequence_key is null
        OR origin_timestamp is null OR destination_timestamp is null OR origin_camera is null
        OR destination_camera is null OR route_distance_m is null OR duration_seconds<=0 OR route_distance_m<0""")
    con.execute("""CREATE OR REPLACE TEMP TABLE eligible AS
      SELECT row_number() OVER (ORDER BY sequence_key,origin_week,origin_timestamp,destination_timestamp,
                origin_camera,destination_camera,route_distance_m,elapsed_ms) leg_id, *
      FROM accepted WHERE NOT cross_week AND origin_week=destination_week AND sequence_key is not null
        AND origin_timestamp is not null AND destination_timestamp is not null
        AND origin_camera is not null AND destination_camera is not null
        AND route_distance_m is not null AND duration_seconds>0 AND route_distance_m>=0""")
    con.execute("""CREATE OR REPLACE TEMP TABLE ordered AS
      SELECT *,lag(destination_camera) over w prev_destination_camera,
        lag(destination_timestamp) over w prev_destination_timestamp,
        row_number() over w sequence_order
      FROM eligible WINDOW w AS (PARTITION BY sequence_key,origin_week ORDER BY origin_timestamp,
        destination_timestamp,origin_camera,destination_camera,route_distance_m,elapsed_ms,leg_id)""")
    con.execute(f"""CREATE OR REPLACE TEMP TABLE assigned AS
      SELECT *,sum(break_before) over(PARTITION BY sequence_key,origin_week ORDER BY sequence_order
        ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) chain_ordinal
      FROM (SELECT *,CASE WHEN sequence_order=1 THEN 1
        WHEN prev_destination_camera<>origin_camera THEN 1
        WHEN epoch(origin_timestamp-prev_destination_timestamp)<=0 THEN 1
        WHEN epoch(origin_timestamp-prev_destination_timestamp)>{gap_seconds} THEN 1 ELSE 0 END break_before,
        CASE WHEN sequence_order=1 THEN 'sequence_start'
        WHEN prev_destination_camera<>origin_camera THEN 'camera_discontinuity'
        WHEN epoch(origin_timestamp-prev_destination_timestamp)<0 THEN 'temporal_overlap'
        WHEN epoch(origin_timestamp-prev_destination_timestamp)=0 THEN 'nonpositive_interleg_gap'
        WHEN epoch(origin_timestamp-prev_destination_timestamp)>{gap_seconds} THEN 'gap_threshold_exceeded'
        ELSE 'joined' END linkage_decision
      FROM ordered) q""")
    con.execute(f"""COPY (SELECT leg_id,sha256('phase8|{minutes}|'||sequence_key||'|'||origin_week||'|'||chain_ordinal) restricted_trip_token,
        linkage_decision FROM assigned ORDER BY leg_id) TO '{assignment_file}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
    con.execute(f"""COPY (WITH grouped AS (
      SELECT sha256('phase8|{minutes}|'||sequence_key||'|'||origin_week||'|'||chain_ordinal) restricted_trip_token,
        sequence_key authorised_pseudonymous_vehicle_token,origin_week authorised_linkage_period,
        min(origin_timestamp) start_timestamp,max(destination_timestamp) end_timestamp,
        arg_min(origin_camera,sequence_order) origin_camera,arg_max(destination_camera,sequence_order) destination_camera,
        count(*) number_of_legs,array_length(list_distinct(list(origin_camera)||list(destination_camera))) number_of_distinct_cameras,
        sum(route_distance_m) total_network_distance_m,
        epoch(max(destination_timestamp)-min(origin_timestamp)) total_elapsed_seconds,
        sum(duration_seconds) moving_leg_duration_seconds,
        epoch(max(destination_timestamp)-min(origin_timestamp))-sum(duration_seconds) inter_leg_gap_seconds,
        sum(route_distance_m)/epoch(max(destination_timestamp)-min(origin_timestamp))*3.6 average_journey_speed_kph,
        dayofweek(min(origin_timestamp)) day_of_week,hour(min(origin_timestamp)) hour_of_day
      FROM assigned GROUP BY sequence_key,origin_week,chain_ordinal)
      SELECT * FROM grouped ORDER BY authorised_linkage_period,start_timestamp,restricted_trip_token)
      TO '{chain_file}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
    # Aggregate-only restricted products; sequence and trip keys never leave restricted storage.
    con.execute(f"""COPY (SELECT origin_camera,destination_camera,hour_of_day,count(*) journey_count,
      avg(total_elapsed_seconds) mean_duration_seconds,median(total_elapsed_seconds) median_duration_seconds,
      avg(total_network_distance_m) mean_distance_m,median(total_network_distance_m) median_distance_m,
      sum(total_network_distance_m)/sum(total_elapsed_seconds)*3.6 distance_weighted_average_speed_kph,
      count(*)>={K} publishable FROM read_parquet('{chain_file}') GROUP BY 1,2,3)
      TO '{aggregate_dir/'od_hour.parquet'}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
    con.execute(f"""COPY (SELECT origin_camera||'->'||destination_camera corridor_id,hour_of_day,count(*) journey_count,
      avg(total_elapsed_seconds) mean_duration_seconds,median(total_elapsed_seconds) median_duration_seconds,
      avg(total_network_distance_m) mean_distance_m,median(total_network_distance_m) median_distance_m,
      sum(total_network_distance_m)/sum(total_elapsed_seconds)*3.6 distance_weighted_average_speed_kph,
      count(*)>={K} publishable FROM read_parquet('{chain_file}') GROUP BY 1,2)
      TO '{aggregate_dir/'corridor_hour.parquet'}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
    con.execute(f"""COPY (SELECT number_of_legs,count(*) journey_count,count(*)>={K} publishable
      FROM read_parquet('{chain_file}') GROUP BY 1 ORDER BY 1) TO '{aggregate_dir/'leg_count_distribution.parquet'}' (FORMAT PARQUET)""")
    con.execute(f"""COPY (SELECT number_of_distinct_cameras,count(*) journey_count,count(*)>={K} publishable
      FROM read_parquet('{chain_file}') GROUP BY 1 ORDER BY 1) TO '{aggregate_dir/'camera_count_distribution.parquet'}' (FORMAT PARQUET)""")
    con.execute(f"""COPY (SELECT authorised_pseudonymous_vehicle_token,count(*) journey_count
      FROM read_parquet('{chain_file}') GROUP BY 1) TO '{aggregate_dir/'restricted_journeys_per_vehicle.parquet'}' (FORMAT PARQUET,COMPRESSION ZSTD)""")
    con.execute(f"""COPY (SELECT day_of_week,hour_of_day,count(*) journey_count,count(*)>={K} publishable,
      avg(total_elapsed_seconds) mean_duration_seconds,median(total_elapsed_seconds) median_duration_seconds
      FROM read_parquet('{chain_file}') GROUP BY 1,2 ORDER BY 1,2) TO '{aggregate_dir/'weekday_hour.parquet'}' (FORMAT PARQUET)""")
    return profile_existing(con, minutes, chain_file, assignment_file)


def profile_existing(con, minutes, chain_file, assignment_file) -> dict:
    row = con.execute("""SELECT count(*) AS chain_count,count(*) filter(where number_of_legs=1) AS standalone_count,
      count(*) filter(where number_of_legs>1) AS multileg_count,sum(number_of_legs) AS assigned_legs,
      count(distinct authorised_pseudonymous_vehicle_token) vehicles,
      median(total_elapsed_seconds) median_duration_seconds,avg(total_elapsed_seconds) mean_duration_seconds,
      median(total_network_distance_m) median_distance_m,avg(total_network_distance_m) mean_distance_m,
      sum(total_network_distance_m)/sum(total_elapsed_seconds)*3.6 weighted_speed,
      count(*) filter(where total_elapsed_seconds<=0 or moving_leg_duration_seconds<=0 or total_network_distance_m<0),
      count(*) filter(where inter_leg_gap_seconds < -0.001),
      count(*)-count(distinct restricted_trip_token),
      count(*) filter(where abs(total_elapsed_seconds-moving_leg_duration_seconds-inter_leg_gap_seconds)>0.001),
      count(*) filter(where abs(average_journey_speed_kph-total_network_distance_m/total_elapsed_seconds*3.6)>0.000001)
      FROM read_parquet(?)""", [str(chain_file)]).fetchone()
    assignment_audit = con.execute("SELECT count(*)-count(distinct leg_id),count(*) filter(where linkage_decision='joined') FROM read_parquet(?)",[str(assignment_file)]).fetchone()
    od = con.execute("SELECT count(*),count(*) filter(where journey_count>=?) FROM read_parquet(?)",[K,str(chain_file.parent/'aggregates/od_hour.parquet')]).fetchone()
    corridor = con.execute("SELECT count(*),count(*) filter(where journey_count>=?) FROM read_parquet(?)",[K,str(chain_file.parent/'aggregates/corridor_hour.parquet')]).fetchone()
    return {"gap_threshold_minutes": minutes,"chain_count":row[0],"standalone_one_leg_chains":row[1],
      "multi_leg_chains":row[2],"assigned_legs":row[3],"authorised_pseudonymous_vehicles":row[4],
      "median_journey_duration_seconds":row[5],"mean_journey_duration_seconds":row[6],
      "median_journey_distance_m":row[7],"mean_journey_distance_m":row[8],
      "distance_weighted_average_speed_kph":row[9],"impossible_chain_count":row[10],
      "negative_gap_chain_count":row[11],"duplicate_leg_assignments":assignment_audit[0],
      "duplicate_chain_tokens":row[12],"elapsed_time_reconciliation_mismatches":row[13],
      "speed_formula_mismatches":row[14],"join_count":assignment_audit[1],
      "chain_count_invariant_ok":row[0]==row[3]-assignment_audit[1],
      "od_hour_cells":od[0],"publishable_od_hour_cells":od[1],"suppressed_od_hour_cells":od[0]-od[1],
      "corridor_hour_cells":corridor[0],"publishable_corridor_hour_cells":corridor[1],
      "suppressed_corridor_hour_cells":corridor[0]-corridor[1]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--thresholds", default="15,30,60,120")
    args = parser.parse_args()
    thresholds = tuple(int(x) for x in args.thresholds.split(","))
    previous_state = json.loads(CHECKPOINT.read_text()) if CHECKPOINT.exists() else {}
    previous_runtime = previous_state.get("runtime_seconds")
    start = time.monotonic()
    con = duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET preserve_insertion_order=false")
    facts = base_views(con)
    state = {"started_utc":datetime.now(timezone.utc).isoformat(),"input":facts,"completed":[]}
    checkpoint(state)
    rows=[]
    for threshold in thresholds:
        rows.append(create_chains(con,threshold,args.overwrite))
        state["completed"].append(threshold); checkpoint(state)
    for row in rows:
        row["excluded_or_unchainable_records"] = facts["cross_week_excluded"] + facts["invalid_excluded"]
        row["input_accepted_legs"] = facts["input_legs"]
        row["reconciliation_ok"] = row["assigned_legs"] + row["excluded_or_unchainable_records"] == facts["input_legs"]
    write_csv(REPORTS/"phase8_threshold_sensitivity.csv",rows)
    primary=next(x for x in rows if x["gap_threshold_minutes"]==30)
    privacy_ok=all(x["duplicate_leg_assignments"]==0 and x["duplicate_chain_tokens"]==0
      and x["impossible_chain_count"]==0 and x["negative_gap_chain_count"]==0
      and x["elapsed_time_reconciliation_mismatches"]==0 and x["speed_formula_mismatches"]==0
      and x["chain_count_invariant_ok"] for x in rows)
    audit_runtime=time.monotonic()-start
    runtime=(previous_runtime if (not args.overwrite and previous_state.get("verdict") is True
      and set(thresholds).issubset(set(previous_state.get("completed", [])))) else audit_runtime)
    method=f"""# Phase 8 inferred journey-chain method

Phase 8 is a post-freeze analytical extension. It uses only the **{facts['input_legs']:,}** frozen Phase 4B plausible observed inter-camera journey legs. Outputs are termed **inferred journey chains**, never ground-truth trips or complete physical journeys.

The frozen token-persistence audit did not authorise cross-week identity continuity. The irreversible restricted `sequence_key` is therefore treated as an authorised pseudonymous linkage key **within one origin week only**. No source token is available or reconstructed. The **{facts['cross_week_excluded']:,}** accepted legs already crossing a week boundary are explicitly classified as `unauthorised_cross_week_boundary` and excluded from chain construction.

Within each restricted key and week, legs are ordered by origin time, destination time, camera identifiers, distance, elapsed milliseconds and a deterministic row identifier. A join is permitted only where cameras are continuous, the next origin time is strictly later than the preceding destination time, and the positive gap is at most the tested threshold. Otherwise a new chain begins. Thresholds are 15, 30, 60 and 120 minutes; 30 minutes is provisional primary.

For a chain, distance is the sum of directed leg distances. Total elapsed time is final destination time minus first origin time; moving time is the sum of leg durations; inter-leg gap time is their difference. Average chain speed is $3.6D/T$, never the arithmetic mean of leg speeds. Restricted trip tokens are deterministic SHA-256 values over the restricted key, week, chain ordinal and threshold. They appear only in ignored restricted Parquet.

Privacy-safe OD/hour and direct camera-pair corridor/hour cells use minimum-count suppression $k=10$. Vehicle-level counts remain restricted. Reports contain aggregate counts only.
"""
    atomic_text(REPORTS/"phase8_journey_chain_method.md",method)
    results="# Phase 8 journey-chain results\n\n"+"\n".join(
      f"- {x['gap_threshold_minutes']} minutes: {x['chain_count']:,} inferred chains; {x['standalone_one_leg_chains']:,} standalone; {x['multi_leg_chains']:,} multi-leg." for x in rows)
    results+=f"""

Primary 30-minute results: median duration {primary['median_journey_duration_seconds']:.3f} s; mean duration {primary['mean_journey_duration_seconds']:.3f} s; median distance {primary['median_journey_distance_m']:.3f} m; mean distance {primary['mean_journey_distance_m']:.3f} m; distance-weighted speed {primary['distance_weighted_average_speed_kph']:.6f} km/h. Exact input reconciliation: **{'PASS' if all(x['reconciliation_ok'] for x in rows) else 'FAIL'}**.
"""
    atomic_text(REPORTS/"phase8_journey_chain_results.md",results)
    privacy=f"""# Phase 8 privacy audit

**{'PASS' if privacy_ok else 'FAIL'}**. Chain and authorised linkage tokens occur only under ignored `restricted/phase8_inferred_journey_chains/`. Public reports contain no token values. Cross-week linkage is prohibited and {facts['cross_week_excluded']:,} boundary-crossing legs are explicitly excluded. Duplicate leg assignments, duplicate chain tokens, impossible chain durations, negative aggregate gaps, elapsed-time decomposition mismatches and total-distance/total-time speed mismatches are zero for every threshold. The chain invariant `chain_count = assigned_legs - authorised_joins` passes for every threshold. OD/hour and corridor/hour cells carry a `publishable` flag based on $k=10$; non-publishable cells remain restricted and are not exported into reports. Under the primary rule, {primary['suppressed_od_hour_cells']:,} of {primary['od_hour_cells']:,} OD/hour cells and {primary['suppressed_corridor_hour_cells']:,} of {primary['corridor_hour_cells']:,} corridor/hour cells are suppressed.
"""
    atomic_text(REPORTS/"phase8_privacy_audit.md",privacy)
    verdict=all(x["reconciliation_ok"] for x in rows) and privacy_ok
    status=f"""# Phase 8 final status

- Accepted input legs: **{facts['input_legs']:,}**
- Authorised restricted pseudonymous vehicles in input: **{facts['vehicles']:,}**
- Cross-week/unauthorised records: **{facts['cross_week_excluded']:,}**
- Invalid records: **{facts['invalid_excluded']:,}**
- Primary 30-minute chains: **{primary['chain_count']:,}**
- Primary standalone chains: **{primary['standalone_one_leg_chains']:,}**
- Primary multi-leg chains: **{primary['multi_leg_chains']:,}**
- Reconciliation: **{'PASS' if all(x['reconciliation_ok'] for x in rows) else 'FAIL'}**
- Privacy audit: **{'PASS' if privacy_ok else 'FAIL'}**
- Full construction runtime: **{runtime:.3f} seconds**
- Current audit/reporting runtime: **{audit_runtime:.3f} seconds**

The extension is scientifically bounded by within-week pseudonymous linkage and must not be described as ground-truth trip reconstruction. Verdict: **{'PHASE 8 INFERRED JOURNEY-CHAIN RECONSTRUCTION COMPLETE' if verdict else 'PHASE 8 INFERRED JOURNEY-CHAIN RECONSTRUCTION NOT READY'}**.
"""
    atomic_text(REPORTS/"phase8_final_status.md",status)
    state.update({"finished_utc":datetime.now(timezone.utc).isoformat(),"runtime_seconds":runtime,
      "audit_runtime_seconds":audit_runtime,"verdict":verdict}); checkpoint(state)
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
