from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from .config import PipelineConfig
from .profiling import TS_EXPR, _save_checkpoint, _write_csv


OBS_SCHEMA = pa.schema([
    pa.field("vehicle", pa.string(), nullable=False),
    pa.field("camera", pa.string(), nullable=False),
    pa.field("timestamp", pa.timestamp("ms"), nullable=False),
    pa.field("confidence", pa.float64(), nullable=True),
    pa.field("source_week", pa.int8(), nullable=False),
    pa.field("source_record_number", pa.int64(), nullable=False),
    pa.field("source_file", pa.string(), nullable=False),
])


def load_config(path: str | Path) -> tuple[dict[str, Any], Path]:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    return raw, path.parent.parent


def _connect(cfg: dict[str, Any]) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads={int(cfg['runtime']['threads'])}")
    con.execute(f"SET memory_limit='{cfg['runtime']['memory_limit']}'")
    temp = Path(cfg["runtime"]["temporary_directory"])
    temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{str(temp).replace(chr(39), chr(39)*2)}'")
    return con


def _atomic_target(path: Path) -> tuple[Path, Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.unlink(missing_ok=True)
    return temporary, path


def _clean_relation(path: Path, week: int, limit: int | None) -> str:
    safe = str(path).replace("'", "''")
    source = (
        f"read_csv('{safe}', header=true, auto_detect=false, delim=',', quote='\"', escape='\"', "
        "new_line='\\n', strict_mode=true, parallel=false, all_varchar=true, "
        "columns={'vehicle':'VARCHAR','camera':'VARCHAR','timestamp':'VARCHAR','confidence':'VARCHAR'})"
    )
    limited = f"SELECT * FROM {source}" + (f" LIMIT {int(limit)}" if limit else "")
    filename = path.name.replace("'", "''")
    return (
        "SELECT vehicle,camera," + TS_EXPR + " AS timestamp,try_cast(confidence AS DOUBLE) AS confidence,"
        f"{week}::TINYINT AS source_week,row_number() OVER ()::BIGINT AS source_record_number,"
        f"'{filename}'::VARCHAR AS source_file FROM ({limited})"
    )


def _write_week35(path: Path, target: Path, limit: int | None = None) -> dict[str, int]:
    temporary, final = _atomic_target(target)
    writer = pq.ParquetWriter(temporary, OBS_SCHEMA, compression="zstd")
    columns: dict[str, list[Any]] = {name: [] for name in OBS_SCHEMA.names}
    input_rows = retained = rejected = 0

    def flush() -> None:
        if not columns["vehicle"]:
            return
        writer.write_table(pa.Table.from_pydict(columns, schema=OBS_SCHEMA))
        for values in columns.values():
            values.clear()

    from datetime import datetime
    try:
        with path.open("rb") as handle:
            header = handle.readline().rstrip(b"\r\n")
            if header != b"vehicle,camera,timestamp,confidence":
                raise ValueError("Week 35 header mismatch")
            while raw := handle.readline():
                input_rows += 1
                if limit and input_rows > limit:
                    input_rows -= 1
                    break
                terminated = raw.endswith(b"\n")
                body = raw[:-1] if terminated else raw
                if body.endswith(b"\r"):
                    body = body[:-1]
                if not terminated or b'"' in body:
                    rejected += 1; continue
                try:
                    text = body.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    rejected += 1; continue
                fields = text.split(",")
                if len(fields) != 4:
                    rejected += 1; continue
                vehicle, camera, timestamp, confidence = fields
                try:
                    parsed_ts = datetime.fromisoformat(timestamp) if timestamp else None
                    parsed_conf = float(confidence) if confidence else None
                except ValueError:
                    rejected += 1; continue
                if not vehicle or not camera or parsed_ts is None:
                    rejected += 1; continue
                retained += 1
                for key, value in zip(columns, (vehicle, camera, parsed_ts, parsed_conf, 35, input_rows, path.name)):
                    columns[key].append(value)
                if len(columns["vehicle"]) >= 250_000:
                    flush()
        flush()
    finally:
        writer.close()
    os.replace(temporary, final)
    return {"input_rows": input_rows, "retained_rows": retained, "rejected_rows": rejected}


def convert_week(con: duckdb.DuckDBPyConnection, path: Path, week: int, target: Path,
                 expected: dict[str, Any], limit: int | None) -> dict[str, Any]:
    started = time.monotonic()
    if week == 35:
        counts = _write_week35(path, target, limit)
    else:
        temporary, final = _atomic_target(target)
        relation = _clean_relation(path, week, limit)
        con.execute(f"COPY ({relation}) TO '{temporary}' (FORMAT PARQUET, COMPRESSION ZSTD)")
        os.replace(temporary, final)
        retained = con.execute(f"SELECT count(*) FROM read_parquet('{final}')").fetchone()[0]
        counts = {"input_rows": retained, "retained_rows": retained, "rejected_rows": 0}
    parquet_rows = con.execute(f"SELECT count(*) FROM read_parquet('{target}')").fetchone()[0]
    if parquet_rows != counts["retained_rows"]:
        raise AssertionError(f"week {week} parquet reconciliation failed")
    if limit is None:
        if counts["input_rows"] != int(expected["input_rows"]) or counts["retained_rows"] != int(expected["retained_rows"]) or counts["rejected_rows"] != int(expected["rejected_malformed_rows"]):
            raise AssertionError(f"week {week} differs from frozen Phase 2 counts")
    return {"week": week, **counts, "parquet_rows": parquet_rows, "path": str(target),
            "runtime_seconds": round(time.monotonic() - started, 3)}


def token_persistence(con: duckdb.DuckDBPyConnection, observation_glob: str,
                      report_csv: Path, interpretation_md: Path) -> list[dict[str, Any]]:
    con.execute(f"CREATE OR REPLACE TEMP VIEW obs AS SELECT * FROM read_parquet('{observation_glob}', union_by_name=true)")
    weekly = dict(con.execute("SELECT source_week,count(DISTINCT vehicle) FROM obs GROUP BY source_week").fetchall())
    rows: list[dict[str, Any]] = []
    weeks = sorted(weekly)
    for left, right in zip(weeks, weeks[1:]):
        overlap = con.execute("SELECT count(*) FROM (SELECT DISTINCT vehicle FROM obs WHERE source_week=? INTERSECT SELECT DISTINCT vehicle FROM obs WHERE source_week=?)", [left,right]).fetchone()[0]
        rows.append({"scope":"adjacent_weeks","week_a":left,"week_b":right,"distinct_tokens_a":weekly[left],
            "distinct_tokens_b":weekly[right],"overlap_count":overlap,"overlap_prop_a":overlap/weekly[left] if weekly[left] else 0,
            "overlap_prop_b":overlap/weekly[right] if weekly[right] else 0})
    multi = con.execute("SELECT count(*) FROM (SELECT vehicle FROM obs GROUP BY vehicle HAVING count(DISTINCT source_week)>1)").fetchone()[0]
    rows.append({"scope":"multi_week_total","week_a":"","week_b":"","distinct_tokens_a":"","distinct_tokens_b":"",
        "overlap_count":multi,"overlap_prop_a":"","overlap_prop_b":""})
    _write_csv(report_csv, rows)
    adjacent = [r for r in rows if r["scope"] == "adjacent_weeks"]
    min_overlap = min((float(r["overlap_prop_a"]) for r in adjacent), default=0)
    w35 = next((r for r in adjacent if r["week_b"] == 35), None)
    interpretation_md.write_text(
        "# Token persistence interpretation\n\n"
        "This audit contains aggregate counts only; no token values are reported. Cross-week identity continuity remains provisional and is not treated as a verified identity guarantee.\n\n"
        f"- Tokens appearing in multiple profiled weeks: {multi:,}.\n"
        f"- Minimum adjacent-week overlap as a proportion of the earlier week: {min_overlap:.6f}.\n"
        + (f"- Week 34→35 overlap: {w35['overlap_count']:,}; proportions {float(w35['overlap_prop_a']):.6f} and {float(w35['overlap_prop_b']):.6f}.\n" if w35 else "")
        + "\nAbruptly near-zero adjacent overlap would be evidence consistent with possible token rotation. Non-zero overlap supports linkability of some tokens only; it does not prove universal persistence or governance permission. Both within-week and provisional cross-week candidates are therefore retained and explicitly marked.\n",
        encoding="utf-8")
    return rows


def build_ordered_and_transitions(con: duckdb.DuckDBPyConnection, observation_glob: str,
                                  ordered_path: Path, transition_path: Path) -> dict[str, int]:
    ordered_tmp, ordered_final = _atomic_target(ordered_path)
    order = "timestamp,source_week,source_record_number,camera,confidence"
    ordered_sql = f"""
      WITH base AS (
        SELECT *,
          lag(camera) OVER (PARTITION BY vehicle ORDER BY {order}) prev_camera,
          lag(timestamp) OVER (PARTITION BY vehicle ORDER BY {order}) prev_timestamp,
          lag(confidence) OVER (PARTITION BY vehicle ORDER BY {order}) prev_confidence,
          lag(source_week) OVER (PARTITION BY vehicle ORDER BY {order}) prev_week,
          lag(source_record_number) OVER (PARTITION BY vehicle ORDER BY {order}) prev_record,
          lag(source_file) OVER (PARTITION BY vehicle ORDER BY {order}) prev_file
        FROM read_parquet('{observation_glob}', union_by_name=true)
      )
      SELECT *, prev_timestamp IS NULL AS sequence_start,
        coalesce(timestamp=prev_timestamp,false) AS identical_timestamp,
        coalesce(camera=prev_camera,false) AS repeated_same_camera,
        coalesce(camera=prev_camera AND timestamp=prev_timestamp AND confidence IS NOT DISTINCT FROM prev_confidence,false) AS exact_duplicate,
        CASE WHEN prev_timestamp IS NULL THEN NULL ELSE epoch_ms(timestamp)-epoch_ms(prev_timestamp) END AS elapsed_ms
      FROM base ORDER BY vehicle,{order}
    """
    con.execute(f"COPY ({ordered_sql}) TO '{ordered_tmp}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)")
    os.replace(ordered_tmp, ordered_final)
    trans_tmp, trans_final = _atomic_target(transition_path)
    trans_sql = f"""
      SELECT sha256(vehicle) AS sequence_key,
        prev_camera AS origin_camera,camera AS destination_camera,
        prev_timestamp AS origin_timestamp,timestamp AS destination_timestamp,
        elapsed_ms,elapsed_ms/1000.0 AS elapsed_seconds,
        prev_week AS origin_week,source_week AS destination_week,
        prev_week<>source_week AS cross_week,
        prev_camera=camera AS same_camera,
        timestamp=prev_timestamp AS identical_timestamp,
        elapsed_ms<0 AS negative_duration,
        prev_confidence AS origin_confidence,confidence AS destination_confidence,
        exact_duplicate,
        repeated_same_camera AND elapsed_ms=0 AS repeat_gap_0s,
        repeated_same_camera AND elapsed_ms BETWEEN 0 AND 1000 AS repeat_gap_1s,
        repeated_same_camera AND elapsed_ms BETWEEN 0 AND 5000 AS repeat_gap_5s,
        repeated_same_camera AND elapsed_ms BETWEEN 0 AND 10000 AS repeat_gap_10s,
        repeated_same_camera AND elapsed_ms BETWEEN 0 AND 30000 AS repeat_gap_30s,
        repeated_same_camera AND elapsed_ms BETWEEN 0 AND 60000 AS repeat_gap_60s,
        prev_file AS origin_source_file,source_file AS destination_source_file,
        prev_record AS origin_source_record_number,source_record_number AS destination_source_record_number,
        'observed consecutive inter-camera transition candidate' AS endpoint_label
      FROM read_parquet('{ordered_final}') WHERE NOT sequence_start
    """
    con.execute(f"COPY ({trans_sql}) TO '{trans_tmp}' (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)")
    os.replace(trans_tmp, trans_final)
    ordered = con.execute(f"SELECT count(*) FROM read_parquet('{ordered_final}')").fetchone()[0]
    sequences = con.execute(f"SELECT count(*) FROM read_parquet('{ordered_final}') WHERE sequence_start").fetchone()[0]
    transitions = con.execute(f"SELECT count(*) FROM read_parquet('{trans_final}')").fetchone()[0]
    if transitions != ordered - sequences:
        raise AssertionError("global transition invariant failed")
    return {"ordered_observations":ordered,"vehicle_sequences":sequences,"candidate_transitions":transitions,
            "sequence_starts":sequences,"destination_observations":transitions}


def _query_dict(con: duckdb.DuckDBPyConnection, query: str) -> list[dict[str, Any]]:
    result = con.execute(query)
    names = [item[0] for item in result.description]
    return [dict(zip(names, row)) for row in result.fetchall()]


def write_reports(con: duckdb.DuckDBPyConnection, project: Path, transition_path: Path,
                  ordered_path: Path, invariant: dict[str, int], bounded: bool, runtime: float) -> None:
    reports = project / "reports"; thesis = project / "thesis"
    reports.mkdir(exist_ok=True); thesis.mkdir(exist_ok=True)
    t = str(transition_path).replace("'", "''"); o = str(ordered_path).replace("'", "''")
    reconciliation = [{"scope":"global",**invariant,
        "expected_transitions":invariant["ordered_observations"]-invariant["vehicle_sequences"],
        "balanced":invariant["candidate_transitions"]==invariant["ordered_observations"]-invariant["vehicle_sequences"]}]
    weekly = _query_dict(con, f"SELECT destination_week AS week,count(*) candidate_transitions,count(*) FILTER(WHERE same_camera) same_camera_count,count(*) FILTER(WHERE NOT same_camera) distinct_camera_count,count(*) FILTER(WHERE identical_timestamp) zero_duration_count,count(*) FILTER(WHERE negative_duration) negative_duration_count,count(*) FILTER(WHERE cross_week) cross_week_count,avg(origin_confidence) mean_origin_confidence,avg(destination_confidence) mean_destination_confidence FROM read_parquet('{t}') GROUP BY destination_week ORDER BY destination_week")
    duration = _query_dict(con, f"SELECT count(*) total,count(*) FILTER(WHERE elapsed_seconds>0) greater_than_0s,count(*) FILTER(WHERE elapsed_seconds BETWEEN 0 AND 300) up_to_5m,count(*) FILTER(WHERE elapsed_seconds BETWEEN 0 AND 900) up_to_15m,count(*) FILTER(WHERE elapsed_seconds BETWEEN 0 AND 1800) up_to_30m,count(*) FILTER(WHERE elapsed_seconds BETWEEN 0 AND 3600) up_to_60m,count(*) FILTER(WHERE elapsed_seconds BETWEEN 0 AND 7200) up_to_120m,count(*) FILTER(WHERE elapsed_seconds BETWEEN 0 AND 14400) up_to_240m,count(*) FILTER(WHERE elapsed_seconds>14400) longer_than_240m,quantile_cont(elapsed_seconds,.01) q01,quantile_cont(elapsed_seconds,.05) q05,quantile_cont(elapsed_seconds,.25) q25,median(elapsed_seconds) median,quantile_cont(elapsed_seconds,.75) q75,quantile_cont(elapsed_seconds,.95) q95,quantile_cont(elapsed_seconds,.99) q99 FROM read_parquet('{t}')")
    repeat = _query_dict(con, f"SELECT count(*) total,count(*) FILTER(WHERE repeat_gap_0s) flagged_0s,count(*) FILTER(WHERE repeat_gap_1s) flagged_1s,count(*) FILTER(WHERE repeat_gap_5s) flagged_5s,count(*) FILTER(WHERE repeat_gap_10s) flagged_10s,count(*) FILTER(WHERE repeat_gap_30s) flagged_30s,count(*) FILTER(WHERE repeat_gap_60s) flagged_60s FROM read_parquet('{t}')")
    sequence = _query_dict(con, f"SELECT sequence_length,count(*) sequence_count FROM (SELECT vehicle,count(*) sequence_length FROM read_parquet('{o}') GROUP BY vehicle) GROUP BY sequence_length ORDER BY sequence_length")
    coverage = _query_dict(con, f"SELECT count(DISTINCT origin_camera) origin_camera_count,count(DISTINCT destination_camera) destination_camera_count,count(DISTINCT (origin_camera,destination_camera)) origin_destination_pair_count,count(*) FILTER(WHERE same_camera) same_camera_count,count(*) FILTER(WHERE NOT same_camera) inter_camera_count FROM read_parquet('{t}')")
    cross = _query_dict(con, f"SELECT origin_week,destination_week,count(*) candidate_count,count(DISTINCT sequence_key) sequence_count FROM read_parquet('{t}') WHERE cross_week GROUP BY origin_week,destination_week ORDER BY origin_week,destination_week")
    suffix = "_bounded" if bounded else ""
    if bounded:
        _write_csv(reports / "phase3a_bounded_transition_reconciliation.csv", reconciliation)
        (reports / "phase3a_bounded_status.md").write_text(f"# Phase 3A bounded status\n\nPASS: {invariant['ordered_observations']:,} observations, {invariant['vehicle_sequences']:,} non-empty token sequences, and {invariant['candidate_transitions']:,} observed consecutive inter-camera transition candidates reconciled in {runtime:.3f} seconds. No final duration or near-duplicate rule was applied.\n",encoding="utf-8")
        return
    _write_csv(reports / "transition_reconciliation.csv", reconciliation)
    _write_csv(reports / "weekly_transition_profile.csv", weekly)
    _write_csv(reports / "transition_duration_sensitivity.csv", duration)
    _write_csv(reports / "repeat_gap_sensitivity.csv", repeat)
    _write_csv(reports / "sequence_length_profile.csv", sequence)
    _write_csv(reports / "camera_transition_coverage.csv", coverage)
    _write_csv(reports / "cross_week_transition_audit.csv", cross)
    for source, target in [(reconciliation,"table_transition_reconciliation.csv"),(duration,"table_transition_duration_sensitivity.csv"),(repeat,"table_repeat_gap_sensitivity.csv"),(weekly,"table_weekly_transition_profile.csv")]:
        _write_csv(thesis / target, source)
    (thesis / "section_3_6_candidate_transition_method.md").write_text("# Section 3.6 candidate-transition method\n\nObservations were ordered within restricted pseudonymous token using naive parsed timestamp and deterministic source provenance tie-breaks. Adjacent observations define an observed consecutive inter-camera transition candidate, not a complete physical trip. No final duration, confidence, near-duplicate, distance, or speed rule was applied. Cross-week candidates are provisional pending token-governance confirmation.\n",encoding="utf-8")
    (thesis / "section_3_10_transition_findings.md").write_text(f"# Section 3.10 transition findings\n\nThe Phase 3A construction reconciled {invariant['ordered_observations']:,} ordered observations into {invariant['vehicle_sequences']:,} sequence starts and {invariant['candidate_transitions']:,} candidate destinations. Detailed aggregate sensitivity results are reported in the accompanying tables.\n",encoding="utf-8")
    (reports / "phase3a_final_status.md").write_text(f"# Phase 3A final status\n\nPASS: {invariant['candidate_transitions']:,} observed consecutive inter-camera transition candidates reconcile exactly from {invariant['ordered_observations']:,} ordered observations and {invariant['vehicle_sequences']:,} non-empty sequences. Runtime: {runtime:.3f} seconds. Candidate data remain restricted and uncommitted. Timezone, token continuity, final study period, confidence threshold, near-duplicate policy, and transition-duration threshold remain unresolved. No OSM routing or Phase 3B was performed.\n",encoding="utf-8")


def run(config_path: str, bounded: bool = False, weeks: list[int] | None = None,
        rows_per_week: int | None = None, restricted_root: str | None = None,
        checkpoint_path: str | None = None) -> dict[str, Any]:
    cfg, project = load_config(config_path)
    p2cfg = PipelineConfig.load(project / cfg["phase2_config"])
    p2state = json.loads((project / cfg["phase2_checkpoint"]).read_text())
    selected = weeks or p2cfg.weeks
    root = Path(restricted_root).resolve() if restricted_root else project / cfg["restricted_root"]
    checkpoint = Path(checkpoint_path).resolve() if checkpoint_path else project / cfg["checkpoint"]
    state = {"mode":"bounded" if bounded else "full","weeks":selected,"rows_per_week":rows_per_week,"converted":{},"stages":{}}
    if checkpoint.exists():
        loaded = json.loads(checkpoint.read_text())
        if loaded.get("mode")==state["mode"] and loaded.get("weeks")==selected and loaded.get("rows_per_week")==rows_per_week:
            state=loaded
    con = _connect(cfg); started=time.monotonic()
    paths=dict(zip(p2cfg.weeks,p2cfg.source_files()))
    obs_dir=root/"observations"
    for week in selected:
        if str(week) in state["converted"]:
            continue
        target=obs_dir/f"week={week:02d}"/"observations.parquet"
        record=convert_week(con,paths[week],week,target,p2state["records"][str(week)],rows_per_week if bounded else None)
        state["converted"][str(week)]=record; _save_checkpoint(checkpoint,state)
    observation_glob=str(obs_dir/"week=*"/"observations.parquet")
    token_persistence(con,observation_glob,project/"reports/token_persistence_audit.csv",project/"reports/token_persistence_interpretation.md")
    ordered=root/"ordered_observations"/"ordered_observations.parquet"
    transitions=root/"candidates"/"candidate_transitions.parquet"
    invariant=build_ordered_and_transitions(con,observation_glob,ordered,transitions)
    runtime=time.monotonic()-started
    write_reports(con,project,transitions,ordered,invariant,bounded,runtime)
    state["stages"]={"token_persistence":"complete","ordering":"complete","transitions":"complete","reports":"complete","invariant":invariant,"runtime_seconds":round(runtime,3)}
    _save_checkpoint(checkpoint,state); con.close()
    return {**invariant,"runtime_seconds":round(runtime,3),"peak_rss_mb":round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,3),"restricted_root":str(root),"checkpoint":str(checkpoint)}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--config",default="configs/phase3a_config.yaml")
    parser.add_argument("--bounded",action="store_true")
    parser.add_argument("--weeks")
    parser.add_argument("--rows-per-week",type=int)
    parser.add_argument("--restricted-root")
    parser.add_argument("--checkpoint")
    args=parser.parse_args()
    selected=[int(x) for x in args.weeks.split(",")] if args.weeks else None
    result=run(args.config,args.bounded,selected,args.rows_per_week,args.restricted_root,args.checkpoint)
    print(json.dumps(result,sort_keys=True))


if __name__ == "__main__":
    main()
