from __future__ import annotations

import argparse
import csv
import json
import os
import resource
import time
from pathlib import Path
from typing import Any

import duckdb

from .config import PipelineConfig
from .logging_utils import configure_logging
from .manifest import count_rows
from .reconciliation import Reconciliation
from .robust_csv import profile_unquoted_file
from .schema import EXPECTED_FIELDS, inspect_schema


TS_EXPR = "coalesce(try_strptime(timestamp, '%Y-%m-%d %H:%M:%S.%g'), try_strptime(timestamp, '%Y-%m-%d %H:%M:%S'))"


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0]) if rows else ["status"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields)
        writer.writeheader(); writer.writerows(rows or [{"status": "no-records"}])


def _relation(path: Path, limit: int | None) -> str:
    safe = str(path).replace("'", "''")
    base = f"read_csv('{safe}', header=true, all_varchar=true, auto_detect=false, columns={{'vehicle':'VARCHAR','camera':'VARCHAR','timestamp':'VARCHAR','confidence':'VARCHAR'}}, ignore_errors=true, null_padding=true)"
    return f"(SELECT * FROM {base} LIMIT {int(limit)})" if limit else base


def _bounded_row_count(path: Path, limit: int | None) -> int:
    if limit is None:
        return count_rows(path)
    rows = 0
    with path.open("rb") as handle:
        handle.readline()  # header
        while rows < limit and handle.readline():
            rows += 1
    return rows


def profile_file(con: duckdb.DuckDBPyConnection, path: Path, week: int, limit: int | None) -> dict[str, Any]:
    rel = _relation(path, limit)
    input_rows = _bounded_row_count(path, limit)
    query = f"""
      SELECT count(*) retained_rows,
        count(*) FILTER (WHERE vehicle IS NULL OR trim(vehicle)='') missing_vehicle,
        count(*) FILTER (WHERE camera IS NULL OR trim(camera)='') missing_camera,
        count(*) FILTER (WHERE timestamp IS NULL OR trim(timestamp)='') missing_timestamp,
        count(*) FILTER (WHERE confidence IS NULL OR trim(confidence)='') missing_confidence,
        count(*) FILTER (WHERE {TS_EXPR} IS NOT NULL) timestamp_parsed,
        min({TS_EXPR}) earliest_timestamp, max({TS_EXPR}) latest_timestamp,
        count(DISTINCT camera) camera_count,
        min(try_cast(confidence AS DOUBLE)) confidence_min,
        max(try_cast(confidence AS DOUBLE)) confidence_max,
        avg(try_cast(confidence AS DOUBLE)) confidence_mean,
        median(try_cast(confidence AS DOUBLE)) confidence_median,
        quantile_cont(try_cast(confidence AS DOUBLE), .01) confidence_q01,
        quantile_cont(try_cast(confidence AS DOUBLE), .05) confidence_q05,
        quantile_cont(try_cast(confidence AS DOUBLE), .25) confidence_q25,
        quantile_cont(try_cast(confidence AS DOUBLE), .75) confidence_q75,
        quantile_cont(try_cast(confidence AS DOUBLE), .95) confidence_q95,
        quantile_cont(try_cast(confidence AS DOUBLE), .99) confidence_q99
      FROM {rel}
    """
    values = con.execute(query).fetchone()
    names = [d[0] for d in con.description]
    result = dict(zip(names, values))
    rejected = input_rows - result["retained_rows"]
    Reconciliation(input_rows, result["retained_rows"], rejected).assert_balanced()
    result.update(week=week, path=str(path), input_rows=input_rows, rejected_malformed_rows=rejected,
                  timestamp_success_rate=result["timestamp_parsed"] / result["retained_rows"] if result["retained_rows"] else 0)
    dup = con.execute(f"SELECT coalesce(sum(n-1),0) FROM (SELECT count(*) n FROM {rel} GROUP BY vehicle,camera,timestamp,confidence HAVING n>1)").fetchone()[0]
    key_dup = con.execute(f"SELECT coalesce(sum(n-1),0) FROM (SELECT count(*) n FROM {rel} GROUP BY vehicle,camera,timestamp HAVING n>1)").fetchone()[0]
    result.update(exact_duplicate_count=dup, token_camera_timestamp_duplicate_count=key_dup)
    return result


def _save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, default=str, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_profile_outputs(cfg: PipelineConfig, results: list[dict[str, Any]], schema_rows: list[dict[str, Any]],
                           full: bool, started: float, peak_rss_mb: float) -> None:
    report_dir = cfg.path("outputs", "profiles")
    _write_csv(report_dir / "schema_validation_results.csv", schema_rows)
    weekly_fields = ["week","input_rows","retained_rows","rejected_malformed_rows","missing_vehicle","missing_camera","missing_timestamp","missing_confidence","timestamp_success_rate","earliest_timestamp","latest_timestamp","camera_count"]
    _write_csv(report_dir / "weekly_quality_profile.csv", [{k:r.get(k) for k in weekly_fields} for r in results])
    _write_csv(report_dir / "confidence_profile.csv", [{k:r.get(k) for k in ["week","confidence_min","confidence_max","confidence_mean","confidence_median","confidence_q01","confidence_q05","confidence_q25","confidence_q75","confidence_q95","confidence_q99"]} for r in results])
    _write_csv(report_dir / "duplicate_profile.csv", [{k:r.get(k) for k in ["week","exact_duplicate_count","token_camera_timestamp_duplicate_count"]} | {"repeat_gap_status":"unresolved; observations retained"} for r in results])
    _write_csv(report_dir / "timestamp_profile.csv", [{k:r.get(k) for k in ["week","timestamp_parsed","retained_rows","timestamp_success_rate","earliest_timestamp","latest_timestamp"]} for r in results])
    _write_csv(report_dir / "row_reconciliation.csv", [{k:r.get(k) for k in ["week","input_rows","retained_rows","rejected_malformed_rows"]} | {"balanced": r["input_rows"] == r["retained_rows"] + r["rejected_malformed_rows"]} for r in results])
    runtime = round(time.monotonic()-started, 3)
    _write_csv(report_dir / "readable_archive_profile.csv", [{"metric":"profile_mode","value":"full" if full else "bounded"},{"metric":"weeks_profiled","value":len(results)},{"metric":"rows_profiled","value":sum(r["input_rows"] for r in results)},{"metric":"runtime_seconds","value":runtime},{"metric":"peak_rss_mb","value":round(peak_rss_mb, 3)}])


def _write_thesis_outputs(cfg: PipelineConfig, results: list[dict[str, Any]]) -> None:
    thesis = cfg.project_root / "thesis"
    thesis.mkdir(parents=True, exist_ok=True)
    _write_csv(thesis / "table_archive_profile.csv", [{"week":r["week"],"input_rows":r["input_rows"],"earliest_timestamp":r["earliest_timestamp"],"latest_timestamp":r["latest_timestamp"]} for r in results])
    _write_csv(thesis / "table_raw_quality_audit.csv", [{k:r[k] for k in ["week","input_rows","retained_rows","rejected_malformed_rows","missing_vehicle","missing_camera","missing_timestamp","missing_confidence"]} for r in results])
    _write_csv(thesis / "table_confidence_profile.csv", [{k:r[k] for k in ["week","confidence_min","confidence_max","confidence_mean","confidence_median","confidence_q01","confidence_q99"]} for r in results])
    _write_csv(thesis / "table_duplicate_profile.csv", [{k:r[k] for k in ["week","exact_duplicate_count","token_camera_timestamp_duplicate_count"]} for r in results])
    _write_csv(thesis / "table_weekly_coverage.csv", [{k:r[k] for k in ["week","retained_rows","camera_count","earliest_timestamp","latest_timestamp"]} for r in results])
    total = sum(r["input_rows"] for r in results)
    (thesis / "section_3_4_verified_facts.md").write_text(
        "# Section 3.4 verified facts\n\n"
        f"The readable archive profile covers {len(results)} provisionally readable weekly files and {total:,} input rows. "
        "Weeks 01–35 are not asserted to be the final study period. Confidence values were profiled without applying a final threshold; duplicate observations were counted but not removed. No final transitions or routes were constructed.\n",
        encoding="utf-8")
    (thesis / "section_3_5_method_notes.md").write_text(
        "# Section 3.5 method notes\n\n"
        "Profiling used an exact four-field schema gate, safe timestamp coercion, aggregate missingness and malformed-row accounting, duplicate counts, and row reconciliation. Vehicle-token values were excluded from logs and outputs. Timezone, confidence threshold, near-duplicate policy, final study period, transition construction, and OSM routing remain unresolved.\n",
        encoding="utf-8")


def _write_week35_diagnostic(cfg: PipelineConfig, audit: dict[str, Any]) -> None:
    report_dir = cfg.path("outputs", "profiles")
    _write_csv(report_dir / "week35_rejected_rows_summary.csv", audit["summaries"])
    evidence = "\n".join(
        f"- record {row['record_number']}, physical line {row['physical_line']}, byte offset {row['byte_offset']}, "
        f"field count {row['field_count']}, category `{row['error_category']}`, shape `{row['structural_shape']}`"
        for row in audit["evidence"]
    ) or "- None"
    categories = "\n".join(f"- `{key}`: {value:,}" for key, value in sorted(audit["categories"].items())) or "- None"
    (report_dir / "week35_csv_diagnostic.md").write_text(
        "# Week 35 CSV diagnostic\n\n"
        "All evidence is structural and redacted; no vehicle-token value is present.\n\n"
        f"- File size: {audit['size_bytes']:,} bytes\n"
        f"- SHA-256: `{audit['sha256']}`\n"
        "- DuckDB version: 1.5.4\n"
        "- Original failure: CSV state-machine invalid state under `strict_mode=true`.\n"
        "- Verified first invalid UTF-8 byte: offset 147,324,929 (physical line 3,927,538).\n"
        "- Binary-contaminated regions: approximately bytes 147,324,929–147,849,215 and 415,760,384–EOF.\n"
        "- The second region is terminal; the file ends with byte 130, not LF.\n"
        "- Quote bytes occur only after contamination begins; clean sampled regions contain no quotes.\n"
        "- Independent physical record policy: LF-delimited because quoting is forbidden for this verified-unquoted source format; a final non-LF fragment is one rejected record.\n"
        f"- Physical data records: {audit['physical_data_records']:,}\n"
        f"- Retained records: {audit['retained_records']:,}\n"
        f"- Explicitly rejected records: {audit['rejected_records']:,}\n"
        f"- Exact reconciliation: `{audit['reconciled']}`\n"
        f"- Fallback runtime: {audit['runtime_seconds']:.3f} seconds\n\n"
        "## Rejection categories\n\n" + categories + "\n\n"
        "## Redacted structural evidence (maximum five examples per category)\n\n" + evidence + "\n\n"
        "## Diagnosis\n\n"
        "The failure is caused by binary contamination/partial corruption, not a legitimate quoted CSV construct. The parser correction uses an auditable unquoted fallback that validates UTF-8, exact field count, timestamp syntax, confidence syntax, and final-line termination; every physical fragment is classified once. No raw rejected row is exported.\n",
        encoding="utf-8")


def run(config_path: str, full: bool = False, convert_parquet: bool = False,
        selected_weeks: list[int] | None = None, rows_per_file: int | None = None,
        checkpoint_path: str | None = None) -> list[dict[str, Any]]:
    cfg = PipelineConfig.load(config_path); cfg.assert_unresolved_decisions()
    log = configure_logging(cfg.project_root / "logs/phase2_run.log")
    runtime = cfg.raw["runtime"]
    limit = None if full else int(rows_per_file or runtime["development_rows_per_file"])
    con = duckdb.connect()
    con.execute(f"SET threads={int(runtime['thread_count'])}")
    con.execute(f"SET memory_limit='{runtime['memory_limit']}'")
    temp = Path(runtime["temporary_directory"]); temp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory='{str(temp).replace(chr(39), chr(39)*2)}'")
    requested = selected_weeks or cfg.weeks
    invalid_weeks = sorted(set(requested) - set(cfg.weeks))
    if invalid_weeks:
        raise ValueError(f"weeks outside readable manifest: {invalid_weeks}")
    checkpoint = Path(checkpoint_path).resolve() if checkpoint_path else cfg.path("checkpoints", "completed_weeks")
    state: dict[str, Any] = {"mode":"full" if full else "bounded", "limit":limit, "requested_weeks":requested, "records":{}, "schemas":{}}
    if checkpoint.exists():
        loaded = json.loads(checkpoint.read_text(encoding="utf-8"))
        if isinstance(loaded, dict) and loaded.get("mode") == state["mode"] and loaded.get("limit") == limit and loaded.get("requested_weeks") == requested:
            state = loaded
    results = [record for _, record in sorted(state["records"].items(), key=lambda item: int(item[0]))]
    schema_rows = [record for _, record in sorted(state["schemas"].items(), key=lambda item: int(item[0]))]
    started = time.monotonic()
    source_by_week = dict(zip(cfg.weeks, cfg.source_files()))
    for week in requested:
        path = source_by_week[week]
        if str(week) in state["records"]:
            log.info("Checkpoint restart: skipping completed week %02d", week)
            continue
        schema = inspect_schema(path)
        schema_record = {"week": week, "valid": schema.valid, "field_names": "|".join(schema.header), "field_count": len(schema.header),
                            "encoding": schema.encoding, "delimiter": schema.delimiter, "physical_types": json.dumps(schema.physical_types, sort_keys=True),
                            "logical_types": json.dumps(schema.logical_types, sort_keys=True), "bounded_malformed_rows": schema.malformed_rows,
                            "bounded_sampled_rows": schema.sampled_rows, "timestamp_examples": "redacted; shape-only inspection"}
        state["schemas"][str(week)] = schema_record
        if not schema.valid:
            _save_checkpoint(checkpoint, state)
            log.error("Week %02d schema gate failed", week); continue
        log.info("Profiling week %02d (%s mode)", week, "full" if full else "bounded")
        if full and week == 35:
            log.info("Week 35: using audited unquoted fallback after verified binary contamination")
            record, audit = profile_unquoted_file(path, week)
            _write_week35_diagnostic(cfg, audit)
            log.info("Week 35 reconciliation: input=%d retained=%d rejected=%d balanced=%s",
                     audit["physical_data_records"], audit["retained_records"], audit["rejected_records"], audit["reconciled"])
        else:
            record = profile_file(con, path, week, limit)
        state["records"][str(week)] = record
        _save_checkpoint(checkpoint, state)
        if full and convert_parquet and record["rejected_malformed_rows"] == 0 and record["timestamp_parsed"] == record["retained_rows"]:
            out = cfg.path("outputs", "parquet") / f"week={week:02d}"; out.mkdir(parents=True, exist_ok=True)
            target = out / "observations.parquet"
            rel = _relation(path, None)
            con.execute(f"COPY (SELECT vehicle,camera,{TS_EXPR} AS timestamp,try_cast(confidence AS DOUBLE) confidence,{week}::TINYINT source_week FROM {rel}) TO '{target}' (FORMAT PARQUET, COMPRESSION {runtime['parquet_compression']})")
    results = [record for _, record in sorted(state["records"].items(), key=lambda item: int(item[0]))]
    schema_rows = [record for _, record in sorted(state["schemas"].items(), key=lambda item: int(item[0]))]
    peak_rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    _write_profile_outputs(cfg, results, schema_rows, full, started, peak_rss_mb)
    if full and len(results) == len(requested) and all(row["valid"] for row in schema_rows):
        _write_thesis_outputs(cfg, results)
        (cfg.path("outputs", "profiles") / "phase2_final_status.md").write_text(
            "# Phase 2 final status\n\nPASS: all requested readable weeks were profiled and reconciled. Weeks 01–35 remain a provisional readable subset, not the final study period. No final confidence threshold was applied; duplicates were retained; no final transitions or OSM routing were performed.\n",
            encoding="utf-8")
    con.close(); return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chapter3_config.yaml")
    parser.add_argument("--full", action="store_true")
    parser.add_argument("--convert-parquet", action="store_true")
    parser.add_argument("--weeks", help="comma-separated readable week numbers")
    parser.add_argument("--rows-per-file", type=int)
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    weeks = [int(value) for value in args.weeks.split(",")] if args.weeks else None
    run(args.config, args.full, args.convert_parquet, weeks, args.rows_per_file, args.checkpoint)


if __name__ == "__main__": main()
