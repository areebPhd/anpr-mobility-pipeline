from __future__ import annotations

from datetime import datetime, timedelta

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

from ch3_pipeline.phase3a import OBS_SCHEMA, build_ordered_and_transitions, token_persistence


def write_observations(tmp_path, rows):
    table = pa.Table.from_pylist(rows, schema=OBS_SCHEMA)
    path = tmp_path / "observations.parquet"
    pq.write_table(table, path)
    return path


def row(vehicle, camera, timestamp, confidence, week, number):
    return {"vehicle": vehicle, "camera": camera, "timestamp": timestamp, "confidence": confidence,
            "source_week": week, "source_record_number": number, "source_file": f"synthetic-w{week:02d}.csv"}


def test_ordering_flags_and_transition_reconciliation(tmp_path):
    t0 = datetime(2018, 1, 1)
    source = write_observations(tmp_path, [
        row("SYNTH_A", "C2", t0 + timedelta(seconds=5), 90, 1, 3),  # unordered input
        row("SYNTH_A", "C1", t0, 91, 1, 1),
        row("SYNTH_A", "C1", t0, 91, 1, 2),  # equal time, exact duplicate, same camera
        row("SYNTH_SINGLE", "C9", t0, 80, 1, 4),  # one-observation sequence
        row("SYNTH_CROSS", "C3", t0, 85, 1, 5),
        row("SYNTH_CROSS", "C4", t0 + timedelta(seconds=1), 86, 2, 1),
    ])
    con = duckdb.connect()
    ordered = tmp_path / "ordered.parquet"; transitions = tmp_path / "transitions.parquet"
    result = build_ordered_and_transitions(con, str(source), ordered, transitions)
    assert result["ordered_observations"] == 6
    assert result["vehicle_sequences"] == 3
    assert result["candidate_transitions"] == 3
    assert result["candidate_transitions"] == result["ordered_observations"] - result["vehicle_sequences"]
    flags = con.execute(f"SELECT sum(exact_duplicate::INT),sum(same_camera::INT),sum(identical_timestamp::INT),sum(cross_week::INT),sum(negative_duration::INT) FROM read_parquet('{transitions}')").fetchone()
    assert flags == (1, 1, 1, 1, 0)


def test_negative_input_order_becomes_nonnegative_after_sort(tmp_path):
    t0 = datetime(2018, 1, 1)
    source = write_observations(tmp_path, [row("SYNTH", "LATE", t0 + timedelta(seconds=10), 1, 1, 1), row("SYNTH", "EARLY", t0, 1, 1, 2)])
    con = duckdb.connect(); ordered = tmp_path/"o.parquet"; transitions = tmp_path/"t.parquet"
    build_ordered_and_transitions(con, str(source), ordered, transitions)
    elapsed, negative = con.execute(f"SELECT elapsed_seconds,negative_duration FROM read_parquet('{transitions}')").fetchone()
    assert elapsed == 10 and not negative


def test_possible_token_rotation_is_aggregate_and_private(tmp_path):
    t0 = datetime(2018, 1, 1)
    source = write_observations(tmp_path, [row("W1_ONLY", "C1", t0, 1, 1, 1), row("W2_ONLY", "C1", t0, 1, 2, 1)])
    con = duckdb.connect(); csv_path=tmp_path/"audit.csv"; md_path=tmp_path/"audit.md"
    rows = token_persistence(con, str(source), csv_path, md_path)
    adjacent = next(item for item in rows if item["scope"] == "adjacent_weeks")
    assert adjacent["overlap_count"] == 0
    evidence = csv_path.read_text() + md_path.read_text()
    assert "W1_ONLY" not in evidence and "W2_ONLY" not in evidence


def test_transition_output_contains_irreversible_key_not_token(tmp_path):
    t0 = datetime(2018, 1, 1)
    source = write_observations(tmp_path, [row("DO_NOT_EXPORT", "C1", t0, 1, 1, 1), row("DO_NOT_EXPORT", "C2", t0 + timedelta(seconds=1), 1, 1, 2)])
    con = duckdb.connect(); ordered=tmp_path/"o.parquet"; transitions=tmp_path/"t.parquet"
    build_ordered_and_transitions(con, str(source), ordered, transitions)
    names = pq.read_schema(transitions).names
    assert "sequence_key" in names and "vehicle" not in names
    key = con.execute(f"SELECT sequence_key FROM read_parquet('{transitions}')").fetchone()[0]
    assert key != "DO_NOT_EXPORT" and len(key) == 64
