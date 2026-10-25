from __future__ import annotations

import csv
import hashlib
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


EXPECTED_HEADER = b"vehicle,camera,timestamp,confidence"


@dataclass(frozen=True)
class StructuralAudit:
    physical_records: int
    accepted_records: int
    rejected_records: int
    categories: dict[str, int]
    embedded_newlines: int


def audit_csv_structure(path: str | Path, expected_fields: int = 4) -> StructuralAudit:
    """RFC-style structural audit for small controls; never returns field values."""
    path = Path(path)
    categories: Counter[str] = Counter()
    physical_records = accepted = embedded = 0
    previous_line = 1
    try:
        with path.open("r", encoding="utf-8", errors="strict", newline="") as handle:
            reader = csv.reader(handle, strict=True)
            next(reader, None)
            previous_line = reader.line_num
            for row in reader:
                physical_records += 1
                embedded += max(0, reader.line_num - previous_line - 1)
                previous_line = reader.line_num
                if len(row) < expected_fields:
                    categories["too_few_fields"] += 1
                elif len(row) > expected_fields:
                    categories["too_many_fields"] += 1
                else:
                    accepted += 1
    except UnicodeDecodeError:
        categories["invalid_byte_sequence"] += 1
        physical_records += 1
    except csv.Error as exc:
        categories["unmatched_quote" if "unexpected end" in str(exc).lower() else "csv_syntax_error"] += 1
        physical_records += 1
    final_newline = True
    if path.stat().st_size:
        with path.open("rb") as handle:
            handle.seek(-1, 2)
            final_newline = handle.read(1) == b"\n"
    if path.stat().st_size and not final_newline and not categories.get("unmatched_quote"):
        categories["incomplete_final_record"] += 1
        if accepted:
            accepted -= 1
        else:
            physical_records += 1
    rejected = sum(categories.values())
    return StructuralAudit(physical_records, accepted, rejected, dict(categories), embedded)


def _quantile(histogram: Counter[float], probability: float) -> float | None:
    total = sum(histogram.values())
    if not total:
        return None
    target = probability * (total - 1)
    ranks = (int(target), min(int(target) + 1, total - 1))
    found: list[float] = []
    cumulative = 0
    for value, count in sorted(histogram.items()):
        prior = cumulative
        cumulative += count
        for index, rank in enumerate(ranks):
            if len(found) <= index and prior <= rank < cumulative:
                found.append(value)
    if len(found) == 1:
        found.append(found[0])
    return found[0] + (target - ranks[0]) * (found[1] - found[0])


def profile_unquoted_file(path: str | Path, week: int) -> tuple[dict[str, Any], dict[str, Any]]:
    """Profile verified-unquoted data and classify every physical fragment once."""
    path = Path(path)
    started = time.monotonic()
    checksum = hashlib.sha256()
    categories: Counter[str] = Counter()
    summaries: dict[str, dict[str, Any]] = {}
    evidence: dict[str, list[dict[str, Any]]] = defaultdict(list)
    input_rows = retained = timestamp_parsed = 0
    missing_vehicle = missing_camera = missing_timestamp = missing_confidence = 0
    camera_digests: set[bytes] = set()
    exact_seen: set[bytes] = set()
    key_seen: set[bytes] = set()
    exact_duplicates = key_duplicates = 0
    confidence_histogram: Counter[float] = Counter()
    confidence_sum = 0.0
    earliest = latest = None
    byte_offset = 0

    def reject(category: str, line_number: int, start: int, field_count: int, shape: str) -> None:
        categories[category] += 1
        summary = summaries.setdefault(category, {"error_category": category, "rejected_count": 0,
            "first_record_number": line_number - 1, "last_record_number": line_number - 1,
            "first_byte_offset": start, "last_byte_offset": start})
        summary["rejected_count"] += 1
        summary["last_record_number"] = line_number - 1
        summary["last_byte_offset"] = start
        if len(evidence[category]) < 5:
            evidence[category].append({"record_number": line_number - 1, "physical_line": line_number,
                "byte_offset": start, "field_count": field_count, "error_category": category,
                "structural_shape": shape})

    with path.open("rb") as handle:
        header = handle.readline()
        checksum.update(header)
        if header.rstrip(b"\r\n") != EXPECTED_HEADER:
            raise ValueError("fallback header mismatch")
        byte_offset += len(header)
        line_number = 1
        while raw := handle.readline():
            line_number += 1
            input_rows += 1
            start = byte_offset
            byte_offset += len(raw)
            checksum.update(raw)
            terminated = raw.endswith(b"\n")
            body = raw[:-1] if terminated else raw
            if body.endswith(b"\r"):
                body = body[:-1]
            field_count = body.count(b",") + 1
            quote_count = body.count(bytes((34,)))
            shape = f"fields={field_count};terminated={terminated};quotes={quote_count}"
            if not terminated:
                reject("incomplete_final_record", line_number, start, field_count, shape)
                continue
            if b'"' in body:
                reject("quote_in_verified_unquoted_record", line_number, start, field_count, shape)
                continue
            try:
                text = body.decode("utf-8", errors="strict")
            except UnicodeDecodeError:
                reject("invalid_byte_sequence", line_number, start, field_count, shape)
                continue
            fields = text.split(",")
            if len(fields) < 4:
                reject("too_few_fields", line_number, start, len(fields), shape)
                continue
            if len(fields) > 4:
                reject("too_many_fields", line_number, start, len(fields), shape)
                continue
            vehicle, camera, timestamp, confidence = fields
            parsed_timestamp = None
            if timestamp:
                try:
                    parsed_timestamp = datetime.fromisoformat(timestamp)
                except ValueError:
                    reject("invalid_timestamp", line_number, start, 4, shape)
                    continue
            parsed_confidence = None
            if confidence:
                try:
                    parsed_confidence = float(confidence)
                except ValueError:
                    reject("invalid_confidence", line_number, start, 4, shape)
                    continue
            retained += 1
            missing_vehicle += not bool(vehicle)
            missing_camera += not bool(camera)
            missing_timestamp += not bool(timestamp)
            missing_confidence += not bool(confidence)
            if parsed_timestamp is not None:
                timestamp_parsed += 1
                earliest = timestamp if earliest is None or timestamp < earliest else earliest
                latest = timestamp if latest is None or timestamp > latest else latest
            if parsed_confidence is not None:
                confidence_histogram[parsed_confidence] += 1
                confidence_sum += parsed_confidence
            camera_digests.add(hashlib.blake2b(camera.encode(), digest_size=16, person=b"ch3-camera").digest())
            exact_digest = hashlib.blake2b(body, digest_size=16, person=b"ch3-exact").digest()
            key_digest = hashlib.blake2b("\x1f".join(fields[:3]).encode(), digest_size=16, person=b"ch3-key").digest()
            if exact_digest in exact_seen:
                exact_duplicates += 1
            else:
                exact_seen.add(exact_digest)
            if key_digest in key_seen:
                key_duplicates += 1
            else:
                key_seen.add(key_digest)

    rejected = sum(categories.values())
    if input_rows != retained + rejected:
        raise AssertionError(f"fallback reconciliation failed: input={input_rows}, retained={retained}, rejected={rejected}")
    confidence_count = sum(confidence_histogram.values())
    result = {
        "retained_rows": retained, "missing_vehicle": missing_vehicle, "missing_camera": missing_camera,
        "missing_timestamp": missing_timestamp, "missing_confidence": missing_confidence,
        "timestamp_parsed": timestamp_parsed, "earliest_timestamp": earliest, "latest_timestamp": latest,
        "camera_count": len(camera_digests), "confidence_min": min(confidence_histogram) if confidence_histogram else None,
        "confidence_max": max(confidence_histogram) if confidence_histogram else None,
        "confidence_mean": confidence_sum / confidence_count if confidence_count else None,
        "confidence_median": _quantile(confidence_histogram, .5), "confidence_q01": _quantile(confidence_histogram, .01),
        "confidence_q05": _quantile(confidence_histogram, .05), "confidence_q25": _quantile(confidence_histogram, .25),
        "confidence_q75": _quantile(confidence_histogram, .75), "confidence_q95": _quantile(confidence_histogram, .95),
        "confidence_q99": _quantile(confidence_histogram, .99), "week": week, "path": str(path),
        "input_rows": input_rows, "rejected_malformed_rows": rejected,
        "timestamp_success_rate": timestamp_parsed / retained if retained else 0,
        "exact_duplicate_count": exact_duplicates, "token_camera_timestamp_duplicate_count": key_duplicates,
    }
    audit = {"sha256": checksum.hexdigest(), "size_bytes": path.stat().st_size,
        "physical_data_records": input_rows, "retained_records": retained, "rejected_records": rejected,
        "categories": dict(categories), "summaries": list(summaries.values()),
        "evidence": [item for items in evidence.values() for item in items],
        "reconciled": input_rows == retained + rejected,
        "runtime_seconds": round(time.monotonic() - started, 3),
        "record_policy": "LF-delimited; quotes forbidden by verified clean archive structure; final non-LF fragment rejected"}
    return result, audit
