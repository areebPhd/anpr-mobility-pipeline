from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from charset_normalizer import from_bytes


EXPECTED_FIELDS = ("vehicle", "camera", "timestamp", "confidence")


@dataclass(frozen=True)
class SchemaResult:
    valid: bool
    header: tuple[str, ...]
    encoding: str
    delimiter: str
    physical_types: dict[str, str]
    logical_types: dict[str, str]
    malformed_rows: int
    sampled_rows: int


def inspect_schema(path: str | Path, sample_bytes: int = 4 * 1024 * 1024) -> SchemaResult:
    path = Path(path)
    # Do not use Path.read_bytes() here: it materialises the complete archive
    # shard before slicing and defeats bounded development profiling.
    with path.open("rb") as handle:
        raw = handle.read(sample_bytes)
    match = from_bytes(raw).best()
    encoding = match.encoding if match else "unknown"
    text = str(match) if match else raw.decode("utf-8", errors="replace")
    first_line = text.splitlines()[0] if text.splitlines() else ""
    try:
        dialect = csv.Sniffer().sniff(first_line, delimiters=",;\t|")
        delimiter = dialect.delimiter
    except csv.Error:
        delimiter = "unknown"
    rows = csv.reader(text.splitlines(), delimiter=delimiter if delimiter != "unknown" else ",")
    header = tuple(next(rows, []))
    malformed = sampled = 0
    confidence_numeric = True
    timestamps_shape = True
    for row in rows:
        sampled += 1
        if len(row) != 4:
            malformed += 1
            continue
        try:
            float(row[3])
        except (TypeError, ValueError):
            confidence_numeric = False
        timestamps_shape &= len(row[2]) >= 19 and row[2][4:5] == "-" and row[2][10:11] in {" ", "T"}
    return SchemaResult(
        valid=header == EXPECTED_FIELDS and delimiter == "," and encoding.lower().replace("_", "-") in {"ascii", "utf-8"},
        header=header,
        encoding=encoding,
        delimiter=delimiter,
        physical_types={"vehicle": "text", "camera": "text", "timestamp": "text", "confidence": "numeric-text" if confidence_numeric else "mixed"},
        logical_types={"vehicle": "restricted-string", "camera": "categorical-string", "timestamp": "naive-datetime" if timestamps_shape else "unresolved-string", "confidence": "float" if confidence_numeric else "unresolved"},
        malformed_rows=malformed,
        sampled_rows=sampled,
    )


def require_header(header: tuple[str, ...]) -> None:
    if header != EXPECTED_FIELDS:
        raise ValueError(f"schema mismatch: expected {EXPECTED_FIELDS}; received field count={len(header)}")
