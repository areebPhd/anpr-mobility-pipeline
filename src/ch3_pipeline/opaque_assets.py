from __future__ import annotations

import argparse
import csv
import math
import os
from collections import Counter
from pathlib import Path


MAGIC = {b"\x1f\x8b": "compressed:gzip", b"PK\x03\x04": "compressed:zip", b"BZh": "compressed:bzip2", b"\xfd7zXZ": "compressed:xz", b"RDX": "r-serialization"}


def entropy(data: bytes) -> float:
    if not data: return 0.0
    counts = Counter(data); n = len(data)
    return -sum((count/n) * math.log2(count/n) for count in counts.values())


def classify(path: Path) -> dict[str, object]:
    info = path.stat()
    readable = os.access(path, os.R_OK)
    head = b""
    if readable:
        try:
            with path.open("rb") as handle: head = handle.read(4096)
        except OSError: readable = False
    signature = next((label for magic, label in MAGIC.items() if head.startswith(magic)), "none")
    ent = entropy(head)
    suffix = path.suffix.lower()
    if not readable: category = "permission-protected"
    elif ".temp" in path.parts or path.name.startswith(".") and ("partial" in path.name or path.name.endswith(".zixSPx")): category = "temporary/partial transfer"
    elif signature.startswith("compressed"): category = "compressed"
    elif ent > 7.7 and signature == "none": category = "encrypted-or-unknown-high-entropy"
    elif b"\x00" in head: category = "unknown binary"
    else: category = "extension-mismatch-or-readable"
    likely_required = suffix in {".csv", ".geojson", ".rds"} and any(p in path.parts for p in ("cameras", "network", "anpr-03-trips"))
    return {"path": str(path), "extension": suffix, "size_bytes": info.st_size, "permissions": oct(info.st_mode & 0o777),
            "readable": readable, "signature": signature, "bounded_header_bytes": len(head), "bounded_entropy": round(ent, 4),
            "classification": category, "likely_required_dataset": likely_required,
            "note": "Encrypted is not proven; no opaque asset is accepted as valid data."}


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--root", default="data/input"); parser.add_argument("--output", default="reports/opaque_assets_classification.csv")
    args = parser.parse_args(); rows = []
    for path in sorted(Path(args.root).rglob("*")):
        if path.is_file() and (path.suffix.lower() in {".csv", ".geojson", ".rds", ".sh"} or path.name.startswith(".")):
            row = classify(path)
            if row["classification"] != "extension-mismatch-or-readable" or "w36" in path.name or row["likely_required_dataset"]: rows.append(row)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, rows[0].keys()); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__": main()
