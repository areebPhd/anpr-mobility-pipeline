from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import stat
from pathlib import Path

from .config import PipelineConfig
from .schema import inspect_schema


def count_rows(path: Path, block_size: int = 16 * 1024 * 1024) -> int:
    lines = 0
    with path.open("rb") as handle:
        while block := handle.read(block_size):
            lines += block.count(b"\n")
    return max(0, lines - 1)


def checksum(path: Path, enabled: bool = False) -> tuple[str, str]:
    if not enabled:
        return "deferred", ""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(16 * 1024 * 1024):
            digest.update(block)
    return "complete", digest.hexdigest()


def build_manifest(config: PipelineConfig, output: Path, exact_counts: bool = False, checksums: bool = False) -> list[dict[str, object]]:
    records = []
    for path in config.source_files():
        week_match = re.search(r"w(\d{2})", path.name)
        record: dict[str, object] = {"path": str(path), "week": int(week_match.group(1)) if week_match else ""}
        try:
            info = path.stat()
            schema = inspect_schema(path)
            checksum_status, sha256 = checksum(path, checksums)
            record.update(file_size=info.st_size, permissions=stat.filemode(info.st_mode), header=",".join(schema.header),
                          exact_row_count=count_rows(path) if exact_counts else "verified-externally-pending-file-level-import",
                          modification_time=info.st_mtime, status="readable" if os.access(path, os.R_OK) else "inaccessible",
                          checksum_status=checksum_status, sha256=sha256)
        except (OSError, UnicodeError) as exc:
            record.update(file_size="", permissions="", header="", exact_row_count="", modification_time="",
                          status="inaccessible", checksum_status="not-computable", sha256="", error_type=type(exc).__name__)
        records.append(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["path", "week", "file_size", "permissions", "header", "exact_row_count", "modification_time", "status", "checksum_status", "sha256", "error_type"]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(records)
    return records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/chapter3_config.yaml")
    parser.add_argument("--exact-counts", action="store_true")
    parser.add_argument("--checksums", action="store_true")
    args = parser.parse_args()
    cfg = PipelineConfig.load(args.config)
    build_manifest(cfg, cfg.path("outputs", "manifest"), args.exact_counts, args.checksums)


if __name__ == "__main__":
    main()
