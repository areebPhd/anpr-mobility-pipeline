from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class PipelineConfig:
    raw: dict[str, Any]
    project_root: Path

    @classmethod
    def load(cls, path: str | Path) -> "PipelineConfig":
        path = Path(path).resolve()
        with path.open(encoding="utf-8") as handle:
            raw = yaml.safe_load(handle)
        return cls(raw=raw, project_root=path.parent.parent)

    def path(self, *keys: str) -> Path:
        value: Any = self.raw
        for key in keys:
            value = value[key]
        result = Path(value)
        return result if result.is_absolute() else self.project_root / result

    @property
    def weeks(self) -> list[int]:
        return list(self.raw["source"]["readable_weeks"])

    def source_files(self) -> list[Path]:
        pattern = self.raw["source"]["readable_glob"]
        return [Path(pattern.replace("{01..35}", f"{week:02d}")) for week in self.weeks]

    def assert_unresolved_decisions(self) -> None:
        method = self.raw["methodology"]
        for name in ("timestamp_timezone", "confidence_threshold", "duplicate_window"):
            if method.get(name) != "unresolved":
                raise ValueError(f"Methodological decision must remain unresolved: {name}")
