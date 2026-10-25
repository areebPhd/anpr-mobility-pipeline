from __future__ import annotations

import logging
import re
from typing import Any


SENSITIVE_KEYS = {"vehicle", "vehicle_token", "token", "registration", "vrm"}
TOKEN_PATTERN = re.compile(r"(?i)\b(vehicle(?:_token)?|token|registration|vrm)\s*[=:]\s*[^\s,;]+")


def redact(value: Any) -> str:
    text = str(value)
    return TOKEN_PATTERN.sub(lambda m: f"{m.group(1)}=[REDACTED]", text)


class PrivacyFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.msg = redact(record.getMessage())
        record.args = ()
        return True


def safe_fields(**fields: Any) -> dict[str, Any]:
    return {key: "[REDACTED]" if key.lower() in SENSITIVE_KEYS else value for key, value in fields.items()}
