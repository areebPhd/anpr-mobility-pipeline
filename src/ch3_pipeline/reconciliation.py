from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Reconciliation:
    input_rows: int
    retained_rows: int
    rejected_rows: int

    def assert_balanced(self) -> None:
        if self.input_rows != self.retained_rows + self.rejected_rows:
            raise AssertionError(
                f"row reconciliation failed: input={self.input_rows}, "
                f"retained={self.retained_rows}, rejected={self.rejected_rows}"
            )
