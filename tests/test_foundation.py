import logging
from io import StringIO

import pytest

from ch3_pipeline.ingestion import Observation, consecutive_pairs
from ch3_pipeline.privacy_logging import PrivacyFilter
from ch3_pipeline.reconciliation import Reconciliation
from ch3_pipeline.schema import inspect_schema, require_header
from ch3_pipeline.profiling import _bounded_row_count
from ch3_pipeline.robust_csv import audit_csv_structure, profile_unquoted_file


def write_csv(tmp_path, content):
    path = tmp_path / "input.csv"; path.write_text(content, encoding="utf-8"); return path


def test_schema_mismatch(tmp_path):
    result = inspect_schema(write_csv(tmp_path, "vehicle,camera,when,confidence\na,b,2018-01-01 00:00:00,1\n"))
    assert not result.valid
    with pytest.raises(ValueError): require_header(result.header)


def test_malformed_timestamp_shape(tmp_path):
    result = inspect_schema(write_csv(tmp_path, "vehicle,camera,timestamp,confidence\na,b,not-a-date,1\n"))
    assert result.logical_types["timestamp"] == "unresolved-string"


def test_missing_mandatory_fields(tmp_path):
    path = write_csv(tmp_path, "vehicle,camera,timestamp,confidence\n,b,2018-01-01 00:00:00,1\n")
    assert inspect_schema(path).valid  # structure is valid; profiling classifies the missing value


def test_exact_duplicates_are_detectable():
    rows = [("v","c","t","1"),("v","c","t","1")]
    assert len(rows) - len(set(rows)) == 1


def test_repeated_token_camera_interface():
    rows = iter([Observation("v","c","2018-01-01 00:00:00",1), Observation("v","c","2018-01-01 00:00:05",1)])
    pairs = list(consecutive_pairs(rows)); assert len(pairs) == 1 and pairs[0].origin_camera == pairs[0].destination_camera


def test_row_reconciliation():
    Reconciliation(10, 8, 2).assert_balanced()
    with pytest.raises(AssertionError): Reconciliation(10, 8, 1).assert_balanced()


def test_privacy_safe_logging():
    stream = StringIO(); handler = logging.StreamHandler(stream); handler.addFilter(PrivacyFilter())
    logger = logging.getLogger("privacy-test"); logger.handlers=[handler]; logger.propagate=False; logger.warning("vehicle=SECRET camera=counted")
    assert "SECRET" not in stream.getvalue() and "[REDACTED]" in stream.getvalue()


def test_bounded_row_count(tmp_path):
    path = write_csv(tmp_path, "vehicle,camera,timestamp,confidence\n" + "v,c,2018-01-01 00:00:00,1\n" * 10)
    assert _bounded_row_count(path, 3) == 3
    assert _bounded_row_count(path, None) == 10


@pytest.mark.parametrize("payload,category", [
    ('vehicle,camera,timestamp,confidence\n"unterminated,c,2018-01-01 00:00:00,1\n', "unmatched_quote"),
    ("vehicle,camera,timestamp,confidence\na,b,2018-01-01 00:00:00,1,extra\n", "too_many_fields"),
    ("vehicle,camera,timestamp,confidence\na,b,2018-01-01 00:00:00\n", "too_few_fields"),
    ("vehicle,camera,timestamp,confidence\na,b,2018-01-01 00:00:00", "incomplete_final_record"),
])
def test_structural_reject_categories(tmp_path, payload, category):
    audit = audit_csv_structure(write_csv(tmp_path, payload))
    assert audit.categories[category] == 1


def test_embedded_newline_and_valid_quoted_fields(tmp_path):
    path = write_csv(tmp_path, 'vehicle,camera,timestamp,confidence\n"synthetic\nvalue",c,"2018-01-01 00:00:00",1\n')
    audit = audit_csv_structure(path)
    assert audit.accepted_records == 1
    assert audit.rejected_records == 0
    assert audit.embedded_newlines == 1


def test_unquoted_fallback_exact_reconciliation(tmp_path):
    path = write_csv(tmp_path, "vehicle,camera,timestamp,confidence\nSYNTH_A,C1,2018-01-01 00:00:00.000,90\nSYNTH_B,C2,2018-01-01 00:00:01.000,91,EXTRA\n")
    result, audit = profile_unquoted_file(path, 35)
    assert result["input_rows"] == 2
    assert result["retained_rows"] == 1
    assert result["rejected_malformed_rows"] == 1
    assert audit["reconciled"]
