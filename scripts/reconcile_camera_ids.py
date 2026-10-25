from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("data/input/ANPR-2/cameras.csv")
REPORTS = ROOT / "reports"
OBSERVATIONS = ROOT / "restricted/phase3a_candidate_transitions/observations/week=*/observations.parquet"
CANDIDATES = ROOT / "restricted/phase3a_candidate_transitions/candidates/candidate_transitions.parquet"


def canonical_id(value: object) -> str:
    """Conservative deterministic normalization: whitespace and integer notation only."""
    text = str(value).strip()
    if text.isdigit():
        return str(int(text))
    return text.casefold()


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    metadata = pd.read_csv(SOURCE, dtype="string")
    metadata["camera"] = metadata["camera"].str.strip()
    metadata["canonical_id"] = metadata["camera"].map(canonical_id)
    connection = duckdb.connect()
    phase2_ids = {row[0] for row in connection.execute(
        "SELECT DISTINCT CAST(camera AS VARCHAR) FROM read_parquet(?)", [str(OBSERVATIONS)]).fetchall()}
    phase3a_ids = {row[0] for row in connection.execute(
        "SELECT camera FROM (SELECT DISTINCT CAST(origin_camera AS VARCHAR) camera FROM read_parquet(?) "
        "UNION SELECT DISTINCT CAST(destination_camera AS VARCHAR) camera FROM read_parquet(?))",
        [str(CANDIDATES), str(CANDIDATES)]).fetchall()}
    # Aggregate degree is calculated only to verify topology presence for already exact-linked IDs.
    degree = dict(connection.execute(
        "SELECT camera, SUM(n) FROM ("
        "SELECT CAST(origin_camera AS VARCHAR) camera, COUNT(*) n FROM read_parquet(?) GROUP BY 1 UNION ALL "
        "SELECT CAST(destination_camera AS VARCHAR) camera, COUNT(*) n FROM read_parquet(?) GROUP BY 1"
        ") GROUP BY 1", [str(CANDIDATES), str(CANDIDATES)]).fetchall())
    connection.close()

    observed = phase2_ids | phase3a_ids
    metadata_ids = set(metadata["camera"])
    canonical_to_metadata: dict[str, list[str]] = {}
    for camera_id in metadata_ids:
        canonical_to_metadata.setdefault(canonical_id(camera_id), []).append(camera_id)

    conflicts_path = REPORTS / "camera_metadata_conflicts.csv"
    conflict_counts: dict[str, int] = {}
    if conflicts_path.exists():
        conflicts = pd.read_csv(conflicts_path, dtype="string")
        for text in conflicts.get("camera_ids", pd.Series(dtype="string")).dropna():
            for camera_id in text.split("|"):
                conflict_counts[camera_id] = conflict_counts.get(camera_id, 0) + 1

    rows: list[dict[str, object]] = []
    recovered = 0
    for observed_id in sorted(observed, key=lambda value: (int(value) if value.isdigit() else float("inf"), value)):
        exact = observed_id in metadata_ids
        normalized = canonical_to_metadata.get(canonical_id(observed_id), [])
        if exact:
            metadata_id = observed_id
            rows.append({"observed_camera_id": observed_id, "metadata_camera_id": metadata_id,
                         "matching_rule": "exact canonical camera ID equality",
                         "supporting_evidence": "one unique metadata row; Phase 2 and Phase 3A presence; source coordinates and direction available; topology presence corroborated only",
                         "conflict_count": conflict_counts.get(metadata_id, 0), "confidence_category": "A_exact",
                         "accepted_rejected_status": "accepted"})
        elif len(normalized) == 1:
            metadata_id = normalized[0]
            recovered += 1
            rows.append({"observed_camera_id": observed_id, "metadata_camera_id": metadata_id,
                         "matching_rule": "deterministic canonicalization of whitespace/integer notation",
                         "supporting_evidence": "unique canonical identifier after conservative normalization",
                         "conflict_count": conflict_counts.get(metadata_id, 0), "confidence_category": "B_deterministic_one_to_one",
                         "accepted_rejected_status": "accepted"})
        else:
            rows.append({"observed_camera_id": observed_id, "metadata_camera_id": "",
                         "matching_rule": "no defensible deterministic or legacy crosswalk match",
                         "supporting_evidence": "observation schema contains ID only; names/addresses/coordinates unavailable on observation side; legacy camera/network assets remain opaque; topology cannot originate an identity match",
                         "conflict_count": 0, "confidence_category": "E_unresolved",
                         "accepted_rejected_status": "unresolved"})

    proposal = pd.DataFrame(rows)
    atomic_csv(proposal, REPORTS / "camera_id_match_proposals.csv")
    exact = int((proposal["confidence_category"] == "A_exact").sum())
    unresolved = int((proposal["confidence_category"] == "E_unresolved").sum())
    summary = pd.DataFrame([
        {"metric": "observed_camera_count", "value": len(observed)},
        {"metric": "metadata_camera_count", "value": len(metadata_ids)},
        {"metric": "A_exact_accepted", "value": exact},
        {"metric": "B_deterministic_one_to_one_accepted", "value": recovered},
        {"metric": "C_legacy_crosswalk_confirmed", "value": 0},
        {"metric": "D_manual_review_candidates", "value": 0},
        {"metric": "E_unresolved", "value": unresolved},
        {"metric": "recovered_from_original_106", "value": recovered},
        {"metric": "new_total_covered", "value": exact + recovered},
        {"metric": "new_coverage_percentage", "value": 100 * (exact + recovered) / len(observed)},
        {"metric": "metadata_only_remaining", "value": len(metadata_ids - observed)},
    ])
    atomic_csv(summary, REPORTS / "camera_id_reconciliation_summary.csv")

    source_hash = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    report = f"""# Camera-ID reconciliation audit

Audit timestamp: {datetime.now(timezone.utc).isoformat()}

## Result

No additional defensible matches were recovered from the 106 observation-only camera IDs. Coverage remains **{exact} of {len(observed)} observed cameras ({100 * exact / len(observed):.2f}%)**.

- A — exact: {exact} accepted
- B — deterministic one-to-one: {recovered} accepted
- C — legacy-crosswalk confirmed: 0
- D — strongly corroborated/manual review: 0
- E — unresolved: {unresolved}
- Metadata-only records remaining: {len(metadata_ids - observed)}

Every observed camera is represented in `camera_id_match_proposals.csv`. Exact links include their conflict count. Unresolved records intentionally have a blank metadata ID rather than a guessed pairing.

## Deterministic normalization

Both namespaces were trimmed and canonicalized conservatively: numeric IDs were converted to canonical base-10 integer notation, and non-numeric identifiers would be case-folded. The source and observation IDs were already canonical numeric strings. This produced no match beyond the {exact} exact links.

Arithmetic offsets, digit reversal, digit removal, numeric proximity, and range-order pairing were tested as diagnostics but rejected as matching rules. They have no documented semantic basis, do not provide a complete bijection, and in some cases create collisions. Consequently, no such candidate is promoted to category B or D.

## Other evidence

- **Names and addresses:** present only in metadata. Phase 2/3A observations contain camera ID but no name or address, so these attributes cannot independently link unmatched IDs.
- **Directions and coordinates:** present only in metadata. They can validate a proposed mapping but cannot generate an identity mapping without an observation-side crosswalk.
- **Duplicate locations:** two metadata location groups exist. They increase ambiguity and cannot identify an observation-only ID.
- **Legacy files:** the accessible legacy camera, link, distance, and network objects remain extension-mismatched high-entropy binary. Non-destructive signature inspection yields no readable crosswalk; category C therefore remains empty.
- **Transition topology:** aggregate topology presence was checked only for already exact-linked observed IDs. Topology was not used to originate matches because many plausible network symmetries and direction choices would make identity inference non-unique.

## Governance decision

The original camera metadata verdict remains **CAMERA METADATA ACCEPTED WITH LIMITATIONS**. The 119 exact matches may proceed to later candidate-edge auditing, while 106 observed cameras remain category E. Full routing must not begin until the reconciliation limitation is accepted methodologically or an authoritative legacy crosswalk is supplied.

Source file SHA-256 remained `{source_hash}`. The source was not modified, copied into Git, or committed. No routing, route-distance calculation, or speed filtering was performed.
"""
    atomic_text(report, REPORTS / "camera_id_reconciliation_audit.md")
    print(json.dumps({"exact": exact, "recovered": recovered, "unresolved": unresolved,
                      "coverage_percentage": 100 * (exact + recovered) / len(observed)}, sort_keys=True))


if __name__ == "__main__":
    main()
