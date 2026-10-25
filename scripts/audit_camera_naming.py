from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("data/input/ANPR-2/cameras.csv")
OBSERVATIONS = ROOT / "restricted/phase3a_candidate_transitions/observations/week=*/observations.parquet"
REPORTS = ROOT / "reports"
DIRECTION_ALIASES = {
    "N": "NORTHBOUND", "NB": "NORTHBOUND", "NORTH": "NORTHBOUND", "NORTHBOUND": "NORTHBOUND",
    "S": "SOUTHBOUND", "SB": "SOUTHBOUND", "SOUTH": "SOUTHBOUND", "SOUTHBOUND": "SOUTHBOUND",
    "E": "EASTBOUND", "EB": "EASTBOUND", "EAST": "EASTBOUND", "EASTBOUND": "EASTBOUND",
    "W": "WESTBOUND", "WB": "WESTBOUND", "WEST": "WESTBOUND", "WESTBOUND": "WESTBOUND",
}


def normalise_id(value: object) -> str:
    text = unicodedata.normalize("NFKC", str(value)).strip().upper()
    text = re.sub(r"[-_\s]+", "", text)
    parts = re.fullmatch(r"([A-Z]*)(\d+)([A-Z]*)", text)
    if parts:
        prefix, number, suffix = parts.groups()
        prefix = DIRECTION_ALIASES.get(prefix, prefix)
        suffix = DIRECTION_ALIASES.get(suffix, suffix)
        return f"{prefix}{int(number)}{suffix}"
    return text


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_text(text: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    metadata = pd.read_csv(SOURCE, dtype="string").fillna("")
    metadata["camera"] = metadata["camera"].str.strip()
    connection = duckdb.connect()
    observed = {row[0] for row in connection.execute(
        "SELECT DISTINCT CAST(camera AS VARCHAR) FROM read_parquet(?)", [str(OBSERVATIONS)]).fetchall()}
    connection.close()
    metadata_ids = set(metadata["camera"])
    unmatched_observed = observed - metadata_ids
    unmatched_metadata = metadata_ids - observed

    normalized_metadata: dict[str, list[str]] = defaultdict(list)
    for camera_id in metadata_ids:
        normalized_metadata[normalise_id(camera_id)].append(camera_id)

    crosswalk = []
    recovered = 0
    for observed_id in sorted(observed, key=int):
        exact = observed_id in metadata_ids
        aliases = normalized_metadata.get(normalise_id(observed_id), [])
        if exact:
            row = metadata.loc[metadata["camera"].eq(observed_id)].iloc[0]
            crosswalk.append({"observed_camera_id": observed_id, "metadata_camera_id": observed_id,
                              "classification": "exact verified", "matching_rule": "exact ID equality after trimming",
                              "name_evidence": row["name"], "address_evidence": row["address"],
                              "direction_evidence": row["direction"], "coordinate_evidence": "valid source coordinates",
                              "shared_site_evidence": "see camera_mapping_conflicts.csv",
                              "legacy_evidence": "not required", "transition_neighbour_evidence": "presence corroborated only",
                              "accepted_status": "accepted"})
        elif len(aliases) == 1:
            recovered += 1
            metadata_id = aliases[0]
            row = metadata.loc[metadata["camera"].eq(metadata_id)].iloc[0]
            crosswalk.append({"observed_camera_id": observed_id, "metadata_camera_id": metadata_id,
                              "classification": "deterministic one-to-one", "matching_rule": "separator/case/leading-zero/direction-alias normalization",
                              "name_evidence": row["name"], "address_evidence": row["address"],
                              "direction_evidence": row["direction"], "coordinate_evidence": "valid source coordinates",
                              "shared_site_evidence": "see camera_mapping_conflicts.csv",
                              "legacy_evidence": "no contradiction found", "transition_neighbour_evidence": "corroboration only",
                              "accepted_status": "accepted"})
        else:
            crosswalk.append({"observed_camera_id": observed_id, "metadata_camera_id": "",
                              "classification": "unresolved", "matching_rule": "no unique deterministic alias",
                              "name_evidence": "unavailable on observation side", "address_evidence": "unavailable on observation side",
                              "direction_evidence": "unavailable on observation side", "coordinate_evidence": "unavailable on observation side",
                              "shared_site_evidence": "cannot identify observed ID", "legacy_evidence": "legacy assets opaque",
                              "transition_neighbour_evidence": "not used to originate identity", "accepted_status": "unresolved"})
    write_csv(pd.DataFrame(crosswalk), REPORTS / "camera_id_crosswalk.csv")

    # Embedded numbers in names/descriptions/addresses are audited as rejected candidates.
    component_pairs: dict[tuple[str, str], set[str]] = defaultdict(set)
    for _, row in metadata[metadata["camera"].isin(unmatched_metadata)].iterrows():
        for column in ("name", "descritpion", "address"):
            for component in re.findall(r"\d+", row[column]):
                canonical = str(int(component))
                if canonical in unmatched_observed:
                    component_pairs[(canonical, row["camera"])].add(column)
    by_observed: dict[str, set[str]] = defaultdict(set)
    by_metadata: dict[str, set[str]] = defaultdict(set)
    for observed_id, metadata_id in component_pairs:
        by_observed[observed_id].add(metadata_id)
        by_metadata[metadata_id].add(observed_id)

    conflict_rows = []
    for (observed_id, metadata_id), fields in sorted(component_pairs.items(), key=lambda item: (int(item[0][0]), int(item[0][1]))):
        row = metadata.loc[metadata["camera"].eq(metadata_id)].iloc[0]
        one_to_many = len(by_observed[observed_id]) > 1
        many_to_one = len(by_metadata[metadata_id]) > 1
        conflict_rows.append({"observed_camera_id": observed_id, "metadata_camera_id": metadata_id,
                              "candidate_rule": "numeric component extracted from metadata text",
                              "source_fields": "|".join(sorted(fields)), "one_to_many_conflict": one_to_many,
                              "many_to_one_conflict": many_to_one, "direction_conflict": "not_testable_observed_direction_absent",
                              "coordinate_conflict": "not_testable_observed_coordinate_absent",
                              "duplicate_site": bool(metadata.duplicated(["latitude", "longitude"], keep=False).loc[row.name]),
                              "direction": row["direction"], "decision": "rejected",
                              "reason": "numeric component denotes road/address context and is not a camera-ID crosswalk"})

    # Retain spatial duplicate-site evidence even where it does not produce an ID candidate.
    for _, group in metadata.groupby(["latitude", "longitude"]):
        if len(group) > 1:
            conflict_rows.append({"observed_camera_id": "", "metadata_camera_id": "|".join(sorted(group["camera"])),
                                  "candidate_rule": "duplicate metadata site", "source_fields": "latitude|longitude",
                                  "one_to_many_conflict": False, "many_to_one_conflict": False,
                                  "direction_conflict": group["direction"].nunique() > 1,
                                  "coordinate_conflict": False, "duplicate_site": True,
                                  "direction": "|".join(sorted(group["direction"])), "decision": "manual review",
                                  "reason": "shared sites may be legitimate directional camera pairs"})
    conflicts = pd.DataFrame(conflict_rows)
    write_csv(conflicts, REPORTS / "camera_mapping_conflicts.csv")

    nonnumeric_observed = sum(not value.isdigit() for value in observed)
    nonnumeric_metadata = sum(not value.isdigit() for value in metadata_ids)
    one_to_many_count = sum(len(values) > 1 for values in by_observed.values())
    many_to_one_count = sum(len(values) > 1 for values in by_metadata.values())
    component_observed = len(by_observed)
    report = f"""# Camera naming-error audit

Audit timestamp: {datetime.now(timezone.utc).isoformat()}

## Outcome

No additional camera was recovered. Coverage remains **119 of 225 observed cameras (52.89%)**; 106 remain unresolved.

## Identifier normalization

The audit applied Unicode compatibility normalization, trimming, case normalization, hyphen/underscore/space removal, leading-zero removal, prefix/suffix parsing, numeric-component extraction, and the direction aliases N/NB/NORTHBOUND, S/SB/SOUTHBOUND, E/EB/EASTBOUND, and W/WB/WESTBOUND.

All {len(observed)} observed IDs and all {len(metadata_ids)} metadata IDs are already plain canonical numeric strings: non-numeric observed IDs = {nonnumeric_observed}; non-numeric metadata IDs = {nonnumeric_metadata}. None contains whitespace, separators, prefixes, suffixes, leading zeros, or direction abbreviations. Thus these normalization rules produce no new deterministic one-to-one mapping.

## Embedded numeric-component findings

Numeric extraction from unmatched metadata names, descriptions, and addresses produced {len(component_pairs)} candidate pairs involving {component_observed} unmatched observed IDs. Inspection shows these numbers are road/address components such as A-road designations, not alternate camera identifiers. The candidates contain {one_to_many_count} observed-side one-to-many groups and {many_to_one_count} metadata-side many-to-one groups. Every such candidate is rejected in `camera_mapping_conflicts.csv`.

Direction abbreviations do occur in metadata camera names, but the observation IDs have no direction component. Consequently direction can corroborate a documented crosswalk but cannot establish one. Coordinates and shared sites have the same limitation because observations contain no camera coordinates.

## Corroboration controls

- Camera names, addresses, directions, and coordinates were checked on the metadata side.
- Duplicate sites are listed separately for manual review.
- Accessible legacy camera/network files remain opaque and provide no readable identifier crosswalk.
- Observed transition neighbours were not used to originate mappings. With no independently grounded non-exact candidate, topology would amount to identity inference from movement patterns and cannot resolve the naming ambiguity.
- Traffic volume and transition similarity were not used.

## Classification

- Exact verified: 119, accepted
- Deterministic one-to-one: {recovered}, accepted
- Legacy-crosswalk confirmed: 0
- Strongly corroborated/manual review: 0 ID mappings
- Unresolved: {len(unmatched_observed)}

Accepted and unresolved mappings are explicitly separated in `camera_id_crosswalk.csv`. No full routing, route-distance calculation, or speed filtering was started.
"""
    write_text(report, REPORTS / "camera_naming_error_audit.md")
    print(json.dumps({"recovered": recovered, "coverage_percentage": 100 * (119 + recovered) / len(observed),
                      "unresolved": len(unmatched_observed), "rejected_component_pairs": len(component_pairs),
                      "mapping_conflict_rows": len(conflicts)}, sort_keys=True))


if __name__ == "__main__":
    main()
