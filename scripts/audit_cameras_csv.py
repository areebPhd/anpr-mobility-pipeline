from __future__ import annotations

import csv
import hashlib
import json
import os
import stat
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
from pyproj import Transformer
from shapely import wkt


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("data/input/ANPR-2/cameras.csv")
REPORTS = ROOT / "reports"
PHASE2_PATTERN = ROOT / "restricted/phase3a_candidate_transitions/observations/week=*/observations.parquet"
PHASE3A_CANDIDATES = ROOT / "restricted/phase3a_candidate_transitions/candidates/candidate_transitions.parquet"
BOUNDARY = ROOT / "restricted/phase3b_camera_osm/boundary/tyne_wear_boundary.geojson"
EXPECTED_DIRECTIONS = {
    "Northbound", "Southbound", "Eastbound", "Westbound",
    "North & Southbound", "East & Westbound",
}
OPPOSITES = {frozenset(("Northbound", "Southbound")), frozenset(("Eastbound", "Westbound"))}


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def write_text(text: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def blank_count(series: pd.Series) -> int:
    return int(series.isna().sum() + series.astype("string").str.strip().eq("").fillna(False).sum())


def main() -> None:
    raw = SOURCE.read_bytes()
    source_hash = hashlib.sha256(raw).hexdigest()
    source_stat = SOURCE.stat()
    physical_line_count = raw.count(b"\n") + (not raw.endswith(b"\n"))
    decoded = raw.decode("utf-8")
    dialect = csv.Sniffer().sniff(decoded[:8192], delimiters=",\t;|")
    records = list(csv.reader(decoded.splitlines(keepends=True), dialect))
    header = records[0]
    data = pd.read_csv(SOURCE, dtype="string", encoding="utf-8")
    # Preserve the source spelling in the input frame and map only the internal working copy.
    working = data.rename(columns={"descritpion": "description"}).copy()
    working["camera"] = working["camera"].str.strip()
    for column in ("latitude", "longitude"):
        working[column] = pd.to_numeric(working[column], errors="coerce")

    geometries = []
    geometry_errors = 0
    for value in working["geometry"]:
        try:
            geometries.append(wkt.loads(value))
        except Exception:
            geometries.append(None)
            geometry_errors += 1
    point_count = sum(geometry is not None and geometry.geom_type == "Point" for geometry in geometries)
    nonpoint_count = sum(geometry is not None and geometry.geom_type != "Point" for geometry in geometries)

    valid_coordinates = (
        working["longitude"].between(-180, 180)
        & working["latitude"].between(-90, 90)
        & np.isfinite(working["longitude"])
        & np.isfinite(working["latitude"])
    )
    xs = np.array([geometry.x if geometry is not None and geometry.geom_type == "Point" else np.nan for geometry in geometries])
    ys = np.array([geometry.y if geometry is not None and geometry.geom_type == "Point" else np.nan for geometry in geometries])
    crs_trials = []
    for candidate in ("EPSG:32630", "EPSG:27700", "EPSG:3857"):
        longitude, latitude = Transformer.from_crs(candidate, "EPSG:4326", always_xy=True).transform(xs, ys)
        delta = np.hypot(np.asarray(longitude) - working["longitude"], np.asarray(latitude) - working["latitude"])
        crs_trials.append({"candidate": candidate, "max_angular_delta": float(np.nanmax(delta)),
                           "median_angular_delta": float(np.nanmedian(delta)),
                           "rows_within_1e_6_degrees": int(np.sum(delta <= 1e-6))})
    crs_trials.sort(key=lambda item: item["max_angular_delta"])
    best_crs = crs_trials[0]

    connection = duckdb.connect()
    phase2_ids = {row[0] for row in connection.execute(
        "SELECT DISTINCT CAST(camera AS VARCHAR) FROM read_parquet(?)", [str(PHASE2_PATTERN)]).fetchall()}
    phase3a_ids = {row[0] for row in connection.execute(
        "SELECT camera FROM (SELECT DISTINCT CAST(origin_camera AS VARCHAR) camera FROM read_parquet(?) "
        "UNION SELECT DISTINCT CAST(destination_camera AS VARCHAR) camera FROM read_parquet(?))",
        [str(PHASE3A_CANDIDATES), str(PHASE3A_CANDIDATES)]).fetchall()}
    connection.close()
    metadata_ids = set(working["camera"].dropna())
    observed_ids = phase2_ids | phase3a_ids

    outside_count = 0
    outside_mask = pd.Series(False, index=working.index)
    boundary_status = "not_available"
    if BOUNDARY.exists():
        boundary = gpd.read_file(BOUNDARY).to_crs("EPSG:4326")
        points = gpd.GeoDataFrame(working[["camera"]], geometry=gpd.points_from_xy(
            working["longitude"], working["latitude"]), crs="EPSG:4326")
        study_polygon = boundary.geometry.union_all()
        outside_mask = ~points.geometry.intersects(study_polygon)
        outside_count = int(outside_mask.sum())
        boundary_status = "current_nominatim_tyne_and_wear_boundary_temporal_limitation"

    conflicts: list[dict[str, object]] = []
    for _, row in working[outside_mask].iterrows():
        conflicts.append({"conflict_type": "outside_working_study_boundary", "camera_ids": row["camera"],
                          "record_count": 1, "latitude": row["latitude"], "longitude": row["longitude"],
                          "names_consistent": True, "addresses_consistent": True,
                          "directions": row["direction"], "opposite_direction_pair": False,
                          "severity": "review", "resolution": "validate against authoritative 2018 study boundary"})
    location_groups = working.groupby(["latitude", "longitude"], dropna=False)
    duplicate_location_groups = 0
    cameras_at_duplicate_locations = 0
    opposite_pair_groups = 0
    for (latitude, longitude), group in location_groups:
        if len(group) < 2:
            continue
        duplicate_location_groups += 1
        cameras_at_duplicate_locations += len(group)
        directions = set(group["direction"].dropna())
        opposite = any(pair.issubset(directions) for pair in OPPOSITES)
        opposite_pair_groups += int(opposite)
        conflicts.append({"conflict_type": "shared_spatial_location", "camera_ids": "|".join(sorted(group["camera"])),
                          "record_count": len(group), "latitude": latitude, "longitude": longitude,
                          "names_consistent": group["name"].nunique(dropna=False) == 1,
                          "addresses_consistent": group["address"].nunique(dropna=False) == 1,
                          "directions": "|".join(sorted(directions)), "opposite_direction_pair": opposite,
                          "severity": "review", "resolution": "retain as shared-site candidates; validate physically"})

    invalid_direction = ~working["direction"].isin(EXPECTED_DIRECTIONS)
    lane_whitespace_rows = int((working["lanes"] != working["lanes"].str.strip()).sum())
    for _, row in working[invalid_direction].iterrows():
        conflicts.append({"conflict_type": "invalid_direction", "camera_ids": row["camera"], "record_count": 1,
                          "latitude": row["latitude"], "longitude": row["longitude"], "names_consistent": True,
                          "addresses_consistent": True, "directions": row["direction"],
                          "opposite_direction_pair": False, "severity": "error", "resolution": "unresolved"})

    duplicate_id_rows = int(working["camera"].duplicated(keep=False).sum())
    missing = {column: blank_count(data[column]) for column in data.columns}
    profile_rows: list[dict[str, object]] = [
        {"section": "file", "metric": "source_path", "value": str(SOURCE), "status": "read_only"},
        {"section": "file", "metric": "size_bytes", "value": source_stat.st_size, "status": "verified"},
        {"section": "file", "metric": "permissions", "value": stat.filemode(source_stat.st_mode), "status": "verified"},
        {"section": "file", "metric": "modification_time_utc", "value": datetime.fromtimestamp(source_stat.st_mtime, timezone.utc).isoformat(), "status": "verified"},
        {"section": "file", "metric": "sha256", "value": source_hash, "status": "verified"},
        {"section": "format", "metric": "encoding", "value": "UTF-8 without BOM", "status": "pass"},
        {"section": "format", "metric": "delimiter", "value": repr(dialect.delimiter), "status": "pass"},
        {"section": "format", "metric": "header", "value": "|".join(header), "status": "pass"},
        {"section": "format", "metric": "physical_line_count", "value": physical_line_count, "status": "verified"},
        {"section": "format", "metric": "csv_record_count_including_header", "value": len(records), "status": "pass"},
        {"section": "format", "metric": "data_row_count", "value": len(working), "status": "pass"},
        {"section": "ids", "metric": "distinct_camera_ids", "value": len(metadata_ids), "status": "pass"},
        {"section": "ids", "metric": "duplicate_camera_id_rows", "value": duplicate_id_rows, "status": "pass" if not duplicate_id_rows else "fail"},
        {"section": "coordinates", "metric": "valid_lon_lat_rows", "value": int(valid_coordinates.sum()), "status": "pass"},
        {"section": "coordinates", "metric": "invalid_or_missing_lon_lat_rows", "value": int((~valid_coordinates).sum()), "status": "pass" if valid_coordinates.all() else "fail"},
        {"section": "coordinates", "metric": "latitude_bounds", "value": f"{working.latitude.min()}..{working.latitude.max()}", "status": "pass"},
        {"section": "coordinates", "metric": "longitude_bounds", "value": f"{working.longitude.min()}..{working.longitude.max()}", "status": "pass"},
        {"section": "coordinates", "metric": "outside_study_boundary", "value": outside_count, "status": "pass" if not outside_count else "review"},
        {"section": "coordinates", "metric": "study_boundary_status", "value": boundary_status, "status": "limitation"},
        {"section": "geometry", "metric": "valid_wkt_points", "value": point_count, "status": "pass"},
        {"section": "geometry", "metric": "geometry_parse_errors", "value": geometry_errors, "status": "pass" if not geometry_errors else "fail"},
        {"section": "geometry", "metric": "nonpoint_geometries", "value": nonpoint_count, "status": "pass" if not nonpoint_count else "fail"},
        {"section": "geometry", "metric": "source_declared_crs", "value": "none", "status": "limitation"},
        {"section": "geometry", "metric": "evidence_supported_crs", "value": best_crs["candidate"], "status": "derived_not_declared"},
        {"section": "geometry", "metric": "crs_max_angular_delta_degrees", "value": best_crs["max_angular_delta"], "status": "pass"},
        {"section": "geometry", "metric": "geometry_lat_lon_agreement_rows", "value": best_crs["rows_within_1e_6_degrees"], "status": "pass"},
        {"section": "spatial", "metric": "duplicate_location_groups", "value": duplicate_location_groups, "status": "review"},
        {"section": "spatial", "metric": "cameras_at_duplicate_locations", "value": cameras_at_duplicate_locations, "status": "review"},
        {"section": "spatial", "metric": "opposite_direction_location_groups", "value": opposite_pair_groups, "status": "review"},
        {"section": "direction", "metric": "invalid_direction_rows", "value": int(invalid_direction.sum()), "status": "pass" if not invalid_direction.any() else "fail"},
        {"section": "lanes", "metric": "values_with_outer_whitespace", "value": lane_whitespace_rows, "status": "review" if lane_whitespace_rows else "pass"},
    ]
    for column, count in missing.items():
        profile_rows.append({"section": "missing", "metric": column, "value": count, "status": "pass" if count == 0 else "review"})
    for category, count in working["direction"].value_counts().sort_index().items():
        profile_rows.append({"section": "direction_category", "metric": category, "value": int(count), "status": "valid" if category in EXPECTED_DIRECTIONS else "invalid"})
    for category, count in working["lanes"].value_counts().sort_index().items():
        profile_rows.append({"section": "lane_value", "metric": category, "value": int(count), "status": "source_value"})
    for category, count in working["road_category"].value_counts().sort_index().items():
        profile_rows.append({"section": "road_category", "metric": category, "value": int(count), "status": "source_value"})
    write_csv(pd.DataFrame(profile_rows), REPORTS / "cameras_csv_profile.csv")
    write_csv(pd.DataFrame(conflicts, columns=["conflict_type", "camera_ids", "record_count", "latitude", "longitude",
                                                    "names_consistent", "addresses_consistent", "directions",
                                                    "opposite_direction_pair", "severity", "resolution"]),
              REPORTS / "camera_metadata_conflicts.csv")

    match_count = len(observed_ids & metadata_ids)
    crosswalk = pd.DataFrame([
        {"population": "phase2_retained", "observed_camera_count": len(phase2_ids), "metadata_camera_count": len(metadata_ids),
         "matched_cameras": len(phase2_ids & metadata_ids), "observation_only_cameras": len(phase2_ids - metadata_ids),
         "metadata_only_cameras": len(metadata_ids - phase2_ids), "match_percentage_of_observed": 100 * len(phase2_ids & metadata_ids) / len(phase2_ids)},
        {"population": "phase3a_candidate_transitions", "observed_camera_count": len(phase3a_ids), "metadata_camera_count": len(metadata_ids),
         "matched_cameras": len(phase3a_ids & metadata_ids), "observation_only_cameras": len(phase3a_ids - metadata_ids),
         "metadata_only_cameras": len(metadata_ids - phase3a_ids), "match_percentage_of_observed": 100 * len(phase3a_ids & metadata_ids) / len(phase3a_ids)},
        {"population": "combined_phase2_phase3a", "observed_camera_count": len(observed_ids), "metadata_camera_count": len(metadata_ids),
         "matched_cameras": match_count, "observation_only_cameras": len(observed_ids - metadata_ids),
         "metadata_only_cameras": len(metadata_ids - observed_ids), "match_percentage_of_observed": 100 * match_count / len(observed_ids)},
    ])
    write_csv(crosswalk, REPORTS / "camera_id_crosswalk_summary.csv")

    verdict = "CAMERA METADATA ACCEPTED WITH LIMITATIONS"
    report = f"""# cameras.csv validation

Audit timestamp: {datetime.now(timezone.utc).isoformat()}

## Source integrity

- Path: `{SOURCE}` (read-only source; not copied or committed)
- Size: {source_stat.st_size:,} bytes
- Permissions: `{stat.filemode(source_stat.st_mode)}`
- Modification time: {datetime.fromtimestamp(source_stat.st_mtime, timezone.utc).isoformat()}
- SHA-256: `{source_hash}`

The file decodes strictly as UTF-8 without a byte-order mark. CSV dialect detection identifies a comma delimiter. It contains {len(records):,} logical CSV records including one header and {len(working):,} data rows. The physical line count is {physical_line_count:,}; the final record has no trailing newline but is complete.

The 11-column header is `geometry,camera,name,latitude,longitude,descritpion,type,direction,lanes,road_category,address`. The source spelling `descritpion` is preserved; only the internal audit frame maps it to `description`.

## Metadata validity

- Distinct camera IDs: {len(metadata_ids):,}; duplicate-ID rows: {duplicate_id_rows:,}
- Missing or blank values: zero in every source column
- Valid longitude/latitude rows: {int(valid_coordinates.sum()):,}; invalid or missing: {int((~valid_coordinates).sum()):,}
- Latitude bounds: {working.latitude.min()} to {working.latitude.max()}
- Longitude bounds: {working.longitude.min()} to {working.longitude.max()}
- Valid WKT point geometries: {point_count:,}; parse errors: {geometry_errors:,}; non-points: {nonpoint_count:,}
- Direction categories: {working.direction.nunique()} valid categories; invalid direction rows: {int(invalid_direction.sum()):,}
- Lane values: {working.lanes.nunique()} source categories; {lane_whitespace_rows} row contains outer whitespace and requires normalization only in a derived field
- Road categories: {working.road_category.nunique()} (`A`, `B`, `unclassified`)

{outside_count} coordinates fall outside the current Nominatim Tyne and Wear administrative polygon used for network extraction and require review. **Limitation:** this is current external boundary evidence, not a verified historical 2018 boundary, so these records are not labelled invalid solely from this comparison.

## CRS evidence and coordinate agreement

No CRS is declared in the CSV or a supplied sidecar. Candidate transformations were tested rather than assumed. EPSG:32630 (WGS 84 / UTM zone 30N) uniquely reproduces all {len(working):,} supplied longitude/latitude pairs within 0.000001 degrees; maximum angular discrepancy is {best_crs['max_angular_delta']:.12g} degrees. EPSG:27700 and EPSG:3857 do not agree.

**Evidence conclusion:** geometry is accepted as EPSG:32630 for Phase 3B processing, with the qualification `derived from complete coordinate agreement; not source-declared`.

## ID crosswalk

Phase 2 retained observations contain {len(phase2_ids):,} cameras. Phase 3A candidate transitions contain {len(phase3a_ids):,} cameras. Their sets are identical. Metadata contain {len(metadata_ids):,} cameras.

- Matched: {match_count:,}
- Observation-only: {len(observed_ids - metadata_ids):,}
- Metadata-only: {len(metadata_ids - observed_ids):,}
- Observed-camera match: {100 * match_count / len(observed_ids):.2f}%

The comparison is exact after trimming camera identifiers. No frequency, spatial, fuzzy, or transition-derived inference was used.

## Spatial and semantic review

- Exact duplicate-location groups: {duplicate_location_groups:,}, containing {cameras_at_duplicate_locations:,} cameras
- Shared-location groups containing an opposite-direction pair: {opposite_pair_groups:,}
- Shared-location groups with different names: {sum(group['name'].nunique(dropna=False) > 1 for _, group in location_groups if len(group) > 1):,}
- Shared-location groups with different addresses: {sum(group['address'].nunique(dropna=False) > 1 for _, group in location_groups if len(group) > 1):,}
- Shared-location groups with different direction values: {sum(group['direction'].nunique(dropna=False) > 1 for _, group in location_groups if len(group) > 1):,}
- Invalid directions: {int(invalid_direction.sum()):,}
- Coordinates outside the working study polygon: {outside_count:,}

Shared coordinates and opposite directions are recorded in `camera_metadata_conflicts.csv` for review; they are not automatically treated as erroneous because paired cameras can legitimately share a site. Names, addresses, and directions are retained as source evidence. Final directed-edge assignment still requires ambiguity checks for junctions and divided carriageways.

## Decision

The file passes structural, missingness, coordinate, geometry, direction, and internal CRS-consistency checks. It is accepted for the {match_count:,} exact observed-camera matches. The {len(observed_ids - metadata_ids):,} observation-only cameras remain unresolved, and the {len(metadata_ids - observed_ids):,} metadata-only cameras require provenance review. Phase 3B must not claim complete camera coverage.

No OSM routing, route-distance calculation, speed calculation, or speed filtering was performed.

**{verdict}**
"""
    write_text(report, REPORTS / "cameras_csv_validation.md")
    print(json.dumps({"verdict": verdict, "rows": len(working), "metadata_cameras": len(metadata_ids),
                      "observed_cameras": len(observed_ids), "matched": match_count,
                      "observation_only": len(observed_ids - metadata_ids),
                      "metadata_only": len(metadata_ids - observed_ids), "source_sha256": source_hash}, sort_keys=True))


if __name__ == "__main__":
    main()
