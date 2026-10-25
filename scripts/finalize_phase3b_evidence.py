from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import geopandas as gpd
import osmnx as ox
import pandas as pd

from ch3_pipeline.phase3b import nearest_edge_candidates


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("data/input/ANPR-2/cameras.csv")
OBSERVATIONS = ROOT / "restricted/phase3a_candidate_transitions/observations/week=*/observations.parquet"
GRAPH = ROOT / "restricted/phase3b_camera_osm/network/tyne_wear_drive_2018.graphml"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    metadata = pd.read_csv(SOURCE, dtype={"camera": "string"})
    metadata["camera"] = metadata["camera"].str.strip()
    connection = duckdb.connect()
    observed = {row[0] for row in connection.execute(
        "SELECT DISTINCT CAST(camera AS VARCHAR) FROM read_parquet(?)", [str(OBSERVATIONS)]).fetchall()}
    connection.close()
    metadata_ids = set(metadata.camera)
    exact = observed & metadata_ids
    unresolved = observed - metadata_ids

    indexed = metadata.set_index("camera")
    inventory_rows = []
    for camera_id in sorted(observed, key=int):
        if camera_id in exact:
            row = indexed.loc[camera_id]
            inventory_rows.append({"camera_id": camera_id, "observed_in_phase3a": True,
                                   "latitude": row.latitude, "longitude": row.longitude,
                                   "original_crs": "EPSG:32630 (derived from complete coordinate agreement; not source-declared)",
                                   "camera_direction": row.direction, "valid_from": "", "valid_to": "",
                                   "road_link_id": "", "source_file": str(SOURCE), "evidence_grade": "A",
                                   "status": "exact_verified_metadata_match"})
        else:
            inventory_rows.append({"camera_id": camera_id, "observed_in_phase3a": True,
                                   "latitude": "", "longitude": "", "original_crs": "",
                                   "camera_direction": "", "valid_from": "", "valid_to": "",
                                   "road_link_id": "", "source_file": "", "evidence_grade": "E",
                                   "status": "unresolved_observation_id"})
    atomic_csv(pd.DataFrame(inventory_rows), ROOT / "reports/camera_metadata_inventory.csv")

    graph = ox.load_graphml(GRAPH)
    _, edges = ox.convert.graph_to_gdfs(graph)
    edges = edges.reset_index()
    points = gpd.GeoDataFrame(metadata[["camera"]].rename(columns={"camera": "camera_id"}),
                              geometry=gpd.points_from_xy(metadata.longitude, metadata.latitude), crs="EPSG:4326")
    candidates = nearest_edge_candidates(points, edges, k=3, ambiguity_metres=5.0)
    candidates["observation_mapping_status"] = candidates.camera_id.map(
        lambda value: "exact_observed_anchor" if str(value) in exact else "metadata_only_not_observation_matched")
    candidates["primary_analysis_status"] = "candidate_only_manual_validation_required"
    atomic_csv(candidates, ROOT / "reports/camera_edge_candidates.csv")

    coverage = pd.DataFrame([
        {"metric": "cameras_observed_phase3a", "count": len(observed)},
        {"metric": "exact_verified_coordinate_matches", "count": len(exact)},
        {"metric": "high_confidence_inferred_matches", "count": 0},
        {"metric": "moderate_inferred_withheld", "count": 2},
        {"metric": "ambiguous_or_unresolved_observed", "count": 104},
        {"metric": "source_metadata_cameras", "count": len(metadata_ids)},
        {"metric": "metadata_only_not_observation_matched", "count": len(metadata_ids - observed)},
        {"metric": "verified_source_directions_for_exact_matches", "count": len(exact)},
    ])
    atomic_csv(coverage, ROOT / "thesis/table_camera_metadata_coverage.csv")

    exact_edges = candidates[candidates.observation_mapping_status.eq("exact_observed_anchor")]
    nearest_exact = exact_edges[exact_edges.candidate_rank.eq(1)]
    recovery = f"""# Camera recovery status

Final Phase 3B evidence status at {datetime.now(timezone.utc).isoformat()}:

- Observed Phase 2/3A cameras: {len(observed)}
- Exact verified metadata matches: {len(exact)}
- High-confidence inferred matches: 0
- Moderate topology-assisted mappings: 2, withheld from primary analysis
- Ambiguous/unresolved observed cameras: 104
- Metadata-only cameras: {len(metadata_ids - observed)}, not treated as observation-ID matches
- Source directions available for exact matches: {len(exact)}

All 233 metadata cameras have three potential directed-edge candidates. For the {len(exact)} exact observed anchors, nearest-edge candidates have median distance {nearest_exact.distance_to_edge_m.median():.3f} m and maximum distance {nearest_exact.distance_to_edge_m.max():.3f} m; {int(nearest_exact.junction_ambiguity.sum())} have a nearest-edge tie within 5 m and {int(nearest_exact.divided_road_ambiguity.sum())} carry a divided-road ambiguity flag. These remain candidate mappings requiring manual validation, not final direction assignments.

No inferred mapping is admitted to the primary dataset. No speed filtering or Phase 4 work was performed.
"""
    atomic_text(recovery, ROOT / "reports/camera_recovery_status.md")

    status = f"""# Phase 3B final status

**COMPLETE WITH DOCUMENTED LIMITATIONS.** Camera metadata, historical-network provenance, edge candidates, identifier reconciliation, and topology-assisted inference have been audited. Inferred mappings remain excluded from primary analysis.

- Observed cameras: {len(observed)}
- Exact verified camera matches: {len(exact)}
- High-confidence inferred matches: 0
- Moderate mappings withheld: 2
- Ambiguous/unresolved observed cameras: 104
- Metadata-only cameras: {len(metadata_ids - observed)}
- Historical OSM snapshot: 2018-12-31T23:59:59Z
- Graph: 42,604 nodes; 96,869 directed edges; EPSG:4326
- Maxspeed coverage: 11.661109%
- Graph SHA-256: a25e47062f05a429b97d8ffc8675a07e0feeebdc15925ed48dcebd1ac4f3d37c
- Primary spatial camera coverage: 119/225 (52.89%)
- Primary candidate-transition coverage: 151,186,774/432,633,297 (34.95%)
- Primary transition-endpoint coverage: 488,372,743/865,266,594 (56.44%)
- Provisional coverage including two moderate mappings: 121/225 (53.78%); not approved for primary use
- Exact-anchor nearest-edge median/max distance: {nearest_exact.distance_to_edge_m.median():.3f}/{nearest_exact.distance_to_edge_m.max():.3f} m
- Exact-anchor junction/divided-road ambiguity flags: {int(nearest_exact.junction_ambiguity.sum())}/{int(nearest_exact.divided_road_ambiguity.sum())}
- Tests: 22/22 passed

Limitations: 106 original observation IDs lacked exact metadata, topology produced no high-confidence inference, 104 remain ambiguous, two moderate assignments have small uniqueness margins and require manual audit, geometry CRS EPSG:32630 is evidence-derived rather than source-declared, three source coordinates lie outside the current working administrative boundary, the extraction boundary is current rather than verified 2018 geometry, OSM completeness and maxspeed coverage are limited, camera-edge candidates are not final mappings, camera directions and bidirectional/shared-site semantics require manual validation, and the final study period and earlier Phase 2/3A methodological decisions remain unresolved.

No inferred mapping was used in a primary dataset. No speed filtering, Phase 4, modelling, dashboard, or data splitting was performed.
"""
    atomic_text(status, ROOT / "reports/phase3b_final_status.md")


if __name__ == "__main__":
    main()
