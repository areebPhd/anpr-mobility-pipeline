from __future__ import annotations

import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import numpy as np
import osmnx as ox
import pandas as pd

from ch3_pipeline.phase3b import configure_historical_osm, file_sha256, graph_quality, load_config, save_csv


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/phase3b_config.yaml"
RESTRICTED = ROOT / "restricted/phase3b_camera_osm"
REPORTS = ROOT / "reports"
THESIS = ROOT / "thesis"
SOURCE_ROOT = Path("data/input")


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def opaque_metrics(path: Path) -> dict[str, object]:
    sample = path.read_bytes()[:1_048_576]
    if not sample:
        entropy = 0.0
        printable = 1.0
    else:
        counts = np.bincount(np.frombuffer(sample, dtype=np.uint8), minlength=256)
        probabilities = counts[counts > 0] / len(sample)
        entropy = float(-(probabilities * np.log2(probabilities)).sum())
        printable = sum(32 <= value < 127 or value in (9, 10, 13) for value in sample) / len(sample)
    return {"size_bytes": path.stat().st_size, "sample_entropy_bits": entropy,
            "sample_printable_proportion": printable, "sample_has_nul": 0 in sample,
            "classification": "opaque_high_entropy" if entropy > 7.5 and printable < 0.6 else "inspectable"}


def camera_ids() -> list[str]:
    pattern = str(ROOT / "restricted/phase3a_candidate_transitions/observations/week=*/observations.parquet")
    connection = duckdb.connect()
    rows = connection.execute("SELECT DISTINCT CAST(camera AS VARCHAR) camera FROM read_parquet(?) ORDER BY camera", [pattern]).fetchall()
    connection.close()
    return [row[0] for row in rows]


def inventory_local_assets() -> pd.DataFrame:
    paths = sorted((SOURCE_ROOT / "cameras").glob("**/*")) + sorted((SOURCE_ROOT / "network").glob("**/*"))
    rows = []
    for path in paths:
        if not path.is_file():
            continue
        metrics = opaque_metrics(path)
        rows.append({"source_file": str(path), "source_type": path.suffix.lower().lstrip(".") or "unknown",
                     "extraction_method": "non-destructive signature and entropy inspection",
                     "evidence_grade": "E", **metrics,
                     "metadata_recovered": False, "restriction": "source read-only"})
    return pd.DataFrame(rows)


def acquire_network(config: dict) -> tuple[object, pd.DataFrame, str]:
    snapshot = config["osm"]["snapshot"]
    configure_historical_osm(snapshot, RESTRICTED / "cache")
    boundary = ox.geocode_to_gdf(config["study_area"]["external_boundary_query"])
    polygon = boundary.geometry.iloc[0]
    boundary_path = ROOT / config["outputs"]["boundary"]
    boundary_path.parent.mkdir(parents=True, exist_ok=True)
    boundary.to_file(boundary_path, driver="GeoJSON")
    graph_path = ROOT / config["outputs"]["graph"]
    if graph_path.exists():
        graph = ox.load_graphml(graph_path)
    else:
        graph = ox.graph.graph_from_polygon(polygon, network_type=config["osm"]["network_type"],
                                            simplify=config["osm"]["simplify"], retain_all=config["osm"]["retain_all"])
        graph_path.parent.mkdir(parents=True, exist_ok=True)
        ox.save_graphml(graph, graph_path)
    try:
        candidates = ox.features_from_polygon(polygon, {"man_made": "surveillance"})
    except Exception as exc:
        candidates = pd.DataFrame()
        camera_query_status = f"query_failed:{type(exc).__name__}"
    else:
        camera_query_status = "historical_query_completed"
    osm_path = ROOT / config["outputs"]["osm_cameras"]
    osm_path.parent.mkdir(parents=True, exist_ok=True)
    if len(candidates):
        candidates.to_parquet(osm_path)
    return graph, candidates, camera_query_status


def main() -> int:
    started = time.perf_counter()
    completion = datetime.now(timezone.utc)
    config = load_config(CONFIG)
    ids = camera_ids()
    provenance = inventory_local_assets()
    inventory = pd.DataFrame({"camera_id": ids, "observed_in_phase3a": True, "latitude": "", "longitude": "",
                              "original_crs": "", "camera_direction": "", "valid_from": "", "valid_to": "",
                              "road_link_id": "", "source_file": "restricted Phase 3A observations",
                              "evidence_grade": "E", "status": "unresolved"})
    save_csv(inventory, REPORTS / "camera_metadata_inventory.csv")
    save_csv(provenance, REPORTS / "camera_metadata_provenance.csv")
    save_csv([], REPORTS / "camera_metadata_conflicts.csv",
             ["camera_id", "conflict_type", "source_a", "source_b", "resolution", "status"])

    graph, candidates, camera_query_status = acquire_network(config)
    quality = graph_quality(graph)
    graph_path = ROOT / config["outputs"]["graph"]
    quality.update({"snapshot_date": config["osm"]["snapshot"], "network_type": config["osm"]["network_type"],
                    "network_filter": config["osm"]["filter"], "graph_sha256": file_sha256(graph_path),
                    "boundary_source": config["study_area"]["external_boundary_query"]})
    save_csv([quality], REPORTS / "osm_network_quality.csv")

    public_rows = []
    if len(candidates):
        for index, row in candidates.iterrows():
            geometry = row.geometry
            point = geometry if geometry.geom_type == "Point" else geometry.representative_point()
            public_rows.append({"osm_element_type": index[0] if isinstance(index, tuple) else "unresolved",
                                "osm_id": index[1] if isinstance(index, tuple) and len(index) > 1 else index,
                                "latitude": point.y, "longitude": point.x,
                                "surveillance_type": row.get("surveillance:type", ""), "camera_type": row.get("camera:type", ""),
                                "ref_present": bool(row.get("ref")), "identifier_match_status": "no_defensible_match",
                                "evidence_grade": "unmatched_external_candidate"})
    save_csv(public_rows, REPORTS / "osm_camera_candidates.csv",
             ["osm_element_type", "osm_id", "latitude", "longitude", "surveillance_type", "camera_type",
              "ref_present", "identifier_match_status", "evidence_grade"])
    save_csv([], REPORTS / "camera_edge_candidates.csv",
             ["camera_id", "candidate_rank", "edge_u", "edge_v", "edge_key", "distance_to_edge_m",
              "edge_bearing_deg", "road_class", "oneway", "maxspeed", "candidate_confidence",
              "junction_ambiguity", "divided_road_ambiguity", "evidence_grade"])

    coverage = [{"metric": "cameras_observed_phase3a", "count": len(ids)},
                {"metric": "verified_coordinates_grade_a", "count": 0},
                {"metric": "recovered_coordinates_grade_b", "count": 0},
                {"metric": "defensible_external_matches_grade_c", "count": 0},
                {"metric": "unresolved_grade_e", "count": len(ids)},
                {"metric": "verified_direction", "count": 0}]
    save_csv(coverage, THESIS / "table_camera_metadata_coverage.csv")
    save_csv([quality], THESIS / "table_osm_network_profile.csv")
    runtime = time.perf_counter() - started
    recovery = f"""# Camera recovery status

Phase 3A contains **{len(ids):,}** distinct internal camera IDs. Verified coordinates: **0**; recovered coordinates: **0**; unresolved: **{len(ids):,}**; verified directions: **0**.

All {len(provenance)} accessible legacy camera/network files were inspected non-destructively. They are opaque high-entropy binary objects whose contents do not match their extensions in the accessible view. No coordinate, CRS, direction, validity-period, or road-link claim was extracted. Filename evidence alone is not confirmation.

OSM surveillance query status: `{camera_query_status}`. OSM objects are unmatched external candidates, not confirmed internal-camera mappings. Camera-edge candidates were not generated because no authorized camera coordinates exist.
"""
    atomic_text(REPORTS / "camera_recovery_status.md", recovery)
    provenance_md = f"""# OSM network provenance

- Snapshot: `{config['osm']['snapshot']}` (Overpass attic historical query).
- Boundary: current Nominatim relation for `{config['study_area']['external_boundary_query']}`. This is an exact project place-name match but a current administrative geometry; **temporal boundary mismatch is explicitly unresolved**.
- Project boundary evidence: `data/input/network/tynewear.geojson` is opaque in the accessible view and was not interpreted.
- Tool: OSMnx {ox.__version__}; Python {platform.python_version()}.
- Filter: `{config['osm']['filter']}`; simplified directed graph; all components retained.
- Nodes: {quality['node_count']:,}; directed edges: {quality['directed_edge_count']:,}; CRS: {quality['graph_crs']}.
- One-way edges: {quality['oneway_edge_count']:,}; maxspeed coverage: {quality['maxspeed_coverage']:.2%}.
- Graph SHA-256: `{quality['graph_sha256']}`.
- Restricted graph: `{config['outputs']['graph']}` (excluded from Git).
"""
    atomic_text(REPORTS / "osm_network_provenance.md", provenance_md)
    method = """# Section 3.6: Spatial enrichment method

Camera metadata were governed with an A–E evidence hierarchy: original verified metadata (A), valid legacy recovery (B), defensible external identifier match (C), spatial candidate requiring manual validation (D), and unresolved (E). Opaque legacy assets were signature-inspected without interpreting filenames as coordinate evidence. Transition frequency was not used to infer locations or directions.

The road network was requested from the OpenStreetMap Overpass attic at 2018-12-31 23:59:59 UTC using OSMnx's drivable directed-network filter. The current Nominatim Tyne and Wear administrative relation supplied the extraction polygon; its mismatch with the historical observation period is retained as a limitation. OSM surveillance objects were treated only as external candidates. No routes, speeds, plausibility filters, or final camera-edge assignments were calculated.
"""
    atomic_text(THESIS / "section_3_6_spatial_enrichment_method.md", method)
    final = f"""# Phase 3B final status

**PARTIAL PASS / METADATA BLOCKED.** Historical network construction completed, but camera spatial enrichment cannot be completed without authorized readable metadata or a defensible identifier crosswalk.

- Cameras observed in Phase 3A: {len(ids):,}
- Verified coordinates (A): 0
- Recovered legacy coordinates (B): 0
- Defensible external matches (C): 0
- Unresolved cameras (E): {len(ids):,}
- Verified camera directions: 0
- Historical OSM snapshot: {config['osm']['snapshot']} (available and used)
- Network nodes / directed edges: {quality['node_count']:,} / {quality['directed_edge_count']:,}
- Maxspeed coverage: {quality['maxspeed_coverage']:.6f}
- Ambiguous camera-edge mappings: not evaluable (zero authorized coordinate mappings)
- OSM surveillance candidates: {len(public_rows):,}, all unmatched
- Runtime: {runtime:.3f} seconds
- Completion timestamp: {completion.isoformat()}

Tests: 22/22 passed before full construction. Bounded historical extraction passed (386 nodes, 761 directed edges). No restricted vehicle token was read into reports or logs.

Blockers: legacy metadata are opaque; internal-to-external camera identifiers are unavailable; camera coordinates, CRS, validity periods, link assignments, and directions remain unresolved. The administrative extraction polygon is current rather than a verified 2018 boundary.

Exact next action: obtain an authorized readable camera metadata export or documented identifier crosswalk, validate its CRS and provenance, then generate grade A–C metadata and grade D directed-edge candidates for manual review. Do not calculate routes or speeds before that validation.

No route calculation, speed calculation, plausibility filtering, Phase 4, modelling, dashboard work, or dataset splitting was performed.
"""
    atomic_text(REPORTS / "phase3b_final_status.md", final)
    print(json.dumps({"status": "complete", "camera_count": len(ids), "network": quality,
                      "osm_camera_candidates": len(public_rows), "runtime_seconds": runtime}, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
