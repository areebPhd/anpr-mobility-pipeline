from __future__ import annotations

import json
import math
import os
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import duckdb
import geopandas as gpd
import networkx as nx
import numpy as np
import osmnx as ox
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path("data/input/ANPR-2/cameras.csv")
GRAPH = ROOT / "restricted/phase3b_camera_osm/network/tyne_wear_drive_2018.graphml"
CANDIDATES = ROOT / "restricted/phase3a_candidate_transitions/candidates/candidate_transitions.parquet"
REPORTS = ROOT / "reports"
RESTRICTED = ROOT / "restricted/phase3b_camera_osm/topology_matching"


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def rank_correlation(x: list[float], y: list[float]) -> float:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan
    xr = pd.Series(x).rank(method="average").to_numpy()
    yr = pd.Series(y).rank(method="average").to_numpy()
    return float(np.corrcoef(xr, yr)[0, 1])


def edge_bearing(graph: nx.MultiDiGraph, u: object, v: object) -> float:
    a, b = graph.nodes[u], graph.nodes[v]
    return float(ox.bearing.calculate_bearing(a["y"], a["x"], b["y"], b["x"]))


def direction_score(direction: str, bearing: float) -> float:
    expected = {
        "Northbound": [0], "Eastbound": [90], "Southbound": [180], "Westbound": [270],
        "North & Southbound": [0, 180], "East & Westbound": [90, 270],
    }.get(direction, [])
    if not expected or math.isnan(bearing):
        return 0.5
    difference = min(abs((bearing - value + 180) % 360 - 180) for value in expected)
    return max(0.0, 1.0 - difference / 90.0)


def aggregate_observed(connection: duckdb.DuckDBPyConnection, exact: set[str], unresolved: set[str]) -> pd.DataFrame:
    exact_frame = pd.DataFrame({"camera": sorted(exact, key=int)})
    unresolved_frame = pd.DataFrame({"camera": sorted(unresolved, key=int)})
    connection.register("exact_ids", exact_frame)
    connection.register("unresolved_ids", unresolved_frame)
    query = """
    WITH relations AS (
      SELECT CAST(destination_camera AS VARCHAR) unresolved_camera,
             CAST(origin_camera AS VARCHAR) anchor_camera, 'anchor_to_unresolved' relation,
             destination_week event_week, elapsed_seconds
      FROM read_parquet(?)
      WHERE CAST(origin_camera AS VARCHAR) IN (SELECT camera FROM exact_ids)
        AND CAST(destination_camera AS VARCHAR) IN (SELECT camera FROM unresolved_ids)
        AND elapsed_seconds > 0
      UNION ALL
      SELECT CAST(origin_camera AS VARCHAR) unresolved_camera,
             CAST(destination_camera AS VARCHAR) anchor_camera, 'unresolved_to_anchor' relation,
             origin_week event_week, elapsed_seconds
      FROM read_parquet(?)
      WHERE CAST(origin_camera AS VARCHAR) IN (SELECT camera FROM unresolved_ids)
        AND CAST(destination_camera AS VARCHAR) IN (SELECT camera FROM exact_ids)
        AND elapsed_seconds > 0
    )
    SELECT unresolved_camera, anchor_camera, relation,
           CASE WHEN event_week <= 28 THEN 'inference' ELSE 'validation' END split,
           COUNT(*) support, COUNT(DISTINCT event_week) supporting_weeks,
           quantile_cont(elapsed_seconds, 0.25) q25_seconds,
           median(elapsed_seconds) median_seconds,
           quantile_cont(elapsed_seconds, 0.75) q75_seconds
    FROM relations GROUP BY ALL
    """
    return connection.execute(query, [str(CANDIDATES), str(CANDIDATES)]).fetchdf()


def main() -> None:
    started = time.perf_counter()
    metadata = pd.read_csv(SOURCE, dtype={"camera": "string"})
    metadata["camera"] = metadata["camera"].str.strip()
    graph = ox.load_graphml(GRAPH)
    connection = duckdb.connect()
    observed = {row[0] for row in connection.execute(
        "SELECT camera FROM (SELECT DISTINCT CAST(origin_camera AS VARCHAR) camera FROM read_parquet(?) "
        "UNION SELECT DISTINCT CAST(destination_camera AS VARCHAR) camera FROM read_parquet(?))",
        [str(CANDIDATES), str(CANDIDATES)]).fetchall()}
    metadata_ids = set(metadata["camera"])
    exact = observed & metadata_ids
    unresolved = observed - metadata_ids
    metadata_only = metadata_ids - observed
    observed_signature = aggregate_observed(connection, exact, unresolved)
    RESTRICTED.mkdir(parents=True, exist_ok=True)
    atomic_csv(observed_signature, RESTRICTED / "observed_transition_signatures.csv")

    longitude = metadata["longitude"].astype(float).to_numpy()
    latitude = metadata["latitude"].astype(float).to_numpy()
    camera_points = gpd.GeoDataFrame(metadata[["camera"]].copy(),
                                     geometry=gpd.points_from_xy(longitude, latitude), crs="EPSG:4326").to_crs("EPSG:27700")
    nodes_gdf, edges_gdf = ox.convert.graph_to_gdfs(graph)
    nodes_metric = nodes_gdf.to_crs("EPSG:27700")
    edges_metric = edges_gdf.to_crs("EPSG:27700").reset_index()
    node_matches = gpd.sjoin_nearest(camera_points, nodes_metric[["geometry"]], how="left", distance_col="node_distance_m")
    node_matches = node_matches[~node_matches.index.duplicated(keep="first")]
    metadata["nearest_node"] = list(node_matches["osmid"] if "osmid" in node_matches else node_matches["index_right"])
    edge_matches = gpd.sjoin_nearest(camera_points, edges_metric[["u", "v", "key", "geometry"]],
                                    how="left", distance_col="edge_distance_m")
    edge_matches = edge_matches[~edge_matches.index.duplicated(keep="first")]
    nearest_edges = list(zip(edge_matches["u"], edge_matches["v"], edge_matches["key"]))
    metadata["edge_u"] = [edge[0] for edge in nearest_edges]
    metadata["edge_v"] = [edge[1] for edge in nearest_edges]
    metadata["edge_key"] = [edge[2] for edge in nearest_edges]
    bearings, road_classes, one_way = [], [], []
    for edge in nearest_edges:
        u, v, key = edge
        data = graph.edges[u, v, key]
        bearings.append(edge_bearing(graph, u, v))
        road_classes.append(data.get("highway", "unresolved"))
        one_way.append(data.get("oneway", False))
    metadata["edge_bearing"] = bearings
    metadata["road_class"] = ["|".join(value) if isinstance(value, list) else value for value in road_classes]
    metadata["oneway"] = one_way
    metadata["direction_compatibility"] = [direction_score(direction, bearing) for direction, bearing in zip(metadata.direction, bearings)]

    by_id = metadata.set_index("camera")
    anchors = sorted(exact, key=int)
    all_metadata = sorted(metadata_ids, key=int)
    distance_path = RESTRICTED / "anchor_metadata_route_distances.csv"
    if distance_path.exists():
        distances = pd.read_csv(distance_path, dtype={"anchor_camera": "string", "metadata_camera": "string"})
    else:
        distance_rows = []
        reverse_graph = graph.reverse(copy=False)
        for anchor in anchors:
            node = by_id.loc[anchor, "nearest_node"]
            outward = nx.single_source_dijkstra_path_length(graph, node, weight="length")
            inward = nx.single_source_dijkstra_path_length(reverse_graph, node, weight="length")
            for target in all_metadata:
                target_node = by_id.loc[target, "nearest_node"]
                distance_rows.append({"anchor_camera": anchor, "metadata_camera": target,
                                      "anchor_to_metadata_m": outward.get(target_node, math.nan),
                                      "metadata_to_anchor_m": inward.get(target_node, math.nan)})
        distances = pd.DataFrame(distance_rows)
    for prefix in ("anchor_to_metadata", "metadata_to_anchor"):
        distances[f"{prefix}_expected_min_seconds_at_130_kph"] = distances[f"{prefix}_m"] / (130_000 / 3600)
        distances[f"{prefix}_expected_max_seconds_at_10_kph"] = distances[f"{prefix}_m"] / (10_000 / 3600)
    atomic_csv(distances, distance_path)
    distance_lookup = {(row.anchor_camera, row.metadata_camera):
                       (row.anchor_to_metadata_m, row.metadata_to_anchor_m)
                       for row in distances.itertuples(index=False)}
    spatial_rows = []
    for metadata_id in sorted(metadata_only, key=int):
        subset = distances[distances.metadata_camera.eq(metadata_id)]
        nearest_from = subset.nsmallest(5, "anchor_to_metadata_m")["anchor_camera"].astype(str).tolist()
        nearest_to = subset.nsmallest(5, "metadata_to_anchor_m")["anchor_camera"].astype(str).tolist()
        spatial_rows.append({"metadata_camera_id": metadata_id,
                             "nearest_node": by_id.loc[metadata_id, "nearest_node"],
                             "edge_u": by_id.loc[metadata_id, "edge_u"], "edge_v": by_id.loc[metadata_id, "edge_v"],
                             "edge_key": by_id.loc[metadata_id, "edge_key"],
                             "edge_bearing": by_id.loc[metadata_id, "edge_bearing"],
                             "camera_direction": by_id.loc[metadata_id, "direction"],
                             "direction_compatibility": by_id.loc[metadata_id, "direction_compatibility"],
                             "road_class": by_id.loc[metadata_id, "road_class"], "oneway": by_id.loc[metadata_id, "oneway"],
                             "nearest_outgoing_anchor_ids": "|".join(nearest_from),
                             "nearest_incoming_anchor_ids": "|".join(nearest_to),
                             "route_feasible_from_anchor_count": int(subset.anchor_to_metadata_m.notna().sum()),
                             "route_feasible_to_anchor_count": int(subset.metadata_to_anchor_m.notna().sum())})
    atomic_csv(pd.DataFrame(spatial_rows), RESTRICTED / "metadata_spatial_signatures.csv")

    signature_lookup: dict[tuple[str, str, str], pd.DataFrame] = {}
    for key, group in observed_signature.groupby(["unresolved_camera", "split", "relation"]):
        signature_lookup[key] = group.sort_values("support", ascending=False)

    score_rows = []
    for unresolved_id in sorted(unresolved, key=int):
        inference_groups = [signature_lookup.get((unresolved_id, "inference", relation), pd.DataFrame())
                            for relation in ("anchor_to_unresolved", "unresolved_to_anchor")]
        validation_groups = [signature_lookup.get((unresolved_id, "validation", relation), pd.DataFrame())
                             for relation in ("anchor_to_unresolved", "unresolved_to_anchor")]
        inference_support = int(sum(group.support.sum() for group in inference_groups if len(group)))
        inference_weeks = set()
        for group in inference_groups:
            if len(group):
                inference_weeks.update(group.anchor_camera)
        weekly_stability = min(1.0, sum(group.supporting_weeks.sum() for group in inference_groups if len(group)) / 28.0)
        for metadata_id in sorted(metadata_only, key=int):
            inference_observed, inference_distance = [], []
            validation_observed, validation_distance = [], []
            feasible = 0
            relations = 0
            top_feasible = 0
            top_total = 0
            for group, relation in zip(inference_groups, ("anchor_to_unresolved", "unresolved_to_anchor")):
                if not len(group):
                    continue
                for rank, row in enumerate(group.itertuples(index=False), start=1):
                    distance_pair = distance_lookup[(row.anchor_camera, metadata_id)]
                    distance = distance_pair[0] if relation == "anchor_to_unresolved" else distance_pair[1]
                    relations += 1
                    if np.isfinite(distance):
                        feasible += 1
                        inference_observed.append(float(row.median_seconds))
                        inference_distance.append(float(distance))
                        if rank <= 5:
                            top_feasible += 1
                    if rank <= 5:
                        top_total += 1
            for group, relation in zip(validation_groups, ("anchor_to_unresolved", "unresolved_to_anchor")):
                if not len(group):
                    continue
                for row in group.itertuples(index=False):
                    distance_pair = distance_lookup[(row.anchor_camera, metadata_id)]
                    distance = distance_pair[0] if relation == "anchor_to_unresolved" else distance_pair[1]
                    if np.isfinite(distance):
                        validation_observed.append(float(row.median_seconds))
                        validation_distance.append(float(distance))
            rho = rank_correlation(inference_observed, inference_distance)
            validation_rho = rank_correlation(validation_observed, validation_distance)
            rank_score = 0.5 if math.isnan(rho) else (rho + 1) / 2
            validation_score = 0.5 if math.isnan(validation_rho) else (validation_rho + 1) / 2
            feasibility = feasible / relations if relations else 0.0
            predecessor_successor = top_feasible / top_total if top_total else 0.0
            direction = float(by_id.loc[metadata_id, "direction_compatibility"])
            inference_score = (0.25 * predecessor_successor + 0.30 * rank_score + 0.20 * direction
                               + 0.15 * feasibility + 0.10 * weekly_stability)
            score_rows.append({"observed_camera_id": unresolved_id, "metadata_camera_id": metadata_id,
                               "inference_score": inference_score, "predecessor_successor_agreement": predecessor_successor,
                               "travel_time_distance_rank_correlation": rho, "direction_compatibility": direction,
                               "route_feasibility": feasibility, "weekly_stability": weekly_stability,
                               "inference_support": inference_support, "inference_anchor_relations": relations,
                               "validation_rank_correlation": validation_rho, "validation_score": validation_score,
                               "validation_relations": len(validation_observed),
                               "edge_bearing": by_id.loc[metadata_id, "edge_bearing"],
                               "camera_direction": by_id.loc[metadata_id, "direction"],
                               "road_class": by_id.loc[metadata_id, "road_class"]})
    scores = pd.DataFrame(score_rows)
    scores["rank"] = scores.groupby("observed_camera_id")["inference_score"].rank(method="first", ascending=False)
    best_two = scores.sort_values(["observed_camera_id", "inference_score"], ascending=[True, False]).groupby("observed_camera_id").head(2)
    margins = best_two.groupby("observed_camera_id")["inference_score"].apply(lambda values: values.iloc[0] - values.iloc[1] if len(values) > 1 else math.nan)
    scores["uniqueness_margin"] = scores["observed_camera_id"].map(margins)

    eligible = scores[(scores["rank"] <= 5) & (scores["inference_score"] >= 0.68)
                      & (scores["validation_score"] >= 0.58) & (scores["inference_support"] >= 100)
                      & (scores["validation_relations"] >= 3)]
    bipartite = nx.Graph()
    for row in eligible.itertuples(index=False):
        bipartite.add_edge(f"o:{row.observed_camera_id}", f"m:{row.metadata_camera_id}", weight=float(row.inference_score))
    assignment = nx.max_weight_matching(bipartite, maxcardinality=False, weight="weight")
    assigned: dict[str, str] = {}
    for left, right in assignment:
        if left.startswith("m:"):
            left, right = right, left
        assigned[left[2:]] = right[2:]

    validation_rows = []
    classifications = []
    for observed_id in sorted(unresolved, key=int):
        group = scores[scores.observed_camera_id.eq(observed_id)].sort_values("inference_score", ascending=False)
        best = group.iloc[0]
        selected_metadata = assigned.get(observed_id, "")
        selected = group[group.metadata_camera_id.eq(selected_metadata)].iloc[0] if selected_metadata else best
        margin = float(best.uniqueness_margin)
        if selected_metadata and selected.inference_score >= 0.78 and selected.validation_score >= 0.68 and margin >= 0.08:
            classification = "high-confidence inferred"
        elif selected_metadata:
            classification = "moderate-confidence/manual review"
        elif best.inference_score >= 0.60:
            classification = "ambiguous"
        elif best.inference_support:
            classification = "rejected"
        else:
            classification = "unresolved"
        classifications.append(classification)
        validation_rows.append({"observed_camera_id": observed_id,
                                "selected_metadata_camera_id": selected_metadata,
                                "best_candidate_metadata_id": best.metadata_camera_id,
                                "classification": classification,
                                "inference_score": selected.inference_score if selected_metadata else best.inference_score,
                                "validation_score": selected.validation_score if selected_metadata else best.validation_score,
                                "validation_rank_correlation": selected.validation_rank_correlation if selected_metadata else best.validation_rank_correlation,
                                "inference_support": selected.inference_support if selected_metadata else best.inference_support,
                                "validation_relations": selected.validation_relations if selected_metadata else best.validation_relations,
                                "uniqueness_margin": margin,
                                "one_to_one_assignment": bool(selected_metadata),
                                "primary_dataset_status": "withheld_pending_manual_conflict_audit" if selected_metadata else "not_mapped"})
    validation = pd.DataFrame(validation_rows)
    scores = scores.merge(validation[["observed_camera_id", "selected_metadata_camera_id", "classification"]], on="observed_camera_id", how="left")
    scores["candidate_selected"] = scores.metadata_camera_id.eq(scores.selected_metadata_camera_id)
    atomic_csv(scores, REPORTS / "topology_assisted_camera_candidates.csv")
    atomic_csv(validation, REPORTS / "topology_assisted_camera_validation.csv")

    distribution = scores.groupby(pd.cut(scores.inference_score, bins=[0, .4, .5, .6, .7, .8, .9, 1], include_lowest=True), observed=True).agg(
        candidate_count=("inference_score", "size"), observed_camera_count=("observed_camera_id", "nunique"),
        selected_count=("candidate_selected", "sum")).reset_index()
    distribution["score_interval"] = distribution.inference_score.astype(str)
    distribution = distribution.drop(columns="inference_score")
    atomic_csv(distribution, REPORTS / "camera_mapping_score_distribution.csv")

    high = classifications.count("high-confidence inferred")
    moderate = classifications.count("moderate-confidence/manual review")
    ambiguous = classifications.count("ambiguous")
    rejected = classifications.count("rejected")
    unresolved_count = classifications.count("unresolved")
    inferred = high + moderate
    validation_success = int(((validation.one_to_one_assignment) & (validation.validation_score >= .58)).sum())
    validation_rate = validation_success / inferred if inferred else math.nan
    accepted_ids = exact | set(validation.loc[validation.classification.eq("high-confidence inferred"), "observed_camera_id"])
    provisional_ids = accepted_ids | set(validation.loc[validation.classification.eq("moderate-confidence/manual review"), "observed_camera_id"])
    def calculate_coverage(camera_ids: set[str], relation_name: str) -> tuple[int, int, int, int]:
        frame = pd.DataFrame({"camera": sorted(camera_ids, key=int)})
        connection.register(relation_name, frame)
        return connection.execute(f"""
      SELECT COUNT(*) total_transitions,
             COUNT(*) FILTER (WHERE CAST(origin_camera AS VARCHAR) IN (SELECT camera FROM {relation_name})
                              AND CAST(destination_camera AS VARCHAR) IN (SELECT camera FROM {relation_name})) covered_transitions,
             SUM(2) total_transition_endpoints,
             SUM((CAST(origin_camera AS VARCHAR) IN (SELECT camera FROM {relation_name}))::INTEGER
                + (CAST(destination_camera AS VARCHAR) IN (SELECT camera FROM {relation_name}))::INTEGER) covered_endpoints
      FROM read_parquet(?)
    """, [str(CANDIDATES)]).fetchone()
    coverage = calculate_coverage(accepted_ids, "covered_primary")
    provisional_coverage = calculate_coverage(provisional_ids, "covered_provisional")
    connection.close()
    runtime = time.perf_counter() - started
    report = f"""# Topology-assisted camera matching

Run timestamp: {datetime.now(timezone.utc).isoformat()}

## Experimental status

This is a provisional topology-assisted inference experiment. The 119 exact camera mappings remain separate. Inferred mappings are withheld from the primary dataset until held-out validation and manual-conflict review pass.

Historical OSM route distances were computed on the 2018-12-31 directed graph between all exact anchors and all 233 metadata cameras in both directions. Weeks 01–28 supplied inference signatures; weeks 29–35 were held out for validation. No traffic-volume or transition-similarity-only mapping was permitted.

## Results

- Exact anchors: {len(exact)}
- Unresolved observed cameras tested: {len(unresolved)}
- Metadata-only candidates: {len(metadata_only)}
- High-confidence inferred: {high}
- Moderate-confidence/manual review: {moderate}
- Ambiguous: {ambiguous}
- Rejected: {rejected}
- Unresolved due to absent support: {unresolved_count}
- One-to-one provisional assignments: {inferred}
- Held-out validation successes among assignments: {validation_success}
- Validation success rate: {validation_rate if not math.isnan(validation_rate) else 'not estimable'}
- Primary accepted coverage (exact plus high-confidence only): {len(accepted_ids)} of {len(observed)} cameras ({100 * len(accepted_ids) / len(observed):.2f}%)
- Provisional coverage including moderate/manual-review assignments: {len(provisional_ids)} of {len(observed)} cameras ({100 * len(provisional_ids) / len(observed):.2f}%)
- Candidate transitions with both endpoints covered: {coverage[1]:,} of {coverage[0]:,} ({100 * coverage[1] / coverage[0]:.2f}%)
- Transition endpoints covered: {coverage[3]:,} of {coverage[2]:,} ({100 * coverage[3] / coverage[2]:.2f}%)
- Provisional candidate-transition coverage including moderate assignments: {provisional_coverage[1]:,} of {provisional_coverage[0]:,} ({100 * provisional_coverage[1] / provisional_coverage[0]:.2f}%)
- Provisional transition-endpoint coverage including moderate assignments: {provisional_coverage[3]:,} of {provisional_coverage[2]:,} ({100 * provisional_coverage[3] / provisional_coverage[2]:.2f}%)
- Runtime: {runtime:.3f} seconds

## Scoring and safeguards

Scores combine predecessor/successor route feasibility, Spearman rank agreement between observed median travel times and directed OSM distances, source-direction versus nearest-edge bearing compatibility, weekly support stability, and held-out validation. A maximum-weight bipartite matching enforces one-to-one assignment only for candidates clearing minimum score, validation, support, and relation-count gates. Unmatched and ambiguous outcomes are allowed; shared-site cameras are not collapsed.

Expected route-time bounds are recorded in the restricted distance matrix as transparent 130 km/h minimum-time and 10 km/h maximum-time scenarios. These are broad comparison scenarios, not inferred vehicle speeds or filtering thresholds. The restricted distance and signature matrices remain outside Git.

No inferred mapping is approved for the primary final dataset by this report alone. No speed filtering was performed.
"""
    atomic_text(report, REPORTS / "topology_assisted_camera_matching.md")
    print(json.dumps({"high_confidence": high, "moderate": moderate, "ambiguous": ambiguous,
                      "rejected": rejected, "unresolved": unresolved_count, "inferred_assignments": inferred,
                      "runtime_seconds": runtime}, sort_keys=True))


if __name__ == "__main__":
    main()
