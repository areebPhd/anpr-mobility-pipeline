from __future__ import annotations

import heapq
import json
import math
import os
import re
import time
from collections import Counter
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
TRANSITIONS = ROOT / "restricted/phase3a_candidate_transitions/candidates/candidate_transitions.parquet"
RESTRICTED = ROOT / "restricted/phase3b_camera_osm/calibrated_time_audit"
REPORTS = ROOT / "reports"
DEFAULT_SPEED_KPH = {"motorway": 100, "motorway_link": 60, "trunk": 80, "trunk_link": 50,
                     "primary": 60, "primary_link": 40, "secondary": 50, "secondary_link": 35,
                     "tertiary": 40, "tertiary_link": 30, "residential": 30,
                     "living_street": 15, "unclassified": 30, "road": 30, "disused": 10}


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def atomic_text(text: str, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def parse_speed(value: object, highway: object) -> float:
    values = value if isinstance(value, list) else [value]
    parsed = []
    for item in values:
        if item is None:
            continue
        match = re.search(r"(\d+(?:\.\d+)?)", str(item))
        if match:
            speed = float(match.group(1))
            if "mph" in str(item).lower():
                speed *= 1.609344
            parsed.append(speed)
    if parsed:
        return float(np.median(parsed))
    classes = highway if isinstance(highway, list) else [highway]
    return max((DEFAULT_SPEED_KPH.get(str(item), 30) for item in classes), default=30)


def prepare_edges(graph: nx.MultiDiGraph) -> None:
    for _, _, _, data in graph.edges(keys=True, data=True):
        speed = parse_speed(data.get("maxspeed"), data.get("highway"))
        data["provisional_speed_kph"] = speed
        data["provisional_time_s"] = float(data.get("length", 0)) / (speed * 1000 / 3600)


def target_dijkstra(graph: nx.MultiDiGraph, source: object, targets: set[object]) -> tuple[dict[object, float], dict[object, object]]:
    distance = {source: 0.0}
    predecessor: dict[object, object] = {}
    queue = [(0.0, source)]
    settled = set()
    remaining = set(targets)
    while queue and remaining:
        current_distance, node = heapq.heappop(queue)
        if node in settled:
            continue
        settled.add(node)
        remaining.discard(node)
        for neighbour, keyed in graph[node].items():
            weight = min(float(data.get("length", math.inf)) for data in keyed.values())
            proposed = current_distance + weight
            if proposed < distance.get(neighbour, math.inf):
                distance[neighbour] = proposed
                predecessor[neighbour] = node
                heapq.heappush(queue, (proposed, neighbour))
    return distance, predecessor


def route_attributes(graph: nx.MultiDiGraph, source: object, target: object,
                     distances: dict[object, float], predecessor: dict[object, object]) -> dict[str, object]:
    if target not in distances:
        return {"route_distance_m": math.nan, "provisional_route_time_s": math.nan,
                "predominant_road_class": "unreachable", "route_edge_count": 0}
    nodes = [target]
    while nodes[-1] != source:
        if nodes[-1] not in predecessor:
            return {"route_distance_m": math.nan, "provisional_route_time_s": math.nan,
                    "predominant_road_class": "unreachable", "route_edge_count": 0}
        nodes.append(predecessor[nodes[-1]])
    nodes.reverse()
    class_lengths: Counter[str] = Counter()
    travel_time = 0.0
    for u, v in zip(nodes[:-1], nodes[1:]):
        alternatives = graph[u][v]
        data = min(alternatives.values(), key=lambda item: float(item.get("length", math.inf)))
        length = float(data.get("length", 0))
        classes = data.get("highway", "unclassified")
        road_class = str(classes[0] if isinstance(classes, list) else classes)
        class_lengths[road_class] += length
        travel_time += float(data.get("provisional_time_s", 0))
    predominant = class_lengths.most_common(1)[0][0] if class_lengths else "unclassified"
    return {"route_distance_m": distances[target], "provisional_route_time_s": travel_time,
            "predominant_road_class": predominant, "route_edge_count": max(0, len(nodes) - 1)}


def nearest_nodes(graph: nx.MultiDiGraph, metadata: pd.DataFrame) -> list[object]:
    points = gpd.GeoDataFrame(metadata[["camera"]], geometry=gpd.points_from_xy(metadata.longitude, metadata.latitude),
                             crs="EPSG:4326").to_crs("EPSG:27700")
    nodes, _ = ox.convert.graph_to_gdfs(graph)
    joined = gpd.sjoin_nearest(points, nodes.to_crs("EPSG:27700")[["geometry"]], how="left")
    joined = joined[~joined.index.duplicated(keep="first")]
    return list(joined["osmid"] if "osmid" in joined else joined["index_right"])


def build_routes(graph: nx.MultiDiGraph, metadata: pd.DataFrame, path: Path) -> pd.DataFrame:
    if path.exists():
        routes = pd.read_csv(path, dtype={"origin_metadata_id": "string", "destination_metadata_id": "string"})
        if len(routes) == len(metadata) * (len(metadata) - 1):
            return routes
    prepare_edges(graph)
    metadata = metadata.copy()
    metadata["node"] = nearest_nodes(graph, metadata)
    targets = set(metadata.node)
    rows = []
    for origin in metadata.itertuples(index=False):
        distances, predecessor = target_dijkstra(graph, origin.node, targets)
        for destination in metadata.itertuples(index=False):
            if origin.camera == destination.camera:
                continue
            rows.append({"origin_metadata_id": origin.camera, "destination_metadata_id": destination.camera,
                         **route_attributes(graph, origin.node, destination.node, distances, predecessor)})
    routes = pd.DataFrame(rows)
    atomic_csv(routes, path)
    return routes


def observed_summaries(connection: duckdb.DuckDBPyConnection, exact: set[str], unresolved: set[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    connection.register("exact_ids", pd.DataFrame({"camera": sorted(exact, key=int)}))
    connection.register("unresolved_ids", pd.DataFrame({"camera": sorted(unresolved, key=int)}))
    base = """
      SELECT CAST(origin_camera AS VARCHAR) origin_camera, CAST(destination_camera AS VARCHAR) destination_camera,
             origin_week event_week, origin_timestamp, elapsed_seconds
      FROM read_parquet(?) WHERE elapsed_seconds > 0
    """
    aggregation = """
      SELECT origin_camera, destination_camera,
             CASE WHEN event_week <= 28 THEN 'inference' ELSE 'validation' END split,
             COUNT(*) transition_count, COUNT(DISTINCT event_week) supporting_weeks,
             quantile_cont(elapsed_seconds,.10) q10_seconds,
             quantile_cont(elapsed_seconds,.25) q25_seconds,
             median(elapsed_seconds) median_seconds,
             quantile_cont(elapsed_seconds,.75) q75_seconds,
             quantile_cont(elapsed_seconds,.90) q90_seconds,
             median(elapsed_seconds) FILTER (WHERE EXTRACT(hour FROM origin_timestamp) >= 22 OR EXTRACT(hour FROM origin_timestamp) < 6) nighttime_median_seconds,
             COUNT(*) FILTER (WHERE EXTRACT(hour FROM origin_timestamp) >= 22 OR EXTRACT(hour FROM origin_timestamp) < 6) nighttime_count
      FROM selected GROUP BY ALL
    """
    anchor_query = f"""WITH raw AS ({base}), selected AS (
      SELECT * FROM raw WHERE origin_camera IN (SELECT camera FROM exact_ids)
       AND destination_camera IN (SELECT camera FROM exact_ids) AND origin_camera <> destination_camera)
      {aggregation}"""
    unresolved_query = f"""WITH raw AS ({base}), selected AS (
      SELECT * FROM raw WHERE (origin_camera IN (SELECT camera FROM unresolved_ids) AND destination_camera IN (SELECT camera FROM exact_ids))
       OR (origin_camera IN (SELECT camera FROM exact_ids) AND destination_camera IN (SELECT camera FROM unresolved_ids)))
      {aggregation}"""
    anchors = connection.execute(anchor_query, [str(TRANSITIONS)]).fetchdf()
    signatures = connection.execute(unresolved_query, [str(TRANSITIONS)]).fetchdf()
    return anchors, signatures


def design_matrix(frame: pd.DataFrame, road_classes: list[str] | None = None) -> tuple[np.ndarray, list[str]]:
    columns = [np.ones(len(frame)), np.log(np.maximum(frame.route_distance_m.to_numpy(float), 1.0))]
    names = ["intercept", "log_distance"]
    if road_classes:
        for road_class in road_classes[1:]:
            columns.append(frame.predominant_road_class.eq(road_class).to_numpy(float))
            names.append(f"class:{road_class}")
    return np.column_stack(columns), names


def robust_fit(x: np.ndarray, y: np.ndarray, iterations: int = 30) -> np.ndarray:
    beta = np.linalg.lstsq(x, y, rcond=None)[0]
    for _ in range(iterations):
        residual = y - x @ beta
        scale = max(np.median(np.abs(residual - np.median(residual))) * 1.4826, 1e-6)
        cutoff = 1.345 * scale
        weights = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1e-12))
        beta = np.linalg.lstsq(x * np.sqrt(weights[:, None]), y * np.sqrt(weights), rcond=None)[0]
    return beta


def spearman(a: list[float], b: list[float]) -> float:
    if len(a) < 3 or len(set(a)) < 2 or len(set(b)) < 2:
        return math.nan
    return float(np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1])


def evaluate(actual: np.ndarray, predicted: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(actual) & np.isfinite(predicted) & (actual > 0) & (predicted > 0)
    if not valid.any():
        return math.nan, math.nan
    error = float(np.median(np.abs(np.log(actual[valid]) - np.log(predicted[valid]))))
    correlation = spearman(actual[valid].tolist(), predicted[valid].tolist())
    return error, correlation


def main() -> None:
    started = time.perf_counter()
    metadata = pd.read_csv(SOURCE, dtype={"camera": "string"})
    metadata["camera"] = metadata.camera.str.strip()
    graph = ox.load_graphml(GRAPH)
    RESTRICTED.mkdir(parents=True, exist_ok=True)
    routes = build_routes(graph, metadata, RESTRICTED / "all_metadata_directed_routes.csv")
    if len(routes) != 233 * 232:
        raise RuntimeError(f"directed route matrix reconciliation failed: {len(routes)} != 54056")
    connection = duckdb.connect()
    observed = {row[0] for row in connection.execute(
        "SELECT camera FROM (SELECT DISTINCT CAST(origin_camera AS VARCHAR) camera FROM read_parquet(?) UNION "
        "SELECT DISTINCT CAST(destination_camera AS VARCHAR) camera FROM read_parquet(?))", [str(TRANSITIONS), str(TRANSITIONS)]).fetchall()}
    metadata_ids = set(metadata.camera)
    exact, unresolved, metadata_only = observed & metadata_ids, observed - metadata_ids, metadata_ids - observed
    anchor_path = RESTRICTED / "anchor_observed_time_summaries.csv"
    signature_path = RESTRICTED / "unresolved_observed_time_signatures.csv"
    if anchor_path.exists() and signature_path.exists():
        anchor_summary = pd.read_csv(anchor_path, dtype={"origin_camera": "string", "destination_camera": "string"})
        signatures = pd.read_csv(signature_path, dtype={"origin_camera": "string", "destination_camera": "string"})
    else:
        anchor_summary, signatures = observed_summaries(connection, exact, unresolved)
        atomic_csv(anchor_summary, anchor_path)
        atomic_csv(signatures, signature_path)
    connection.close()

    route_lookup = {(row.origin_metadata_id, row.destination_metadata_id): row for row in routes.itertuples(index=False)}
    calibration = anchor_summary.merge(routes, left_on=["origin_camera", "destination_camera"],
                                       right_on=["origin_metadata_id", "destination_metadata_id"], how="inner")
    inference = calibration[(calibration.split == "inference") & calibration.route_distance_m.notna() & (calibration.median_seconds > 0)].copy()
    validation_anchor = calibration[(calibration.split == "validation") & calibration.route_distance_m.notna() & (calibration.median_seconds > 0)].copy()
    classes = sorted(inference.predominant_road_class.unique())
    x_distance, distance_names = design_matrix(inference)
    beta_distance = robust_fit(x_distance, np.log(inference.median_seconds.to_numpy(float)))
    x_class, class_names = design_matrix(inference, classes)
    beta_class = robust_fit(x_class, np.log(inference.median_seconds.to_numpy(float)))
    speed_source = inference[(inference.nighttime_count >= 30) & inference.nighttime_median_seconds.notna()].copy()
    speed_source["freeflow_kph"] = speed_source.route_distance_m / speed_source.nighttime_median_seconds * 3.6
    class_speeds = speed_source.groupby("predominant_road_class").freeflow_kph.median().clip(10, 130).to_dict()
    global_speed = float(speed_source.freeflow_kph.median()) if len(speed_source) else 40.0

    def predict(frame: pd.DataFrame, model: str) -> np.ndarray:
        if model == "distance_robust":
            return np.exp(design_matrix(frame)[0] @ beta_distance)
        if model == "road_class_robust":
            return np.exp(design_matrix(frame, classes)[0] @ beta_class)
        if model == "class_freeflow":
            speeds = frame.predominant_road_class.map(class_speeds).fillna(global_speed).to_numpy(float)
            return frame.route_distance_m.to_numpy(float) / (speeds * 1000 / 3600)
        return frame.provisional_route_time_s.to_numpy(float)

    model_rows = []
    for name in ("distance_robust", "road_class_robust", "class_freeflow", "osm_provisional"):
        inf_pred, val_pred = predict(inference, name), predict(validation_anchor, name)
        inf_error, inf_rank = evaluate(inference.median_seconds.to_numpy(float), inf_pred)
        val_error, val_rank = evaluate(validation_anchor.median_seconds.to_numpy(float), val_pred)
        model_rows.append({"model": name, "inference_median_abs_log_error": inf_error,
                           "inference_rank_correlation": inf_rank, "validation_median_abs_log_error": val_error,
                           "validation_rank_correlation": val_rank, "calibration_pairs": len(inference),
                           "validation_pairs": len(validation_anchor)})
    model_results = pd.DataFrame(model_rows).sort_values(["inference_median_abs_log_error", "model"])
    chosen_model = str(model_results.iloc[0].model)
    route_predictions = predict(routes, chosen_model)
    predicted_lookup = {(origin, destination): float(prediction)
                        for origin, destination, prediction in zip(routes.origin_metadata_id,
                                                                    routes.destination_metadata_id,
                                                                    route_predictions)}

    metadata_index = metadata.set_index("camera")
    spatial = pd.read_csv(ROOT / "restricted/phase3b_camera_osm/topology_matching/metadata_spatial_signatures.csv", dtype={"metadata_camera_id": "string"}).set_index("metadata_camera_id")
    signature_groups = {(u, split): group for (u, split), group in signatures.assign(
        unresolved_camera=np.where(signatures.origin_camera.isin(unresolved), signatures.origin_camera, signatures.destination_camera)).groupby(["unresolved_camera", "split"])}

    score_rows = []
    for unresolved_id in sorted(unresolved, key=int):
        for metadata_id in sorted(metadata_only, key=int):
            split_metrics = {}
            for split in ("inference", "validation"):
                group = signature_groups.get((unresolved_id, split), pd.DataFrame())
                observed_times, expected_times, supports, within_ranges = [], [], [], []
                incoming = outgoing = 0
                anchors = set()
                supporting_weeks = []
                for row in group.itertuples(index=False):
                    if row.origin_camera == unresolved_id:
                        key = (metadata_id, row.destination_camera); outgoing += 1
                    else:
                        key = (row.origin_camera, metadata_id); incoming += 1
                    if key not in route_lookup:
                        continue
                    route = route_lookup[key]
                    if not np.isfinite(route.route_distance_m):
                        continue
                    expected = predicted_lookup[key]
                    observed_times.append(float(row.median_seconds)); expected_times.append(expected)
                    supports.append(int(row.transition_count)); anchors.add(row.destination_camera if row.origin_camera == unresolved_id else row.origin_camera)
                    supporting_weeks.append(int(row.supporting_weeks))
                    within_ranges.append(float(row.q10_seconds) <= expected <= float(row.q90_seconds))
                if observed_times:
                    errors = np.abs(np.log(np.maximum(observed_times, 1e-9)) - np.log(np.maximum(expected_times, 1e-9)))
                    robust_error = float(np.average(np.minimum(errors, 3), weights=np.sqrt(np.maximum(supports, 1))))
                else:
                    robust_error = math.nan
                split_metrics[split] = {"robust_log_error": robust_error,
                    "rank_correlation": spearman(observed_times, expected_times),
                    "agreement": float(np.mean(within_ranges)) if within_ranges else 0.0,
                    "supporting_anchors": len(anchors), "incoming_relations": incoming, "outgoing_relations": outgoing,
                    "supporting_observations": sum(supports),
                    "weekly_stability": min(1.0, sum(supporting_weeks) / (28 if split == "inference" else 7))}
            inf, val = split_metrics["inference"], split_metrics["validation"]
            direction = float(spatial.loc[metadata_id, "direction_compatibility"])
            feasibility = float(spatial.loc[metadata_id, "route_feasible_from_anchor_count"] + spatial.loc[metadata_id, "route_feasible_to_anchor_count"]) / (2 * len(exact))
            error_score = 0 if not np.isfinite(inf["robust_log_error"]) else math.exp(-inf["robust_log_error"])
            rank_score = .5 if not np.isfinite(inf["rank_correlation"]) else (inf["rank_correlation"] + 1) / 2
            evidence_score = min(1.0, inf["supporting_anchors"] / 20)
            score = (.35 * error_score + .15 * rank_score + .15 * inf["agreement"] + .10 * direction
                     + .10 * feasibility + .10 * inf["weekly_stability"] + .05 * evidence_score)
            score_rows.append({"observed_camera_id": unresolved_id, "metadata_camera_id": metadata_id,
                "score": score, "inference_robust_log_error": inf["robust_log_error"],
                "inference_rank_correlation": inf["rank_correlation"], "predecessor_successor_agreement": inf["agreement"],
                "direction_compatibility": direction, "oneway_route_feasibility": feasibility,
                "supporting_anchors": inf["supporting_anchors"], "incoming_relations": inf["incoming_relations"],
                "outgoing_relations": inf["outgoing_relations"], "supporting_observations": inf["supporting_observations"],
                "weekly_stability": inf["weekly_stability"], "validation_robust_log_error": val["robust_log_error"],
                "validation_rank_correlation": val["rank_correlation"], "validation_agreement": val["agreement"],
                "validation_supporting_anchors": val["supporting_anchors"], "calibration_model": chosen_model})
    scores = pd.DataFrame(score_rows).sort_values(["observed_camera_id", "score"], ascending=[True, False])
    scores["rank"] = scores.groupby("observed_camera_id").cumcount() + 1
    margins = scores.groupby("observed_camera_id").score.apply(lambda values: values.iloc[0] - values.iloc[1])
    scores["uniqueness_margin"] = scores.observed_camera_id.map(margins)
    top = scores[scores["rank"] <= 5].copy()
    classifications = []
    for row in top.itertuples(index=False):
        minimum = row.supporting_anchors >= 10 and row.incoming_relations > 0 and row.outgoing_relations > 0 and row.weekly_stability >= .5
        no_conflict = row.direction_compatibility >= .25 and row.oneway_route_feasibility >= .95
        heldout = np.isfinite(row.validation_robust_log_error) and row.validation_supporting_anchors >= 5
        if row.rank == 1 and minimum and no_conflict and heldout and row.score >= .78 and row.uniqueness_margin >= .10 and row.validation_robust_log_error <= .5:
            category = "high-confidence inferred"
        elif row.rank == 1 and minimum and no_conflict and heldout and row.score >= .68 and row.uniqueness_margin >= .05 and row.validation_robust_log_error <= .8:
            category = "moderate/manual review"
        elif not minimum:
            category = "insufficient evidence"
        elif not no_conflict:
            category = "rejected"
        else:
            category = "ambiguous"
        classifications.append(category)
    top["classification"] = classifications
    top["primary_dataset_status"] = "excluded"
    atomic_csv(top, REPORTS / "calibrated_camera_time_candidates.csv")

    best = top[top["rank"] == 1].copy()
    earlier = pd.read_csv(REPORTS / "topology_assisted_camera_validation.csv", dtype={"observed_camera_id": "string", "best_candidate_metadata_id": "string"})
    validation = best.merge(earlier[["observed_camera_id", "best_candidate_metadata_id", "classification"]],
                            on="observed_camera_id", how="left", suffixes=("", "_earlier"))
    validation["same_best_as_earlier"] = validation.metadata_camera_id.eq(validation.best_candidate_metadata_id)
    validation["material_improvement"] = ((validation.classification == "high-confidence inferred")
                                           & (validation.classification_earlier != "high-confidence inferred"))
    atomic_csv(validation, REPORTS / "calibrated_camera_time_validation.csv")
    review = top[top.classification.isin(["moderate/manual review", "ambiguous", "high-confidence inferred"])].copy()
    atomic_csv(review, REPORTS / "camera_candidate_manual_review.csv")

    high = int((best.classification == "high-confidence inferred").sum())
    moderate = int((best.classification == "moderate/manual review").sum())
    ambiguous = int((best.classification == "ambiguous").sum())
    rejected = int((best.classification == "rejected").sum())
    insufficient = int((best.classification == "insufficient evidence").sum())
    same_best = int(validation.same_best_as_earlier.sum())
    improved = int(validation.material_improvement.sum())
    runtime = time.perf_counter() - started
    coefficients_distance = ", ".join(f"{name}={value:.6g}" for name, value in zip(distance_names, beta_distance))
    coefficients_class = ", ".join(f"{name}={value:.6g}" for name, value in zip(class_names, beta_class))
    model_table_lines = ["| Model | Inference MALE | Inference rank | Validation MALE | Validation rank | Calibration pairs | Validation pairs |",
                         "|---|---:|---:|---:|---:|---:|---:|"]
    for row in model_results.itertuples(index=False):
        model_table_lines.append(f"| {row.model} | {row.inference_median_abs_log_error:.6f} | {row.inference_rank_correlation:.6f} | {row.validation_median_abs_log_error:.6f} | {row.validation_rank_correlation:.6f} | {row.calibration_pairs} | {row.validation_pairs} |")
    model_table = "\n".join(model_table_lines)
    report = f"""# Calibrated travel-time model and camera-signature audit

Run timestamp: {datetime.now(timezone.utc).isoformat()}

This audit refines but does not replace the 119 exact camera mappings. No inferred mapping is approved for the primary spatial dataset.

## Data separation and reconciliation

- Metadata cameras: 233
- Directed metadata pairs: {len(routes):,}, exactly 233 × 232 = 54,056
- Exact anchors: {len(exact)}
- Unresolved observed cameras: {len(unresolved)}
- Metadata-only candidates: {len(metadata_only)}
- Calibration/matching period: weeks 01–28
- Held-out validation period: weeks 29–35
- Anchor calibration pairs: {len(inference):,}; held-out anchor pairs: {len(validation_anchor):,}

Nighttime medians use the recorded naive clock hours 22:00–05:59 and require at least 30 observations for free-flow calibration. Because timezone remains unresolved, `nighttime` is an operational clock-time label rather than a confirmed local-time claim.

## Transparent calibration models

{model_table}

Chosen by lowest inference median absolute log error: **{chosen_model}**.

- Distance-only robust coefficients: {coefficients_distance}
- Road-class robust coefficients: {coefficients_class}
- Free-flow route classes calibrated: {len(class_speeds)}; global fallback speed: {global_speed:.3f} km/h
- Rank-based agreement is reported for every model rather than used as an identity claim by itself.

Robust regressions use iterative Huber weighting on log median time. The road-class model uses predominant shortest-route class. Provisional route time uses observed OSM maxspeed where parseable and transparent class defaults otherwise; it is a comparison model, not speed filtering.

## Candidate outcome

- High-confidence inferred: {high}
- Moderate/manual review: {moderate}
- Ambiguous: {ambiguous}
- Rejected: {rejected}
- Insufficient evidence: {insufficient}
- Same top candidate as earlier topology audit: {same_best}/{len(unresolved)}
- Material improvements to high confidence: {improved}

Each unresolved camera has exactly five reported candidates. Minimum evidence requires at least ten distinct anchors, incoming and outgoing evidence where available, multi-week stability, direction compatibility, directed route feasibility, held-out support, and a documented uniqueness margin. No one-to-one assignment is forced.

## Governance

All inferred candidates remain excluded from primary analysis. Results based on small uniqueness margins, direction conflicts, unreachable directed routes, weak held-out evidence, or insufficient anchors remain ambiguous, rejected, or insufficient. No traffic volume or transition similarity is accepted alone. No speed filtering was performed.

Runtime: {runtime:.3f} seconds. Restricted route and signature matrices remain outside Git.
"""
    atomic_text(report, REPORTS / "calibrated_travel_time_model.md")
    print(json.dumps({"directed_pairs": len(routes), "high": high, "moderate": moderate, "ambiguous": ambiguous,
                      "rejected": rejected, "insufficient": insufficient, "material_improvement": improved,
                      "runtime_seconds": runtime}, sort_keys=True))


if __name__ == "__main__":
    main()
