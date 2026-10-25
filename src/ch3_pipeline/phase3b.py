from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import geopandas as gpd
import networkx as nx
import pandas as pd
import yaml
from pyproj import Transformer
from shapely.geometry import LineString, Point


EVIDENCE_GRADES = {
    "A": "original verified metadata",
    "B": "coordinates recovered from valid legacy objects",
    "C": "externally mapped camera with defensible identifier match",
    "D": "spatially inferred candidate requiring manual validation",
    "E": "unresolved",
}
PROVENANCE_FIELDS = ("source_file", "source_type", "extraction_method", "evidence_grade")


@dataclass(frozen=True)
class CameraMetadata:
    camera_id: str
    latitude: float | None
    longitude: float | None
    original_crs: str | None
    direction: str | None = None
    valid_from: str | None = None
    valid_to: str | None = None
    road_link_id: str | None = None
    source_file: str = ""
    source_type: str = ""
    extraction_method: str = ""
    evidence_grade: str = "E"


def valid_lon_lat(longitude: float | None, latitude: float | None) -> bool:
    return longitude is not None and latitude is not None and -180 <= longitude <= 180 and -90 <= latitude <= 90


def transform_xy(x: float, y: float, source_crs: str, target_crs: str = "EPSG:4326") -> tuple[float, float]:
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    lon, lat = transformer.transform(x, y)
    if target_crs.upper() == "EPSG:4326" and not valid_lon_lat(lon, lat):
        raise ValueError("transformed coordinate outside longitude/latitude bounds")
    return lon, lat


def validate_metadata(records: list[CameraMetadata]) -> list[str]:
    errors: list[str] = []
    seen: set[str] = set()
    coordinates: dict[str, set[tuple[float, float]]] = {}
    for record in records:
        if record.camera_id in seen:
            errors.append(f"duplicate_camera_id:{record.camera_id}")
        seen.add(record.camera_id)
        if record.evidence_grade not in EVIDENCE_GRADES:
            errors.append(f"invalid_evidence_grade:{record.camera_id}")
        if record.latitude is not None or record.longitude is not None:
            if not valid_lon_lat(record.longitude, record.latitude):
                errors.append(f"invalid_coordinate:{record.camera_id}")
            else:
                coordinates.setdefault(record.camera_id, set()).add((record.longitude, record.latitude))
        if record.evidence_grade != "E" and not all(getattr(record, field) for field in PROVENANCE_FIELDS):
            errors.append(f"incomplete_provenance:{record.camera_id}")
    for camera_id, values in coordinates.items():
        if len(values) > 1:
            errors.append(f"multiple_coordinates:{camera_id}")
    return errors


def bearing_degrees(line: LineString) -> float:
    start = line.coords[0]; end = line.coords[-1]
    dx = end[0] - start[0]; dy = end[1] - start[1]
    return (math.degrees(math.atan2(dx, dy)) + 360) % 360


def nearest_edge_candidates(camera_points: gpd.GeoDataFrame, edges: gpd.GeoDataFrame,
                            k: int = 3, ambiguity_metres: float = 5.0) -> pd.DataFrame:
    if camera_points.crs is None or edges.crs is None:
        raise ValueError("camera and edge CRS are required")
    metric_crs = "EPSG:27700"
    cameras = camera_points.to_crs(metric_crs)
    roads = edges.to_crs(metric_crs)
    output: list[dict[str, Any]] = []
    for _, camera in cameras.iterrows():
        distances = roads.geometry.distance(camera.geometry).sort_values()
        selected = distances.head(k)
        near_tie = len(selected) > 1 and float(selected.iloc[1] - selected.iloc[0]) <= ambiguity_metres
        selected_rows = roads.loc[selected.index]
        refs = [str(value) for value in selected_rows.get("ref", pd.Series(index=selected.index, dtype=object)).dropna()]
        bearings = [bearing_degrees(geom) for geom in selected_rows.geometry]
        divided = len(set(refs)) == 1 and len(refs) > 1 and any(abs(((a-b+180)%360)-180) > 150 for a in bearings for b in bearings)
        for rank, (edge_index, distance) in enumerate(selected.items(), start=1):
            edge = roads.loc[edge_index]
            output.append({"camera_id":camera["camera_id"],"candidate_rank":rank,
                "edge_u":edge.get("u",edge_index[0] if isinstance(edge_index,tuple) else ""),
                "edge_v":edge.get("v",edge_index[1] if isinstance(edge_index,tuple) and len(edge_index)>1 else ""),
                "edge_key":edge.get("key",edge_index[2] if isinstance(edge_index,tuple) and len(edge_index)>2 else ""),
                "distance_to_edge_m":float(distance),"edge_bearing_deg":bearing_degrees(edge.geometry),
                "road_class":edge.get("highway"),"oneway":edge.get("oneway"),"maxspeed":edge.get("maxspeed"),
                "candidate_confidence":"manual_validation_required","junction_ambiguity":bool(near_tie),
                "divided_road_ambiguity":bool(divided),"evidence_grade":"D"})
    return pd.DataFrame(output)


def graph_quality(G: nx.MultiDiGraph) -> dict[str, Any]:
    edges = list(G.edges(data=True))
    maxspeed = sum(1 for _,_,data in edges if data.get("maxspeed") not in (None,""))
    oneway = sum(1 for _,_,data in edges if data.get("oneway") in (True,"yes","1"))
    classes: set[str] = set()
    length_total = 0.0
    for _,_,data in edges:
        value = data.get("highway")
        classes.update(value if isinstance(value,list) else [value] if value else [])
        length_total += float(data.get("length",0) or 0)
    return {"node_count":len(G.nodes),"directed_edge_count":len(edges),"oneway_edge_count":oneway,
        "oneway_edge_proportion":oneway/len(edges) if edges else 0,"road_class_count":len(classes),
        "road_classes":"|".join(sorted(classes)),"total_edge_length_m":length_total,
        "maxspeed_edge_count":maxspeed,"maxspeed_coverage":maxspeed/len(edges) if edges else 0,
        "graph_crs":str(G.graph.get("crs","unresolved"))}


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(16*1024*1024):
            digest.update(block)
    return digest.hexdigest()


def configure_historical_osm(snapshot: str, cache_folder: str | Path) -> None:
    """Configure OSMnx to make auditable Overpass attic queries."""
    import osmnx as ox
    ox.settings.use_cache = True
    ox.settings.cache_folder = str(cache_folder)
    ox.settings.requests_timeout = 600
    ox.settings.overpass_settings = f'[out:json][timeout:{{timeout}}][date:"{snapshot}"]{{maxsize}}'


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def save_csv(rows: list[dict[str, Any]] | pd.DataFrame, path: str | Path,
             columns: list[str] | None = None) -> None:
    frame = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if columns is not None:
        frame = frame.reindex(columns=columns)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
